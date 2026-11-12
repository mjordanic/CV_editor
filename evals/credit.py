"""Out-of-credit handling for testing mode, the evaluator and the matrix.

When the OpenAI account runs out of credit, every further call fails with HTTP 429 and the error code
`insufficient_quota` (or a billing error). Anthropic answers a used-up balance with "Your credit balance is
too low" (or a `billing_error`), and a reached spend cap with `enforced_spend_limit_reached`. That is not an ordinary rate limit: waiting and retrying
cannot help, and a session that goes on would only produce degraded, unscorable output. So:

- `is_out_of_credit(error)` tells the two apart (code `insufficient_quota` / billing codes, or the
  quota and billing wording of the message), also through chained exceptions.
- `OutOfCredit` is a BaseException, like KeyboardInterrupt: the app's agents catch `Exception` and
  `with_retry` retries `Exception`, so neither swallows nor retries it; it ends the session at once.
- `install_no_retry_patch()` makes the OpenAI client skip its own retries for an out-of-credit 429
  (ordinary 429s keep the client's backoff).
- `CreditGuard` is a LangChain callback: it turns an out-of-credit model error into `OutOfCredit`,
  and once tripped it stops any further model call before it is sent.

run_case and evaluate exit with `EXIT_OUT_OF_CREDIT` and mark their records `interrupted: out of
credit`; run_matrix stops starting new work when it sees that exit code (see evals/README.md).

`is_infrastructure_error(error)` tells a provider or network failure (connection error, timeout, HTTP 5xx, an
ordinary rate limit; the SDK's own retries already ran) from a failure of the system under test. run_case and
single_prompt mark such a session `interrupted: infrastructure`: it is not graded, and run_matrix re-runs it once.
A session stopped by hand (Ctrl+C or SIGTERM) is marked `interrupted: stopped` by run_case: not graded either, and
redone from scratch on the next run.
"""
from __future__ import annotations

import threading
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

EXIT_OUT_OF_CREDIT = 75  # EX_TEMPFAIL: try again later (after a top-up)
INTERRUPTED_STATUS = "interrupted: out of credit"
INFRASTRUCTURE_STATUS = "interrupted: infrastructure"
STOPPED_STATUS = "interrupted: stopped"  # Ctrl+C or SIGTERM: not graded, redone on the next run

# OpenAI codes, then Anthropic's: billing_error (HTTP 402) and enforced_spend_limit_reached (a workspace or
# organisation spend cap); an exhausted Anthropic prepaid balance says "Your credit balance is too low".
_CREDIT_CODES = {"insufficient_quota", "billing_hard_limit_reached", "billing_not_active", "account_deactivated",
                 "credit_balance_too_low", "insufficient_credits", "billing_error", "enforced_spend_limit_reached"}
_CREDIT_PHRASES = ("insufficient_quota", "exceeded your current quota", "check your plan and billing",
                   "billing hard limit", "billing_hard_limit", "credit balance", "insufficient credit",
                   "out of credit", "billing details", "spend limit")


class OutOfCredit(BaseException):
    """The model provider refused a call because the account has no credit left."""


def _codes(exc: Any) -> list[str]:
    out = []
    for attr in ("code", "type", "error_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, str):
            out.append(value)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") if isinstance(body.get("error"), dict) else body
        for key in ("code", "type"):
            if isinstance(err.get(key), str):
                out.append(err[key])
    return out


def is_out_of_credit(error: Any) -> bool:
    """True for an exhausted-credit / billing error (an exception, its chain, or an error text);
    False for an ordinary rate limit and anything else."""
    if error is None:
        return False
    if isinstance(error, OutOfCredit):
        return True
    if isinstance(error, str):
        text = error.lower()
        return any(p in text for p in _CREDIT_PHRASES)
    seen = set()
    exc = error
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, OutOfCredit) or any(c in _CREDIT_CODES for c in _codes(exc)):
            return True
        if any(p in str(exc).lower() for p in _CREDIT_PHRASES):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def is_infrastructure_error(error: BaseException) -> bool:
    """True for an OpenAI connection error, timeout, HTTP 5xx or a rate limit that is not out of credit (also
    through chained exceptions): it says nothing about the system under test."""
    import openai
    seen = set()
    exc = error
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, (openai.APIConnectionError, openai.InternalServerError)):
            return True
        if isinstance(exc, openai.RateLimitError) and not is_out_of_credit(exc):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def _response_is_out_of_credit(response: Any) -> bool:
    if getattr(response, "status_code", None) not in (402, 429):
        return False
    try:
        response.read()
        return is_out_of_credit(response.text)
    except Exception:  # noqa: BLE001 - an unreadable body is treated as an ordinary error
        return False


_PATCH_LOCK = threading.Lock()


def install_no_retry_patch() -> bool:
    """Make the OpenAI client not retry an out-of-credit 429 (other retries unchanged). Idempotent."""
    try:
        from openai._base_client import SyncAPIClient
    except Exception:  # noqa: BLE001 - no openai package: nothing to patch
        return False
    with _PATCH_LOCK:
        if getattr(SyncAPIClient._should_retry, "_credit_patched", False):
            return True
        original = SyncAPIClient._should_retry

        def _should_retry(self, response):
            if _response_is_out_of_credit(response):
                return False
            return original(self, response)

        _should_retry._credit_patched = True
        SyncAPIClient._should_retry = _should_retry
    return True


class CreditGuard(BaseCallbackHandler):
    """Callback for every model call of a run: raises OutOfCredit on the first exhausted-credit error,
    and before any later call once tripped (so nothing else is sent)."""

    raise_error = True

    def __init__(self):
        self.tripped: str | None = None

    def _check(self):
        if self.tripped:
            raise OutOfCredit(self.tripped)

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self._check()

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._check()

    def on_llm_error(self, error, **kwargs):
        if is_out_of_credit(error):
            self.tripped = self.tripped or f"{type(error).__name__}: {error}"[:500]
            raise OutOfCredit(self.tripped) from error

    def guard(self, error: BaseException) -> None:
        """For calls outside LangChain callbacks (embeddings): re-raise a credit error as OutOfCredit."""
        if is_out_of_credit(error):
            self.tripped = self.tripped or f"{type(error).__name__}: {error}"[:500]
            raise OutOfCredit(self.tripped) from error
