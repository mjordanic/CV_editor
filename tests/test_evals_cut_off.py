"""Offline: an OpenAI grading cut off at the output limit (openai.LengthFinishReasonError) is billed, so its cost is
recorded (evals/grader.py). The chat model is a stub that raises the SDK's error; nothing is sent."""
import sys
from pathlib import Path

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


class Answer(BaseModel):
    score: int


class CutOffChat(BaseChatModel):
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "cut-off"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        from tests.test_single_prompt import cut_off_error
        raise cut_off_error("gpt-6.1-sol", 8000, 20000)

    def with_structured_output(self, schema, include_raw=False, **kwargs):
        return self


def test_a_grading_cut_off_at_the_output_limit_is_billed_and_asked_once_more(tmp_path):
    from evals.grader import GraderSpec, build_request, send
    from provenance import RunRecorder, cost_usd
    request = build_request(GraderSpec("openai", "gpt-6.1-sol"), "system", "shared", "documents", Answer)
    recorder = RunRecorder(tmp_path / "grading")
    llm = CutOffChat()
    result = send(request, recorder, llm=llm)
    billed = cost_usd("gpt-6.1-sol", 8000, 20000)
    assert llm.calls == 2 and result.status == "parse_error" and result.answer is None
    assert [a["stop_reason"] for a in result.attempts] == ["length", "length"]
    assert result.cost_usd == pytest.approx(2 * billed, abs=1e-6)
    assert recorder.cost == pytest.approx(2 * billed)  # summary.json's cost, which run_matrix puts in the ledger
