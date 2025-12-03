from typing import Optional
import logging

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config.config_loader import get_agent_config

logger = logging.getLogger(__name__)


LATEX_FIX_SYSTEM_PROMPT = (
    "You are an expert LaTeX engineer. Your job is to fix LaTeX documents that fail to compile.\n\n"
    "CRITICAL RULES:\n"
    "- Preserve the original document's semantic content and structure as much as possible.\n"
    "- Fix ONLY what is necessary for successful compilation (e.g., unmatched braces, bad characters, "
    "  malformed commands, incorrect environments).\n"
    "- Do NOT remove large chunks of content unless they are clearly causing errors and cannot be safely fixed.\n"
    "- Do NOT add preamble commands that conflict with the existing document.\n"
    "- You MUST return a complete, compilable LaTeX document.\n"
)

LATEX_FIX_HUMAN_PROMPT = (
    "The following LaTeX document failed to compile.\n\n"
    "===== COMPILER ERROR OUTPUT (excerpt) =====\n"
    "{error_log}\n"
    "===========================================\n\n"
    "===== ORIGINAL LATEX DOCUMENT =====\n"
    "{latex_source}\n"
    "===================================\n\n"
    "Task:\n"
    "- Fix the LaTeX so that it compiles successfully.\n"
    "- Preserve the content and layout as much as possible.\n"
    "- Do NOT introduce new packages unless clearly necessary.\n"
    "- Prefer minimal edits over large rewrites.\n\n"
    "Return ONLY the full, fixed LaTeX document."
)


class LatexFixResponse(BaseModel):
    """Structured response for LaTeX fixing."""

    fixed_latex: str = Field(
        ...,
        description="The full LaTeX document with minimal changes applied so that it compiles.",
    )


class LatexFixer:
    """Small helper agent that repairs LaTeX documents based on compiler errors."""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        """
        Initialize the LatexFixer.

        If model/temperature are not provided, they are loaded from the
        `latex_fixer` entry in `config/agent_models.yaml`.
        """
        if model is None or temperature is None:
            cfg = get_agent_config("latex_fixer", default_model="openai:gpt-5-nano", default_temperature=0.1)
            model = model or cfg.get("model", "openai:gpt-5-nano")
            temperature = temperature if temperature is not None else cfg.get("temperature", 0.1)

        logger.info(f"Initializing LatexFixer with model: {model}")
        self.llm = init_chat_model(model, temperature=temperature)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", LATEX_FIX_SYSTEM_PROMPT),
                ("human", LATEX_FIX_HUMAN_PROMPT),
            ]
        )
        structured_llm = self.llm.with_structured_output(LatexFixResponse)
        self._chain = prompt | structured_llm

    def fix(self, latex_source: str, error_log: str) -> str:
        """Return a minimally edited LaTeX document that should compile."""
        logger.info("LatexFixer: attempting to fix LaTeX compilation errors")
        logger.debug(
            f"Original LaTeX length: {len(latex_source)}, error log length: {len(error_log)}"
        )

        try:
            response: LatexFixResponse = self._chain.invoke(
                {"latex_source": latex_source, "error_log": error_log}
            )
            fixed = response.fixed_latex or ""
            logger.info(
                f"LatexFixer: produced fixed LaTeX (length: {len(fixed)} characters)"
            )
            return fixed
        except Exception as e:
            logger.error(f"LatexFixer: error while fixing LaTeX: {e}")
            # In case of failure, return original source so caller can decide what to do
            return latex_source


