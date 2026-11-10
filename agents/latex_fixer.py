from typing import Optional
import logging
import re

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config.config_loader import get_agent_config, local_model_kwargs

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
        self.llm = init_chat_model(model, temperature=temperature, **local_model_kwargs(model))
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




# Added to every generated document so that the PDF's text layer reads like the text version:
# ATS parsers and pdf text extractors merge words on justified lines whose spaces TeX squeezed,
# and split keywords at hyphenation points ("Kuber-\nnetes"). Ragged-right text without
# hyphenation avoids both, and glyph-to-Unicode maps keep ligatures and accents searchable.
_EXTRACTABLE_PREAMBLE = (
    "% Plain text layer for ATS parsers (added by the CV editor)\n"
    "\\IfFileExists{lmodern.sty}{\\usepackage{lmodern}}{}\n"
    "\\ifdefined\\pdfgentounicode\\input{glyphtounicode}\\pdfgentounicode=1\\fi\n"
)
_EXTRACTABLE_BODY = "\\raggedright\\hyphenpenalty=10000\\exhyphenpenalty=10000\n"


_DOCUMENTCLASS = re.compile(r"\\documentclass\s*(?:\[[^\]]*\])?\s*\{[^}]*\}")


def ensure_text_extractable(latex: str) -> str:
    """
    Make a LaTeX document's PDF text extract cleanly (no merged or hyphen-split words).

    Inserts a short preamble block right after \\documentclass, so any font package the document
    loads still wins over lmodern, and ragged-right, no-hyphenation settings right after
    \\begin{document}, unless the document already has them. Leaves text without a
    \\documentclass or \\begin{document} unchanged.
    """
    marker = "\\begin{document}"
    if not latex or marker not in latex or "\\pdfgentounicode" in latex:
        return latex
    head, body = latex.split(marker, 1)
    match = _DOCUMENTCLASS.search(head)
    if not match:
        return latex
    head = f"{head[:match.end()]}\n{_EXTRACTABLE_PREAMBLE}{head[match.end():].lstrip(chr(10))}"
    return f"{head.rstrip()}\n{marker}\n{_EXTRACTABLE_BODY}{body.lstrip(chr(10))}"


_UNESCAPED_DOLLAR = re.compile(r"(?<!\\)\$")
_PERCENT_AFTER_NUMBER = re.compile(r"(?<=\d)(\s?)(?<!\\)%")
_PLAIN_WORDS = re.compile(r"[^\W\d_]{2,}\s+[^\W\d_]{2,}|\d\s+[^\W\d_]{2,}|[^\W\d_]{2,}\s+\d")


def escape_text_specials(latex: str) -> str:
    """
    Escape % and $ that the LaTeX writer left in plain text, which compile without an error but
    silently change the PDF: "by 30% in 2024" drops everything after "30" (a comment), and
    "$38,000 to $24,000" is set as math, losing the dollar signs and the spaces between words.

    Only the document body is touched, and only cases that are plainly text: a % right after a
    number, and a pair of $ around ordinary words with no LaTeX command inside (real math such as
    $0$--$4\\,^\\circ$C is kept).
    """
    marker = "\\begin{document}"
    if not latex or marker not in latex:
        return latex
    head, body = latex.split(marker, 1)
    lines = []
    for line in body.split("\n"):
        if line.lstrip().startswith("%"):
            lines.append(line)
            continue
        line = _PERCENT_AFTER_NUMBER.sub(lambda m: f"{m.group(1)}\\%", line)
        positions = [m.start() for m in _UNESCAPED_DOLLAR.finditer(line)]
        escape = set()
        for a, b in zip(positions[::2], positions[1::2]):
            inside = line[a + 1:b]
            if "\\" not in inside and "^" not in inside and "_" not in inside and _PLAIN_WORDS.search(inside):
                escape.update((a, b))
        if len(positions) % 2 and line[positions[-1] + 1:positions[-1] + 2].isdigit():
            escape.add(positions[-1])  # a lone currency sign, e.g. "budget of $40,000"
        for pos in sorted(escape, reverse=True):
            line = line[:pos] + "\\" + line[pos:]
        lines.append(line)
    return head + marker + "\n".join(lines)
