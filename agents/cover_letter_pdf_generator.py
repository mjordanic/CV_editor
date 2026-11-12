import logging
import sys
from pathlib import Path

# Add project root to path when running as standalone script
if __file__ and Path(__file__).parent.name == "agents":
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from pydantic import BaseModel, Field

from agents.latex_pdf_generator import LatexPDFGenerator

logger = logging.getLogger(__name__)


COVER_LETTER_LATEX_SYSTEM_PROMPT = (
    "You are an expert LaTeX cover letter formatter. "
    "Your task is to take an existing LaTeX cover letter document, a plain-text cover letter, "
    "and style preferences, and produce a COMPLETE LaTeX cover letter document that is ready "
    "to compile with pdflatex.\n\n"
    "CRITICAL RULES:\n"
    "- You MUST output a full LaTeX document, including the preamble (\\documentclass, \\usepackage, "
    "  color definitions, custom commands, and \\begin{{document}} ... \\end{{document}}).\n"
    "- Start from the existing LaTeX document provided to you. You may rewrite or simplify it as needed, "
    "  but keep any working structure that is still appropriate.\n"
    "- Apply the style_preferences to BOTH the preamble (colors, spacing, fonts, layout) and the content "
    "  (paragraph structure, emphasis) where appropriate.\n"
    "- Copy the text of the plain-text cover letter word for word, in the same order and language. Change only the "
    "  LaTeX markup and layout: do NOT reword, add, drop or reorder sentences, and do NOT change names or "
    "  accented letters (that text was already checked against the candidate's material). Escaping a special "
    "  character is not a change.\n"
    "- Escape LaTeX special characters where needed (%, _, $, &, #, {{, }}, ~, ^, \\).\n"
    "- Keep the letter professional, readable, and well-structured.\n"
)

COVER_LETTER_LATEX_HUMAN_PROMPT = (
    "You are given:\n\n"
    "1) The CURRENT LaTeX COVER LETTER DOCUMENT (starting point):\n"
    "-----------------------------------------------\n"
    "{existing_latex}\n"
    "-----------------------------------------------\n\n"
    "2) The plain-text cover letter content (ground truth for the letter text):\n"
    "--------------------------------------\n"
    "{cover_letter_text}\n"
    "--------------------------------------\n\n"
    "3) PDF style preferences (APPEARANCE ONLY, no factual content changes):\n"
    "--------------------------------------------------------------------------\n"
    "{style_description}\n"
    "--------------------------------------------------------------------------\n\n"
    "Your job is to produce a COMPLETE LaTeX cover letter document that:\n"
    "- Keeps or improves the existing layout and structure when it makes sense.\n"
    "- Applies the style preferences (e.g., colors, spacing, typography, margins).\n"
    "- Contains the plain-text cover letter word for word (only LaTeX markup and layout change).\n\n"
    "IMPORTANT:\n"
    "- You MUST output a single, self-contained LaTeX document that can be compiled with pdflatex.\n"
    "- You may modify or remove existing commands and add new ones as needed, as long as the document compiles.\n"
    "- Do NOT include any explanatory comments or prose—ONLY LaTeX code.\n"
    "- Assume the color and package setup from the existing document can be adjusted as needed.\n\n"
    "Return exactly one field:\n"
    "- latex_document: the full LaTeX source for the updated cover letter.\n"
)


class CoverLetterLatexDocumentResponse(BaseModel):
    """Structured response for cover letter LaTeX full document generation."""

    latex_document: str = Field(
        ...,
        description="Complete LaTeX document for the cover letter, including preamble and document environment.",
    )


class CoverLetterPDFGenerator(LatexPDFGenerator):
    """Generate cover letter PDFs from plain-text letters using an LLM + LaTeX template."""

    kind = "cover_letter"
    label = "cover letter"
    file_stem = "generated_cover_letter"
    text_variable = "cover_letter_text"
    system_prompt = COVER_LETTER_LATEX_SYSTEM_PROMPT
    human_prompt = COVER_LETTER_LATEX_HUMAN_PROMPT
    response_model = CoverLetterLatexDocumentResponse

    def __init__(
        self,
        template_path: str = "templates/cover_letter_template.tex",
        output_folder: str = "data/generated_CVs",
        temp_folder: str = "data/generated_CVs/temp_latex",
        model: str = "openai:gpt-5-nano",
        temperature: float = 0.1,
    ):
        super().__init__(template_path, output_folder, temp_folder, model, temperature)
