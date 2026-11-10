"""What the CV and cover letter PDF generators share: LLM -> LaTeX -> pdflatex, then a text check.

A model turns the checked plain text into a full LaTeX document (and a second model may repair it
when it does not compile). The critic and the invented-claims check validated the plain text, but
the employer and the ATS read the PDF, so after compiling the PDF's text is compared word by word
with the plain text (agents/pdf_text_check.py). When they differ, the document is converted once
more from the base template and the closer PDF is kept; if it still differs, the user is told.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from config.config_loader import load_config, local_model_kwargs
from agents.latex_fixer import LatexFixer, ensure_text_extractable, escape_text_specials
from agents.notices import NOTICES_KEY, set_notices
from agents.pdf_text_check import check_pdf

logger = logging.getLogger(__name__)


class LatexPDFGenerator:
    """Base class of CVPDFGenerator and CoverLetterPDFGenerator."""

    kind: str = ""                # state key prefix: "cv" or "cover_letter"
    label: str = ""               # "CV" or "cover letter"
    file_stem: str = ""           # "generated_CV" or "generated_cover_letter"
    text_variable: str = ""       # the human prompt's variable for the plain text
    system_prompt: str = ""
    human_prompt: str = ""
    response_model: type[BaseModel] = BaseModel

    def __init__(
        self,
        template_path: str,
        output_folder: str = "data/generated_CVs",
        temp_folder: str = "data/generated_CVs/temp_latex",
        model: str = "openai:gpt-5-nano",
        temperature: float = 0.1,
    ):
        logger.info(f"Initializing {type(self).__name__} with template: {template_path}")
        self.template_path = template_path
        self.output_folder = output_folder
        self.temp_folder = temp_folder
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.temp_folder, exist_ok=True)
        if not os.path.exists(self.template_path):
            logger.error(f"{self.label} template not found at {self.template_path}")
            raise FileNotFoundError(f"{self.label} template not found at {self.template_path}")

        self._content_llm = init_chat_model(model, temperature=temperature, **local_model_kwargs(model))
        self._prompt = ChatPromptTemplate.from_messages([("system", self.system_prompt), ("human", self.human_prompt)])
        self._content_chain = self._prompt | self._content_llm.with_structured_output(self.response_model)
        self._latex_fixer = LatexFixer()
        try:
            self.max_latex_fix_iterations = load_config().get("workflow", {}).get("max_latex_fix_iterations", 1)
        except Exception as e:
            logger.warning(f"Failed to load max_latex_fix_iterations from config: {e}. Using default: 1")
            self.max_latex_fix_iterations = 1

    # ------------------------------------------------------------------ LaTeX
    def _load_template(self) -> str:
        with open(self.template_path, "r", encoding="utf-8") as f:
            return f.read()

    def _build_latex_document(self, template_content: str, text: str, style_description: Optional[str]) -> str:
        """Use the LLM to produce a full LaTeX document, starting from the given template."""
        style_for_llm = (style_description or "").strip() or "No special style preferences."
        logger.info(f"Generating LaTeX {self.label} content using LLM (style: {style_for_llm[:100]})")
        response = self._content_chain.with_retry(stop_after_attempt=2).invoke({
            "existing_latex": template_content,
            self.text_variable: text,
            "style_description": style_for_llm,
        })
        return (response.latex_document or "").strip()

    def _compile_latex(self, latex_content: str, output_name: str, max_fix_iterations: int = 1) -> tuple[str, int, str | None]:
        """
        Compile LaTeX content to PDF, letting the LaTeX fixer repair it up to max_fix_iterations times.

        Returns:
            (path to the PDF in the output folder, number of fix attempts, None)

        Raises:
            RuntimeError: if it does not compile; the message holds the last compiler output.
        """
        latex_content = escape_text_specials(ensure_text_extractable(latex_content))
        tex_file = os.path.join(self.temp_folder, f"{output_name}.tex")
        tex_filename = f"{output_name}.tex"
        pdf_file = os.path.join(self.temp_folder, f"{output_name}.pdf")
        final_pdf = os.path.join(self.output_folder, f"{output_name}.pdf")

        def _run_pdflatex() -> subprocess.CompletedProcess:
            # cwd is the temp folder, so every output file (.log, .aux, .pdf) is written there.
            return subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", tex_filename],
                capture_output=True,
                text=True,
                # pdflatex echoes text from the document in 8-bit form (e.g. "\xe4" for "ä" in an
                # overfull-box warning), which is not UTF-8; strict decoding raised and the PDF failed
                # before the LaTeX fixer could run.
                encoding="utf-8",
                errors="replace",
                cwd=self.temp_folder,
            )

        def _error_output(result) -> str:
            # LaTeX errors typically go to stdout; keep stderr too
            output = result.stdout if result.stdout else result.stderr
            if result.stderr and result.stderr != result.stdout:
                output = f"{result.stdout}\n\nSTDERR:\n{result.stderr}"
            return output

        fix_attempts = 0
        try:
            os.makedirs(self.temp_folder, exist_ok=True)
            if not os.access(self.temp_folder, os.W_OK):
                raise RuntimeError(f"Temp folder is not writable: {self.temp_folder}")
            with open(tex_file, "w", encoding="utf-8") as f:
                f.write(latex_content)
            logger.info(f"Compiling LaTeX to PDF: {tex_file} (working directory: {self.temp_folder})")
            result = _run_pdflatex()

            while result.returncode != 0 and fix_attempts < max_fix_iterations:
                fix_attempts += 1
                error_output = _error_output(result)
                logger.warning(f"pdflatex compilation failed (attempt {fix_attempts}) with code {result.returncode}")
                logger.debug(f"pdflatex output (attempt {fix_attempts}): {error_output}")
                fixed_latex = self._latex_fixer.fix(latex_content, error_output)
                if not fixed_latex or fixed_latex == latex_content:
                    logger.warning("LatexFixer did not produce a different LaTeX document. Stopping further fix attempts.")
                    break
                logger.info("LatexFixer produced an updated LaTeX document. Retrying compilation.")
                latex_content = fixed_latex.encode("utf-8", errors="replace").decode("utf-8")
                with open(tex_file, "w", encoding="utf-8", errors="replace") as f:
                    f.write(latex_content)
                result = _run_pdflatex()

            if result.returncode != 0:
                final_error = _error_output(result)
                logger.error(f"LaTeX compilation failed after {fix_attempts} fix attempts: {final_error}")
                raise RuntimeError(f"Failed to compile LaTeX: {final_error}")

            # A second pass resolves references; its warnings are ignored unless it fails.
            second_result = _run_pdflatex()
            if second_result.returncode != 0:
                logger.warning(f"Second pdflatex compilation returned non-zero code: {second_result.returncode}")

            if not os.path.exists(pdf_file):
                raise RuntimeError("PDF file was not created")
            shutil.move(pdf_file, final_pdf)
            logger.info(f"PDF generated successfully: {final_pdf}")
            return final_pdf, fix_attempts, None
        except FileNotFoundError:
            logger.error("pdflatex not found. Please install a LaTeX distribution (e.g., TeX Live, MiKTeX)")
            raise RuntimeError("LaTeX compiler (pdflatex) not found. Please install a LaTeX distribution.")

    # ------------------------------------------------------------------ public helpers
    def generate_pdf(self, text: str, output_filename: Optional[str] = None, style_description: Optional[str] = None,
                     template_content: Optional[str] = None) -> str:
        """Generate a PDF from plain text (outside the graph); returns its path."""
        output_filename = output_filename or f"{self.file_stem}.pdf"
        latex_document = self._build_latex_document(template_content or self._load_template(), text, style_description)
        pdf_path, _, _ = self._compile_latex(latex_document, os.path.splitext(output_filename)[0],
                                             max_fix_iterations=self.max_latex_fix_iterations)
        return pdf_path

    def generate_pdf_from_file(self, txt_file_path: str, output_filename: Optional[str] = None,
                               style_description: Optional[str] = None, template_content: Optional[str] = None) -> str:
        """Generate a PDF from a plain-text file (defaults to the same name with .pdf)."""
        with open(txt_file_path, "r", encoding="utf-8") as f:
            text = f.read()
        if output_filename is None:
            output_filename = f"{os.path.splitext(os.path.basename(txt_file_path))[0]}.pdf"
        return self.generate_pdf(text, output_filename, style_description=style_description, template_content=template_content)

    # ------------------------------------------------------------------ the PDF node
    def _render_checked(self, text: str, template_content: str, style_description: Optional[str],
                        output_name: str, max_fix_iterations: int) -> tuple[str, int, dict]:
        """Render, compare the PDF's words with the text, and convert once more from the base template if they differ."""
        latex_document = self._build_latex_document(template_content, text, style_description)
        pdf_path, fix_attempts, _ = self._compile_latex(latex_document, output_name, max_fix_iterations)
        check = check_pdf(pdf_path, text)
        if check.get("ok") is not False:
            return pdf_path, fix_attempts, check

        logger.warning(f"{self.label} PDF text differs from the checked text ({check.get('text_words_found_in_pdf_pct')}% "
                       f"of its words found, {check.get('extra_word_count')} added) - converting it once more")
        retry_name = f"{output_name}_retry"
        try:
            retry_latex = self._build_latex_document(self._load_template(), text, style_description)
            retry_path, retry_fixes, _ = self._compile_latex(retry_latex, retry_name, max_fix_iterations)
        except Exception as e:
            logger.warning(f"Second conversion failed ({type(e).__name__}: {e}) - keeping the first PDF")
            return pdf_path, fix_attempts, check
        retry_check = check_pdf(retry_path, text)
        better = retry_check.get("ok") or (
            (retry_check.get("text_words_found_in_pdf_pct") or 0, -(retry_check.get("extra_word_count") or 0))
            > (check.get("text_words_found_in_pdf_pct") or 0, -(check.get("extra_word_count") or 0))
        )
        if better:
            # Keep the closer PDF and its .tex under the draft's own name (a style update reuses that .tex).
            shutil.move(retry_path, pdf_path)
            retry_tex = os.path.join(self.temp_folder, f"{retry_name}.tex")
            if os.path.exists(retry_tex):
                shutil.move(retry_tex, os.path.join(self.temp_folder, f"{output_name}.tex"))
            return pdf_path, fix_attempts + retry_fixes, {**retry_check, "converted_twice": True}
        if os.path.exists(retry_path):
            os.remove(retry_path)
        return pdf_path, fix_attempts, {**check, "converted_twice": True}

    def run(self, state):
        """Render the final text of the document in state to PDF (named after the draft it holds)."""
        kind, label = self.kind, self.label
        title = label[0].upper() + label[1:]
        logger.info(f"{type(self).__name__}.run() called")
        text = state.get(f"generated_{kind}")
        if not text:
            logger.warning(f"No generated {label} found in state")
            return {
                f"{kind}_pdf_path": None,
                "messages": state.get("messages", []) + [{"role": "assistant", "content": f"No {label} available to generate PDF."}],
            }

        try:
            history = state.get(f"{kind}_history", []) or []
            # Name the PDF after the draft it renders: finalize may pick an earlier draft than
            # the latest one, and the file name must say which.
            iteration = state.get(f"{kind}_final_iteration") or len(history) or 1
            output_name = f"{self.file_stem}_draft_{iteration}"
            style_description = state.get(f"{kind}_pdf_style")
            logger.info(f"{title} PDF style from state: {style_description or '(none)'}")

            # Reuse this draft's .tex if it was rendered before (a style-only update); else the base template.
            template_content = None
            last_tex_path = os.path.join(self.temp_folder, f"{output_name}.tex")
            if history and os.path.exists(last_tex_path):
                logger.info(f"Using previous generated {label} .tex as template: {last_tex_path}")
                with open(last_tex_path, "r", encoding="utf-8") as f:
                    template_content = f.read()

            max_fix_iterations = state.get("max_latex_fix_iterations", self.max_latex_fix_iterations)
            pdf_path, fix_attempts, check = self._render_checked(
                text, template_content or self._load_template(), style_description, output_name, max_fix_iterations,
            )

            # Always copy the latest draft to the final name so it's always the latest version
            final_pdf_path = os.path.join(self.output_folder, f"{self.file_stem}.pdf")
            shutil.copy2(pdf_path, final_pdf_path)
            pages = f", {check['pages']} page{'s' if check['pages'] != 1 else ''}" if check.get("pages") else ""
            message = f"{title} PDF generated successfully: {pdf_path} (also saved as {self.file_stem}.pdf{pages})"
            updates = {}
            notice = None
            if check.get("ok") is False:
                details = []
                if check.get("missing_words"):
                    details.append("missing from the PDF: " + ", ".join(check["missing_words"][:10]))
                if check.get("extra_words"):
                    details.append("in the PDF but not in the checked text: " + ", ".join(check["extra_words"][:10]))
                if check.get("number_changes"):
                    details.append("numbers that differ (- missing, + added): " + ", ".join(check["number_changes"][:10]))
                notice = (f"Warning: the {label} PDF does not match the checked text exactly "
                          f"({check.get('text_words_found_in_pdf_pct')}% of its words found"
                          + (f"; {'; '.join(details)}" if details else "") + "). "
                          f"Compare the PDF with {os.path.join(self.output_folder, self.file_stem + '.txt')} before you send it.")
                message += "\n\n" + notice
            # Set or cleared on every PDF of this document, so a warning about an earlier PDF does not outlive it.
            updates[NOTICES_KEY] = set_notices(state, updates, {f"{kind}_pdf": notice})
            logger.info(message)
            return {
                **updates,
                f"{kind}_pdf_path": final_pdf_path,  # Return the final path, not the draft path
                f"{kind}_pdf_check": check,
                f"{kind}_latex_fix_attempts": fix_attempts,
                f"{kind}_latex_last_error": None,
                "messages": state.get("messages", []) + [{"role": "assistant", "content": message}],
            }
        except Exception as e:
            error_message = f"Error generating {label} PDF: {str(e)}"
            logger.error(error_message)
            return {
                f"{kind}_pdf_path": None,
                f"{kind}_latex_fix_attempts": state.get(f"{kind}_latex_fix_attempts", 0),
                f"{kind}_latex_last_error": str(e),
                "messages": state.get("messages", []) + [{"role": "assistant", "content": error_message}],
            }
