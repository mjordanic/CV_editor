from typing import Optional
import logging
import os
import subprocess
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.latex_fixer import LatexFixer
from config.config_loader import load_config

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
    "- Keep the factual content faithful to the plain-text cover letter, but you may improve formatting, "
    "  ordering, and wording for clarity and professionalism.\n"
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
    "- Preserves the factual content from the plain-text cover letter, but may reorganize/rephrase for clarity.\n\n"
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


class CoverLetterPDFGenerator:
    """Generate cover letter PDFs from plain-text letters using an LLM + LaTeX template."""

    def __init__(
        self,
        template_path: str = "templates/cover_letter_template.tex",
        output_folder: str = "data/generated_CVs",
        temp_folder: str = "data/generated_CVs/temp_latex",
        model: str = "openai:gpt-5-nano",
        temperature: float = 0.1,
    ):
        """
        Initialize the CoverLetterPDFGenerator.
        
        Args:
            template_path: Path to the LaTeX template file
            output_folder: Folder where generated PDFs will be saved
            temp_folder: Temporary folder for LaTeX compilation
            model: LLM model identifier for LaTeX content generation
            temperature: Temperature for the content LLM
        """
        logger.info(f"Initializing CoverLetterPDFGenerator with template: {template_path}")
        
        self.template_path = template_path
        self.output_folder = output_folder
        self.temp_folder = temp_folder
        
        # Ensure directories exist
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.temp_folder, exist_ok=True)
        
        # Verify template exists
        if not os.path.exists(self.template_path):
            logger.error(f"Cover letter template not found at {self.template_path}")
            raise FileNotFoundError(f"Cover letter template not found at {self.template_path}")
        
        # Initialize LLM for LaTeX content
        logger.info(f"Initializing cover letter content LLM - model: {model}")
        self._content_llm = init_chat_model(model, temperature=temperature)
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", COVER_LETTER_LATEX_SYSTEM_PROMPT),
                ("human", COVER_LETTER_LATEX_HUMAN_PROMPT),
            ]
        )
        structured_llm = self._content_llm.with_structured_output(CoverLetterLatexDocumentResponse)
        self._content_chain = self._prompt | structured_llm
        # Helper for repairing LaTeX compilation errors (config-driven)
        self._latex_fixer = LatexFixer()
        
        # Load max_latex_fix_iterations from config
        try:
            config = load_config()
            self.max_latex_fix_iterations = config.get("workflow", {}).get("max_latex_fix_iterations", 1)
        except Exception as e:
            logger.warning(f"Failed to load max_latex_fix_iterations from config: {e}. Using default: 1")
            self.max_latex_fix_iterations = 1
        
        logger.debug(
            f"CoverLetterPDFGenerator initialized - template: {self.template_path}, "
            f"output_folder: {self.output_folder}, temp_folder: {self.temp_folder}"
        )
    
    def _load_template(self) -> str:
        """Load the LaTeX template from disk."""
        try:
            with open(self.template_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading cover letter template: {e}")
            raise

    def _build_latex_document(
        self,
        template_content: str,
        cover_letter_text: str,
        style_description: Optional[str],
    ) -> str:
        """Use the LLM to produce a full LaTeX document, starting from the given template."""
        style_description = (style_description or "").strip()

        logger.info("Generating LaTeX cover letter content using LLM...")
        logger.debug(
            f"Plain-text cover letter length: {len(cover_letter_text)} characters, "
            f"style description length: {len(style_description)}"
        )
        
        # Prepare style description for LLM
        style_for_llm = style_description or "No special style preferences."
        logger.info(f"Passing style description to LLM: {style_for_llm[:100]}..." if len(style_for_llm) > 100 else f"Passing style description to LLM: {style_for_llm}")

        try:
            response: CoverLetterLatexDocumentResponse = self._content_chain.invoke(
                {
                    "existing_latex": template_content,
                    "cover_letter_text": cover_letter_text,
                    "style_description": style_for_llm,
                }
            )
        except Exception as e:
            logger.error(f"Error generating cover letter LaTeX content with LLM: {e}")
            raise

        latex_document = (response.latex_document or "").strip()
        logger.debug(f"LLM LaTeX document generated - length: {len(latex_document)} characters")

        return latex_document

    def _compile_latex(
        self, latex_content: str, output_name: str, max_fix_iterations: int = 1
    ) -> tuple[str, int, str | None]:
        """
        Compile LaTeX content to PDF with iterative error fixing.
        
        Args:
            latex_content: Complete LaTeX document
            output_name: Base name for output files
            max_fix_iterations: Maximum number of LaTeX fix attempts
            
        Returns:
            tuple[str, int, str | None]:
                - Path to generated PDF file
                - Number of LaTeX fix attempts performed
                - Last LaTeX error log (if compilation ultimately failed), otherwise None
        """
        # Create temporary LaTeX file
        tex_file = os.path.join(self.temp_folder, f"{output_name}.tex")
        tex_filename = f"{output_name}.tex"  # Relative filename for pdflatex
        pdf_file = os.path.join(self.temp_folder, f"{output_name}.pdf")
        final_pdf = os.path.join(self.output_folder, f"{output_name}.pdf")

        # Helper function to run pdflatex once
        def _run_pdflatex() -> subprocess.CompletedProcess:
            # Use relative filename and set cwd to temp_folder
            # This way pdflatex writes all output files (.log, .aux, .pdf) to temp_folder
            compile_cmd = [
                "pdflatex",
                "-interaction=nonstopmode",
                tex_filename,  # Use relative filename since cwd is set to temp_folder
            ]
            return subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                cwd=self.temp_folder,  # All output files will be written here
            )

        fix_attempts = 0
        last_error: str | None = None

        try:
            # Ensure temp_folder exists and is writable
            os.makedirs(self.temp_folder, exist_ok=True)
            if not os.access(self.temp_folder, os.W_OK):
                raise RuntimeError(f"Temp folder is not writable: {self.temp_folder}")
            
            # Write initial LaTeX file
            with open(tex_file, "w", encoding="utf-8") as f:
                f.write(latex_content)

            logger.info(f"Compiling LaTeX to PDF: {tex_file} (working directory: {self.temp_folder})")

            # First compilation attempt
            result = _run_pdflatex()

            # Iteratively try to fix LaTeX if compilation fails
            while result.returncode != 0 and fix_attempts < max_fix_iterations:
                fix_attempts += 1
                # LaTeX errors typically go to stdout, not stderr
                # Combine both for comprehensive error reporting
                error_output = result.stdout if result.stdout else result.stderr
                if result.stderr and result.stderr != result.stdout:
                    error_output = f"{result.stdout}\n\nSTDERR:\n{result.stderr}"
                last_error = error_output

                logger.warning(
                    f"pdflatex compilation failed (attempt {fix_attempts}) with code {result.returncode}"
                )
                logger.debug(f"pdflatex stdout (attempt {fix_attempts}): {result.stdout}")
                logger.debug(f"pdflatex stderr (attempt {fix_attempts}): {result.stderr}")

                fixed_latex = self._latex_fixer.fix(latex_content, error_output)
                if not fixed_latex or fixed_latex == latex_content:
                    logger.warning(
                        "LatexFixer did not produce a different LaTeX document. "
                        "Stopping further fix attempts."
                    )
                    break

                logger.info(
                    "LatexFixer produced an updated LaTeX document. Retrying compilation."
                )
                latex_content = fixed_latex
                
                # Ensure the fixed LaTeX is valid UTF-8 before writing
                # If it contains invalid bytes, encode/decode to clean it up
                try:
                    # Try to encode/decode to ensure valid UTF-8
                    latex_content = latex_content.encode('utf-8', errors='replace').decode('utf-8')
                except Exception as e:
                    logger.warning(f"Error cleaning UTF-8 in fixed LaTeX: {e}. Attempting to write anyway.")
                
                with open(tex_file, "w", encoding="utf-8", errors='replace') as f:
                    f.write(latex_content)

                result = _run_pdflatex()

            # Second compilation for references (only if first/repair attempt succeeded)
            if result.returncode == 0:
                # run a second pass to resolve references; ignore its warnings unless it fails
                second_result = _run_pdflatex()
                if second_result.returncode != 0:
                    logger.warning(
                        f"Second pdflatex compilation returned non-zero code: {second_result.returncode}"
                    )
                    logger.debug(f"pdflatex stdout (second attempt): {second_result.stdout}")
                    logger.debug(f"pdflatex stderr (second attempt): {second_result.stderr}")
            else:
                # Use combined error output for final error message
                final_error = result.stdout if result.stdout else result.stderr
                if result.stderr and result.stderr != result.stdout:
                    final_error = f"{result.stdout}\n\nSTDERR:\n{result.stderr}"
                last_error = final_error
                logger.error(
                    f"LaTeX compilation failed after {fix_attempts} fix attempts: {final_error}"
                )
                raise RuntimeError(f"Failed to compile LaTeX: {final_error}")

            # Move PDF to output folder
            if os.path.exists(pdf_file):
                import shutil

                shutil.move(pdf_file, final_pdf)
                logger.info(f"PDF generated successfully: {final_pdf}")
                return final_pdf, fix_attempts, None
            else:
                raise RuntimeError("PDF file was not created")

        except FileNotFoundError:
            logger.error(
                "pdflatex not found. Please install a LaTeX distribution (e.g., TeX Live, MiKTeX)"
            )
            raise RuntimeError(
                "LaTeX compiler (pdflatex) not found. Please install a LaTeX distribution."
            )
        except Exception as e:
            logger.error(f"Error compiling LaTeX: {e}")
            raise
    
    def generate_pdf(
        self,
        cover_letter_text: str,
        output_filename: str = "generated_cover_letter.pdf",
        style_description: Optional[str] = None,
        template_content: Optional[str] = None,
    ) -> str:
        """
        Generate a PDF from cover letter text.
        
        Args:
            cover_letter_text: The cover letter text content
            output_filename: Name for the output PDF file
            style_description: Optional appearance-only style instructions for the PDF
            template_content: Optional full LaTeX document to treat as a template
            
        Returns:
            str: Path to the generated PDF file
        """
        logger.info(
            f"Generating PDF from cover letter text (length: {len(cover_letter_text)} characters)"
        )
        
        # Decide which template to use:
        # - If template_content is provided, use it (e.g., previous generated .tex for style-only updates)
        # - Otherwise, use the base cover letter template from disk
        if template_content is None:
            template_content = self._load_template()
        
        # Build full LaTeX document from chosen template + LLM-generated content
        latex_document = self._build_latex_document(
            template_content,
            cover_letter_text,
            style_description,
        )
        
        # Generate output name without extension
        output_name = os.path.splitext(output_filename)[0]
        
        # Compile to PDF using max_fix_iterations from config
        pdf_path, fix_attempts, last_error = self._compile_latex(latex_document, output_name, max_fix_iterations=self.max_latex_fix_iterations)
        
        logger.info(f"PDF generation completed: {pdf_path}")
        return pdf_path
    
    def generate_pdf_from_file(
        self,
        txt_file_path: str,
        output_filename: Optional[str] = None,
        style_description: Optional[str] = None,
        template_content: Optional[str] = None,
    ) -> str:
        """
        Generate a PDF from a cover letter text file.
        
        Args:
            txt_file_path: Path to the cover letter text file
            output_filename: Optional name for the output PDF (defaults to same name as input with .pdf extension)
            style_description: Optional appearance-only style instructions for the PDF
            template_content: Optional full LaTeX document to treat as a template
            
        Returns:
            str: Path to the generated PDF file
        """
        logger.info(f"Generating PDF from file: {txt_file_path}")
        
        # Read cover letter text from file
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            cover_letter_text = f.read()
        
        # Determine output filename
        if output_filename is None:
            base_name = os.path.splitext(os.path.basename(txt_file_path))[0]
            output_filename = f"{base_name}.pdf"
        
        return self.generate_pdf(
            cover_letter_text,
            output_filename,
            style_description=style_description,
            template_content=template_content,
        )
    
    def run(self, state):
        """
        Main method to generate PDF from generated cover letter in state.
        
        Args:
            state: The state dictionary containing generated_cover_letter and optional template_customization_description
            
        Returns:
            dict: Updated state with PDF path information
        """
        logger.info("CoverLetterPDFGenerator.run() called")
        
        generated_cover_letter = state.get("generated_cover_letter")
        if not generated_cover_letter:
            logger.warning("No generated cover letter found in state")
            return {
                "cover_letter_pdf_path": None,
                "messages": state.get("messages", []) + [{
                    "role": "assistant",
                    "content": "No cover letter available to generate PDF."
                }]
            }
        
        try:
            # Always create draft versions with iteration number
            # Then copy to generated_cover_letter.pdf so it always has the latest version
            cover_letter_history = state.get("cover_letter_history", [])
            
            if cover_letter_history:
                # This is a draft version during refinement
                iteration = len(cover_letter_history)
            else:
                # No history, first generation (treat as draft_1)
                iteration = 1
            
            draft_filename = f"generated_cover_letter_draft_{iteration}.pdf"
            logger.info(f"Generating draft cover letter PDF (iteration {iteration})")
            
            # Use draft filename for the actual generation
            output_filename = draft_filename

            # Appearance-only PDF style preferences (from router/user)
            style_description = state.get("cover_letter_pdf_style")
            
            # Debug: Log style information
            logger.info(f"Cover letter PDF style from state: {style_description}")
            if style_description:
                logger.info(f"Style description length: {len(style_description)} characters")
                logger.debug(f"Style description content: {style_description[:200]}...")
            else:
                logger.info("No style description found in state - using default styling")

            # Choose template content:
            # - First iteration: use the base template file
            # - Subsequent iterations: use the previously generated .tex file as template
            template_content: Optional[str] = None
            if cover_letter_history:
                last_iteration = len(cover_letter_history)
                last_tex_path = os.path.join(
                    self.temp_folder,
                    f"generated_cover_letter_draft_{last_iteration}.tex",
                )
                if os.path.exists(last_tex_path):
                    logger.info(
                        f"Using previous generated cover letter .tex as template: {last_tex_path}"
                    )
                    with open(last_tex_path, "r", encoding="utf-8") as f:
                        template_content = f.read()
                else:
                    logger.warning(
                        f"Previous cover letter .tex not found at {last_tex_path}, "
                        "falling back to base template."
                    )

            # Build LaTeX document once so we can track fixes/attempts
            latex_document = self._build_latex_document(
                template_content or self._load_template(),
                generated_cover_letter,
                style_description,
            )

            # Generate output name without extension
            output_name = os.path.splitext(output_filename)[0]

            # Use max_latex_fix_iterations from state if provided, otherwise use instance default from config
            max_fix_iterations = state.get("max_latex_fix_iterations", self.max_latex_fix_iterations)

            pdf_path, fix_attempts, last_error = self._compile_latex(
                latex_document,
                output_name,
                max_fix_iterations,
            )

            # Always copy the latest draft to generated_cover_letter.pdf so it's always the latest version
            final_pdf_path = os.path.join(self.output_folder, "generated_cover_letter.pdf")
            import shutil
            shutil.copy2(pdf_path, final_pdf_path)
            logger.info(f"Copied latest draft to final PDF: {final_pdf_path}")

            message = f"Cover letter PDF generated successfully: {pdf_path} (also saved as generated_cover_letter.pdf)"
            logger.info(message)
            
            return {
                "cover_letter_pdf_path": final_pdf_path,  # Return the final path, not the draft path
                "cover_letter_latex_fix_attempts": fix_attempts,
                "cover_letter_latex_last_error": last_error,
                "messages": state.get("messages", []) + [{
                    "role": "assistant",
                    "content": message
                }]
            }
        except Exception as e:
            error_message = f"Error generating cover letter PDF: {str(e)}"
            logger.error(error_message)
            return {
                "cover_letter_pdf_path": None,
                "cover_letter_latex_fix_attempts": state.get("cover_letter_latex_fix_attempts", 0),
                "cover_letter_latex_last_error": str(e),
                "messages": state.get("messages", []) + [{
                    "role": "assistant",
                    "content": error_message
                }]
            }

