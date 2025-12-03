from typing import Optional
import logging
import os
import subprocess
import sys
from pathlib import Path

# Add project root to path when running as standalone script
# This allows imports like 'from agents.latex_fixer import LatexFixer' to work
if __file__ and Path(__file__).parent.name == "agents":
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.latex_fixer import LatexFixer
from config.config_loader import load_config

logger = logging.getLogger(__name__)


CV_LATEX_SYSTEM_PROMPT = (
    "You are an expert LaTeX CV formatter. "
    "Your task is to take an existing LaTeX CV document, a plain-text CV, and style preferences, "
    "and produce a COMPLETE LaTeX CV document that is ready to compile with pdflatex.\n\n"
    "CRITICAL RULES:\n"
    "- You MUST output a full LaTeX document, including the preamble (\\documentclass, \\usepackage, "
    "  color definitions, custom commands, and \\begin{{document}} ... \\end{{document}}).\n"
    "- Start from the existing LaTeX document provided to you. You may rewrite or simplify it as needed, "
    "  but keep any working structure that is still appropriate.\n"
    "- Apply the style_preferences to BOTH the preamble (colors, spacing, fonts, layout) and the content "
    "  (section structure, emphasis) where appropriate.\n"
    "- Keep the factual content faithful to the plain-text CV, but you may improve formatting, ordering, "
    "  and wording for clarity and professionalism.\n"
    "- Escape LaTeX special characters where needed (%, _, $, &, #, {{, }}, ~, ^, \\).\n"
    "- Keep the CV professional, readable, and well-structured.\n"
)

CV_LATEX_HUMAN_PROMPT = (
    "You are given:\n\n"
    "1) The CURRENT LaTeX CV DOCUMENT (starting point):\n"
    "-------------------------------------\n"
    "{existing_latex}\n"
    "-------------------------------------\n\n"
    "2) The plain-text CV content (ground truth for the resume data):\n"
    "-----------------------------\n"
    "{cv_text}\n"
    "-----------------------------\n\n"
    "3) PDF style preferences (APPEARANCE ONLY, no factual content changes):\n"
    "--------------------------------------------------------------------------\n"
    "{style_description}\n"
    "--------------------------------------------------------------------------\n\n"
    "Your job is to produce a COMPLETE LaTeX CV document that:\n"
    "- Keeps or improves the existing layout and structure when it makes sense.\n"
    "- Applies the style preferences (e.g., colors, spacing, typography, section layout).\n"
    "- Preserves the factual content from the plain-text CV, but may reorganize/rephrase for clarity.\n\n"
    "IMPORTANT:\n"
    "- You MUST output a single, self-contained LaTeX document that can be compiled with pdflatex.\n"
    "- You may modify or remove existing commands and add new ones as needed, as long as the document compiles.\n"
    "- Do NOT include any explanatory comments or prose—ONLY LaTeX code.\n"
    "- Assume the color and package setup from the existing document can be adjusted as needed.\n\n"
    "Return exactly one field:\n"
    "- latex_document: the full LaTeX source for the updated CV.\n"
)


class CVLatexDocumentResponse(BaseModel):
    """Structured response for CV LaTeX full document generation."""

    latex_document: str = Field(
        ...,
        description="Complete LaTeX document for the CV, including preamble and document environment.",
    )


class CVPDFGenerator:
    """Generate CV PDFs from plain-text CVs using an LLM + LaTeX template."""

    def __init__(
        self,
        template_path: str = "templates/cv_template.tex",
        output_folder: str = "data/generated_CVs",
        temp_folder: str = "data/generated_CVs/temp_latex",
        model: str = "openai:gpt-5-nano",
        temperature: float = 0.1,
    ):
        logger.info(f"Initializing CVPDFGenerator with template: {template_path}")

        self.template_path = template_path
        self.output_folder = output_folder
        self.temp_folder = temp_folder

        # Ensure directories exist
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.temp_folder, exist_ok=True)

        # Verify template exists
        if not os.path.exists(self.template_path):
            logger.error(f"CV template not found at {self.template_path}")
            raise FileNotFoundError(f"CV template not found at {self.template_path}")

        # Initialize LLM for LaTeX content
        logger.info(f"Initializing CV content LLM - model: {model}")
        self._content_llm = init_chat_model(model, temperature=temperature)
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CV_LATEX_SYSTEM_PROMPT),
                ("human", CV_LATEX_HUMAN_PROMPT),
            ]
        )
        structured_llm = self._content_llm.with_structured_output(CVLatexDocumentResponse)
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
            f"CVPDFGenerator initialized - template: {self.template_path}, "
            f"output_folder: {self.output_folder}, temp_folder: {self.temp_folder}, "
            f"max_latex_fix_iterations: {self.max_latex_fix_iterations}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_template(self) -> str:
        """Load the LaTeX template from disk."""
        try:
            with open(self.template_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading CV template: {e}")
            raise

    def _build_latex_document(
        self,
        template_content: str,
        cv_text: str,
        style_description: Optional[str],
    ) -> str:
        """Use the LLM to produce a full LaTeX document, starting from the given template."""
        style_description = (style_description or "").strip()

        logger.info("Generating LaTeX CV content using LLM...")
        logger.debug(
            f"Plain-text CV length: {len(cv_text)} characters, "
            f"style description length: {len(style_description)}"
        )
        
        # Prepare style description for LLM
        style_for_llm = style_description or "No special style preferences."
        logger.info(f"Passing style description to LLM: {style_for_llm[:100]}..." if len(style_for_llm) > 100 else f"Passing style description to LLM: {style_for_llm}")

        try:
            response: CVLatexDocumentResponse = self._content_chain.invoke(
                {
                    "existing_latex": template_content,
                    "cv_text": cv_text,
                    "style_description": style_for_llm,
                }
            )
        except Exception as e:
            logger.error(f"Error generating LaTeX content with LLM: {e}")
            raise

        latex_document = (response.latex_document or "").strip()
        logger.debug(f"LLM LaTeX document generated - length: {len(latex_document)} characters")

        return latex_document
    
    def _compile_latex(
        self,
        latex_content: str,
        output_name: str,
        max_fix_iterations: int = 1,
    ) -> tuple[str, int, str | None]:
        """
        Compile LaTeX content to PDF.
        
        Args:
            latex_content: Complete LaTeX document
            output_name: Base name for output files
            
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
            # Propagate last_error alongside the exception
            if last_error is None and isinstance(e, RuntimeError):
                last_error = str(e)
            raise
    
    def generate_pdf(
        self,
        cv_text: str,
        output_filename: str = "generated_CV.pdf",
        style_description: Optional[str] = None,
        template_content: Optional[str] = None,
    ) -> str:
        """
        Generate a PDF from CV text.
        
        Args:
            cv_text: The CV text content
            output_filename: Name for the output PDF file
            style_description: Optional appearance-only style instructions for the PDF
            template_content: Optional LaTeX template content (uses base template if None)
            
        Returns:
            str: Path to the generated PDF file
        """
        logger.info(f"Generating PDF from CV text (length: {len(cv_text)} characters)")
        
        # Decide which template to use:
        # - If template_content is provided, use it (e.g., previous generated .tex for style-only updates)
        # - Otherwise, use the base CV template from disk
        if template_content is None:
            template_content = self._load_template()
        
        # Build full LaTeX document from chosen template + LLM-generated content
        latex_document = self._build_latex_document(template_content, cv_text, style_description)
        
        # Generate output name without extension
        output_name = os.path.splitext(output_filename)[0]

        # Compile to PDF using max_fix_iterations from config
        pdf_path, _, _ = self._compile_latex(latex_document, output_name, max_fix_iterations=self.max_latex_fix_iterations)
        
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
        Generate a PDF from a CV text file.
        
        Args:
            txt_file_path: Path to the CV text file
            output_filename: Optional name for the output PDF (defaults to same name as input with .pdf extension)
            style_description: Optional appearance-only style instructions for the PDF
            
        Returns:
            str: Path to the generated PDF file
        """
        logger.info(f"Generating PDF from file: {txt_file_path}")
        
        # Read CV text from file
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            cv_text = f.read()
        
        # Determine output filename
        if output_filename is None:
            base_name = os.path.splitext(os.path.basename(txt_file_path))[0]
            output_filename = f"{base_name}.pdf"
        
        return self.generate_pdf(
            cv_text,
            output_filename,
            style_description=style_description,
            template_content=template_content,
        )
    
    def run(self, state):
        """
        Main method to generate PDF from generated CV in state.
        
        Args:
            state: The state dictionary containing generated_cv and optional template_customization_description
            
        Returns:
            dict: Updated state with PDF path information
        """
        logger.info("CVPDFGenerator.run() called")
        
        generated_cv = state.get("generated_cv")
        if not generated_cv:
            logger.warning("No generated CV found in state")
            return {
                "cv_pdf_path": None,
                "messages": state.get("messages", []) + [{
                    "role": "assistant",
                    "content": "No CV available to generate PDF."
                }]
            }
        
        try:
            # Always create draft versions with iteration number
            # Then copy to generated_CV.pdf so it always has the latest version
            cv_history = state.get("cv_history", [])
            
            if cv_history:
                # This is a draft version during refinement
                iteration = len(cv_history)
            else:
                # No history, first generation (treat as draft_1)
                iteration = 1
            
            draft_filename = f"generated_CV_draft_{iteration}.pdf"
            logger.info(f"Generating draft CV PDF (iteration {iteration})")
            
            # Use draft filename for the actual generation
            output_filename = draft_filename

            # Appearance-only PDF style preferences (from router/user)
            style_description = state.get("cv_pdf_style")
            
            # Debug: Log style information
            logger.info(f"CV PDF style from state: {style_description}")
            if style_description:
                logger.info(f"Style description length: {len(style_description)} characters")
                logger.debug(f"Style description content: {style_description[:200]}...")
            else:
                logger.info("No style description found in state - using default styling")

            # Choose template content:
            # - First iteration: use the base template file
            # - Subsequent iterations (drafts or final): use the previously generated .tex file as template
            template_content: Optional[str] = None
            if cv_history:
                # Last iteration index
                last_iteration = len(cv_history)
                # Try to find the last draft's .tex file
                last_tex_path = os.path.join(
                    self.temp_folder,
                    f"generated_CV_draft_{last_iteration}.tex",
                )
                if os.path.exists(last_tex_path):
                    logger.info(
                        f"Using previous generated CV .tex as template: {last_tex_path}"
                    )
                    with open(last_tex_path, "r", encoding="utf-8") as f:
                        template_content = f.read()
                else:
                    logger.warning(
                        f"Previous CV .tex not found at {last_tex_path}, "
                        "falling back to base template."
                    )

            # Build LaTeX document once so we can track fixes/attempts
            latex_document = self._build_latex_document(
                template_content or self._load_template(),
                generated_cv,
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

            # Always copy the latest draft to generated_CV.pdf so it's always the latest version
            final_pdf_path = os.path.join(self.output_folder, "generated_CV.pdf")
            import shutil
            shutil.copy2(pdf_path, final_pdf_path)
            logger.info(f"Copied latest draft to final PDF: {final_pdf_path}")

            message = f"CV PDF generated successfully: {pdf_path} (also saved as generated_CV.pdf)"
            logger.info(message)

            return {
                "cv_pdf_path": final_pdf_path,  # Return the final path, not the draft path
                "cv_latex_fix_attempts": fix_attempts,
                "cv_latex_last_error": last_error,
                "messages": state.get("messages", []) + [{
                    "role": "assistant",
                    "content": message
                }]
            }
        except Exception as e:
            error_message = f"Error generating CV PDF: {str(e)}"
            logger.error(error_message)
            return {
                "cv_pdf_path": None,
                "cv_latex_fix_attempts": state.get("cv_latex_fix_attempts", 0),
                "cv_latex_last_error": str(e),
                "messages": state.get("messages", []) + [{
                    "role": "assistant",
                    "content": error_message
                }]
            }


if __name__ == "__main__":
    """
    Standalone test script for CV PDF generation.
    Creates dummy CV data and generates a PDF for testing.
    """
    # Load environment variables (for API keys)
    from dotenv import load_dotenv
    load_dotenv()
    
    # Setup basic logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Dummy CV text for testing
    dummy_cv_text = """John Doe
New York, NY, USA
Phone: +1 (555) 123-4567
Email: john.doe@example.com

Professional Summary
Experienced Software Engineer with 5+ years developing scalable web applications. 
Expert in Python, JavaScript, and cloud technologies. Passionate about building 
innovative solutions that drive business value.

Professional Experience
Senior Software Engineer
Tech Company Inc.
San Francisco, CA (2020 - Present)
- Led development of microservices architecture serving 1M+ users
- Implemented CI/CD pipelines reducing deployment time by 60%
- Mentored junior developers and established coding best practices

Software Engineer
Startup Corp
New York, NY (2018 - 2020)
- Built RESTful APIs using Python and Flask
- Developed frontend components with React and TypeScript
- Collaborated with cross-functional teams to deliver features on time

Education
Bachelor of Science in Computer Science
University of Technology
New York, NY (2014 - 2018)

Technical Skills
- Programming Languages: Python, JavaScript, TypeScript, Java
- Frameworks: React, Flask, Django, Node.js
- Cloud: AWS, Docker, Kubernetes
- Databases: PostgreSQL, MongoDB, Redis
"""
    
    print("=" * 60)
    print("CV PDF Generator - Standalone Test")
    print("=" * 60)
    print(f"\nDummy CV text length: {len(dummy_cv_text)} characters")
    print("\nInitializing CVPDFGenerator...")
    
    try:
        # Initialize the generator
        generator = CVPDFGenerator()

        # Test 1: Generate PDF with default settings
        print("\n" + "-" * 60)
        print("Test 1: Generating PDF with default template")
        print("-" * 60)
        output_filename = "test_cv_standalone.pdf"
        pdf_path = generator.generate_pdf(
            cv_text=dummy_cv_text,
            output_filename=output_filename,
            style_description=None
        )
        print(f"✓ PDF generated successfully: {pdf_path}")
        
        # Test 2: Generate PDF with style description
        print("\n" + "-" * 60)
        print("Test 2: Generating PDF with style preferences")
        print("-" * 60)
        output_filename_2 = "test_cv_standalone_styled.pdf"
        pdf_path_2 = generator.generate_pdf(
            cv_text=dummy_cv_text,
            output_filename=output_filename_2,
            style_description="Make the CV more compact with less whitespace, use bold headings"
        )
        print(f"✓ Styled PDF generated successfully: {pdf_path_2}")
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

