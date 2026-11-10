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
    "- Copy the text of the plain-text CV word for word, in the same order and language. Change only the "
    "  LaTeX markup and layout: do NOT reword, add, drop or reorder sentences, and do NOT change names or "
    "  accented letters (that text was already checked against the candidate's material). Escaping a special "
    "  character is not a change.\n"
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
    "- Contains the plain-text CV word for word (only LaTeX markup and layout change).\n\n"
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


class CVPDFGenerator(LatexPDFGenerator):
    """Generate CV PDFs from plain-text CVs using an LLM + LaTeX template."""

    kind = "cv"
    label = "CV"
    file_stem = "generated_CV"
    text_variable = "cv_text"
    system_prompt = CV_LATEX_SYSTEM_PROMPT
    human_prompt = CV_LATEX_HUMAN_PROMPT
    response_model = CVLatexDocumentResponse

    def __init__(
        self,
        template_path: str = "templates/cv_template.tex",
        output_folder: str = "data/generated_CVs",
        temp_folder: str = "data/generated_CVs/temp_latex",
        model: str = "openai:gpt-5-nano",
        temperature: float = 0.1,
    ):
        super().__init__(template_path, output_folder, temp_folder, model, temperature)

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
            dummy_cv_text,
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
            dummy_cv_text,
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

