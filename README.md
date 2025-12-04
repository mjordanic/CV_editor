CV Editor Agentic System
========================

Create high‑quality, tailored CVs and cover letters with an AI-first, LangGraph‑based agentic workflow that reads your existing documents, analyzes a job description, researches the target company, and iteratively generates polished drafts with LLM-guided user feedback loops.

Table of Contents
-----------------

1. [Features](#features)  
2. [High-Level Architecture](#high-level-architecture)  
3. [Prerequisites](#prerequisites)  
4. [Project Setup](#project-setup)  
5. [Environment Variables](#environment-variables)  
6. [Running the Agent](#running-the-agent)  
7. [Workflow Walkthrough](#workflow-walkthrough)  
8. [Logging and Debugging](#logging-and-debugging)  
9. [Configuration](#configuration)  
10. [Customization Tips](#customization-tips)  
11. [Troubleshooting](#troubleshooting)

Features
--------

![LangGraph workflow](images/graph_visualization.png)

- **Smart Experience Retrieval (RAG)**: Automatically retrieves relevant projects and experiences from your portfolio (`data/portfolio/`) based on the job description.
- **Relevance Verification**: Uses a two-stage filtering process (Vector Similarity + LLM Verification) to ensure only truly relevant experience is included.
- **Automatic Refinement**: Documents are automatically refined based on critique feedback until quality thresholds are met.
- **Version Control**: All document iterations are saved as draft versions with quality scores.
- **Structured Data**: Centralized data management in `data/` folder (`CV`, `portfolio`, `generated_CVs`).
- **Flexible Input**: Supports reading job descriptions from text files (`job_description/`).
- **Human-in-the-loop**: You steer the AI at every decision point.
- **Intelligent PDF Generation**: 
  - **LLM-Powered LaTeX Generation**: The system uses an LLM to generate complete LaTeX documents (including preamble and content), not just filling template placeholders. This allows for full customization of document structure, styling, and formatting.
  - **Natural Language Style Customization**: Specify PDF appearance preferences in plain English (e.g., "make titles blue and bolded", "use minimal spacing", "make it more compact"). The LLM modifies the LaTeX code accordingly.
  - **Post-Hoc Style Modifications**: Change PDF appearance after generation without regenerating content. Simply describe the desired changes and the system updates the PDF styling.
  - **Compiler-in-the-Loop Error Fixing**: Automatic LaTeX compilation error detection and fixing using an LLM. The system attempts to fix compilation errors iteratively (configurable max iterations).
  - **Automatic Versioning**: Each PDF generation creates a draft version (`generated_CV_draft_N.pdf`). The latest version is always copied to `generated_CV.pdf` and `generated_cover_letter.pdf` for easy access.
- **Advanced Feedback System**:
  - **Separate Feedback Fields**: Provide feedback separately for CV and cover letter, each with distinct content and style feedback fields.
  - **Combined Feedback in Single Messages**: Include both content modifications and style preferences in a single message. For example: "Add my research stay at Berkeley. Make titles blue and bolded."
  - **Sequential Execution**: Request both CV and cover letter in one message, and the system will generate them sequentially without asking for additional input between documents.
  - **Intelligent Routing**: The router uses conversation history to understand user intent, automatically routing to generate missing documents when both were requested.

High-Level Architecture
-----------------------

- **LangGraph Orchestrator (`MasterAgent`)**: Builds the workflow graph, handles streaming execution, and coordinates interrupts for user input.
- **DocumentReaderAgent**: Loads user-provided documents from disk.
- **JobDescriptionAgent**: Pauses execution to request a job description, then extracts structured data.
- **SearchAgent**: Queries Tavily, summarizes results, and determines remote-work stance.
- **ExperienceRetrievalAgent**: Retrieves relevant experience from your portfolio using RAG (Retrieval-Augmented Generation) with relevance verification.
- **RouterAgent**: Uses conversation context and LLM reasoning to decide the next action (generate CV, cover letter, prompt user, exit). Intelligently handles sequential document generation when both are requested.
- **CVWriterAgent / CoverLetterWriterAgent**: Generate tailored drafts using LangChain chat models with iterative refinement based on critique feedback.
- **CritiqueAgent**: Evaluates generated documents for quality, ATS compatibility, and job alignment. Provides actionable improvement instructions and quality scores.
- **CVPDFGenerator / CoverLetterPDFGenerator**: Convert generated text documents to professional PDFs using LLM-generated LaTeX. Supports style customization, error fixing, and automatic versioning.
- **LatexFixer**: LLM-based agent that automatically fixes LaTeX compilation errors by analyzing error logs and generating corrected LaTeX code.
- **UserInputAgent**: Collects free-form responses when more clarity or feedback is required.

All agents share a common `State` TypedDict managed by LangGraph, which keeps track of messages, extracted info, generated artifacts, critique feedback, and version history. The result is a cohesive AI agentic system where multiple specialized models collaborate.

Prerequisites
-------------

- Python 3.11+ (project tested with 3.12).
- Access to OpenAI API with GPT-5 series models (`gpt-5-mini`, `gpt-5-nano`) or configure alternative models in `config/agent_models.yaml`.
- Tavily API key for company research.
- LaTeX distribution (for PDF generation): TeX Live, MiKTeX, or MacTeX with `pdflatex` command available.
- macOS/Linux (Windows works but paths/logging instructions assume POSIX).

Project Setup
-------------

```bash
# clone the repo (example path)
git clone <repo-url> CV_editor
cd CV_editor

# create and activate a virtual environment (uv or venv)
uv venv               # or: python3 -m venv .venv
source .venv/bin/activate

# install dependencies
uv pip install -r requirements.txt   # or `uv pip sync pyproject.toml` if using uv/lockfile
```

Environment Variables
---------------------

Create a `.env` file in the project root (or set variables in your shell):

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

`dotenv` loads these before any agent imports, so keep `.env` up to date.

Running the Agent
-----------------

```bash
source .venv/bin/activate
python langgraph_agent.py
```

The program will:

1. Load user CV/cover-letter data from `data/CV/`.
2. Load portfolio data from `data/portfolio/`.
3. Ask you to paste the target job description (or provide a file path).
4. Fetch company insights via Tavily.
5. Retrieve relevant experience from your portfolio.
6. Chat with you about what to generate next.
7. Produce tailored documents and save them to `data/generated_CVs/`.

While streaming, the CLI may pause with prompts such as “Please paste the job description…” or “Please provide your input:”. Paste multiline text and press `Ctrl+D` (or Enter twice) to continue.

**Tip:** For long job descriptions that might exceed terminal buffer limits, you can save the text to a file (e.g., `job_description/my_job.txt`) and simply paste the file path when prompted.

**Example Workflow:**
```
User: "Please create CV and cover letter. Add my research stay at Berkeley in 2021 to the CV. 
       Also add MATLAB to my skills. I want all titles in dark blue color and bolded. 
       For the cover letter, emphasize my research experience and make it more compact."

System: [Generates CV with content and style changes]
        [Generates cover letter with content and style changes]
        [No additional prompts needed - sequential execution]
```

Workflow Walkthrough
--------------------

1. **Document ingestion**: `DocumentReaderAgent` loads `data/CV/CV.(txt|pdf)` and `data/CV/cover_letter.(txt|pdf)`.
2. **Job description capture**: `JobDescriptionAgent` interrupts for pasted text or file path (e.g., `job_description/job.txt`), extracts company name, work type, requirements, etc.
3. **Company research**: `SearchAgent` builds a Tavily query, summarizes results with an LLM, and determines remote-work support.
4. **Experience Retrieval**: `ExperienceRetrievalAgent` searches your portfolio (`data/portfolio/`) for projects relevant to the job description. It uses vector similarity (threshold defined in `config/agent_models.yaml`) and an LLM to verify relevance.
5. **Routing decision**: `RouterAgent` considers chat history, existing outputs, and user feedback to pick the next node.
5. **Generation**:
   - `CVWriterAgent` creates or updates a CV using the selected LLM.
   - `CoverLetterWriterAgent` creates or updates a cover letter (often leveraging the freshly generated CV).
6. **Quality assessment and refinement**:
   - `CritiqueAgent` evaluates generated documents on three dimensions: content quality, ATS compatibility, and job alignment.
   - Provides a quality score (0-100) and actionable improvement instructions.
   - All document iterations are saved to version history with their quality scores as draft files.
   - If quality score is below threshold (default: 85) or critical issues are identified, the document is automatically refined based on the improvement instructions.
   - Refinement loop continues until quality threshold is met or no further improvements are suggested.
7. **Final selection**: The highest-scoring version from the iteration history is selected as the final output.
8. **PDF Generation**: 
   - `CVPDFGenerator` and `CoverLetterPDFGenerator` convert the final text documents to professional PDFs.
   - The LLM generates complete LaTeX documents based on the text content and any style preferences.
   - LaTeX compilation errors are automatically fixed using the `LatexFixer` agent (up to a configurable number of iterations).
   - Each generation creates a draft version (`generated_CV_draft_N.pdf`), with the latest always available as `generated_CV.pdf`.
9. **User feedback loop**: 
   - The router intelligently extracts content and style feedback from user messages.
   - You can provide feedback for both CV and cover letter in a single message.
   - Content feedback modifies the text, while style feedback modifies PDF appearance.
   - If both documents are requested, the system generates them sequentially without additional prompts.
   - `UserInputAgent` interrupts the graph when clarification is needed, allowing you to provide instructions that reset version history and initiate new refinement cycles.

Logging and Debugging
---------------------

- **Structured logs**: Stored in `logs/cv_editor_<timestamp>.log`.
- **Debug transcripts**: `debug/cv_editor_<timestamp>.log` captures prompts, state snapshots, and generated text segments for auditing.
- **Graph visualization**: On startup the orchestrator attempts to render `images/graph_visualization.png`. Install Graphviz + `pygraphviz` if you want this artifact.

Configuration
-------------

The system uses a centralized YAML configuration file (`config/agent_models.yaml`) to manage AI models and workflow settings. This approach provides a single source of truth for all configuration, making it easy to adjust behavior without code changes.

**Key Configuration Options:**
- **Agent Models**: Each agent (CV writer, cover letter writer, router, critique, etc.) can be configured with specific models and temperatures.
- **LaTeX Fixer**: Configure `max_latex_fix_iterations` in the `workflow` section to control how many times the system attempts to fix LaTeX compilation errors automatically (default: 1).
- **Quality Thresholds**: Set quality score thresholds for automatic refinement in the critique agent configuration.
- **RAG Settings**: Configure vector similarity thresholds and relevance verification settings for experience retrieval.

All agents load configuration at startup from the YAML file. No code changes needed to adjust models or workflow parameters.

Customization Tips
------------------

- **Models**: Change model IDs or temperatures in each agent (e.g., `CVWriterAgent` uses `temperature=0.7`). Ensure the chosen model is available in your OpenAI organization.
- **Prompt tone**: Update the system/human prompt strings in each agent to reflect a different writing style or formatting rules.
- **PDF Templates**: Base LaTeX templates are in `templates/` directory. The LLM uses these as starting points but can modify them completely based on your style preferences. See `templates/README.md` for details.
- **PDF Style Customization**: Provide style preferences in natural language when generating PDFs. Examples:
  - "Make all titles dark blue and bolded"
  - "Use minimal spacing and make it more compact"
  - "Change the font to Arial and increase margins"
  - The LLM will modify the LaTeX code (including preamble) to implement these changes.
- **LaTeX Error Fixing**: Configure `max_latex_fix_iterations` in `config/agent_models.yaml` to control how many times the system attempts to fix LaTeX compilation errors automatically.
- **State fields**: Extend `State` in `langgraph_agent.py` if you want to track additional data between nodes.
- **Alternate search providers**: Replace `SearchAgent.search_tavily` with another API if Tavily isn't desired. Just keep `company_info` schema consistent.

Troubleshooting
---------------

- **Missing docstring lint**: Already resolved; if you add new modules, remember to include docstrings for all public classes/functions.
- **Graphviz errors**: Install Graphviz (`brew install graphviz`) and `pygraphviz` per the instructions in `langgraph_agent.py`.
- **Tracing interrupts**: If the CLI seems stuck, make sure you finished input with `Ctrl+D` or typed `END` on its own line.
- **LaTeX compilation errors**: If PDF generation fails repeatedly, check the logs for LaTeX error messages. The system will attempt to fix errors automatically, but some issues may require manual template adjustments.
- **Missing pdflatex**: Ensure `pdflatex` is in your PATH. On macOS, install with `brew install texlive`. Verify with `which pdflatex`.
- **Imports failing in `langgraph_agent_example.py`**: This legacy sample references optional packages. Either install the missing dependencies or ignore the warnings; it is not used by the main workflow.

Questions or ideas? Open an issue or drop a note in the repo discussions. Happy document crafting!


TODO list
---------
- ~~**PDF generation agent**: Add an agent that uses LaTeX to generate professional PDF versions of CVs and cover letters.~~ ✅ **Completed**: CVPDFGenerator and CoverLetterPDFGenerator implemented with customizable LaTeX templates.
- **Enhanced candidate information system**: Currently, only the candidate's CV and cover letter can be uploaded. This should be improved to:
  - Accept any document type (portfolio, certificates, transcripts, etc.) and process them automatically
  - Create a more complete representation of the candidate's skills, aspirations, and experience
  - Allow candidates to input additional information via prompts
  - Enable the agent to proactively ask for missing information to build a comprehensive candidate profile
- ~~**Quality assessment**: Add an agent that evaluates generated documents and provides feedback on content quality, ATS compatibility, and alignment with job requirements.~~ ✅ **Completed**: CritiqueAgent implemented with quality scoring and automatic refinement.
- **ATS optimization**: Add features to optimize documents for Applicant Tracking Systems (ATS), including keyword optimization, format validation, and ATS compatibility scoring.
- **Multi-format export**: Support exporting generated documents to multiple formats (Word, HTML, Markdown) in addition to PDF and plain text.
- **Template system**: Implement a template selection and customization system, allowing users to choose from different CV/cover letter styles and formats.
- **Batch processing**: Enable processing multiple job applications simultaneously, generating tailored documents for multiple positions in a single run.
- **Web interface**: Develop a web-based UI to replace or complement the CLI, making the system more accessible and user-friendly.
- **Integration capabilities**: Add integrations with professional networks (LinkedIn), job boards, and document storage services (Google Drive, Dropbox).
- ~~**Configuration management**: Create a configuration file system for default settings, model preferences, and workflow customization without code changes.~~ ✅ **Completed**: YAML-based configuration system implemented for agent models and workflow settings.
- **Testing and CI/CD**: Add comprehensive unit tests, integration tests, and set up a CI/CD pipeline for automated testing and deployment.
- **Multi-language support**: Extend the system to support CV and cover letter generation in multiple languages.

