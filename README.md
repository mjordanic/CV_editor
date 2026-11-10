CV Editor
=========

CV Editor helps job seekers tailor a CV and cover letter to one specific job ad. You give it your current CV and the text of the ad. It reads both, looks up the company, writes a matching CV and cover letter, and turns them into PDFs. You can then ask for changes in plain English ("add my research stay at Berkeley", "make the titles blue") until you are happy. It runs on your own computer from a terminal, so it is aimed at people who are comfortable installing a Python program. It uses paid OpenAI and Tavily accounts: your CV and the job ad are sent to those services, and each run costs a small amount in API fees.

Table of Contents
-----------------

1. [Quick Start](#quick-start)
2. [Features](#features)
3. [High-Level Architecture](#high-level-architecture)
4. [Prerequisites](#prerequisites)
5. [Project Setup](#project-setup)
6. [Environment Variables](#environment-variables)
7. [Running the Agent](#running-the-agent)
8. [Workflow Walkthrough](#workflow-walkthrough)
9. [Logging and Debugging](#logging-and-debugging)
10. [Configuration](#configuration)
11. [Customization Tips](#customization-tips)
12. [Troubleshooting](#troubleshooting)
13. [Drawing the workflow graph](#drawing-the-workflow-graph)
14. [Licence](#licence)

Quick Start
-----------

You need Python 3.12, [uv](https://docs.astral.sh/uv/), a LaTeX install with `pdflatex` (see [Prerequisites](#prerequisites)), an OpenAI API key and a Tavily API key.

```bash
git clone https://github.com/mjordanic/CV_editor.git
cd CV_editor
uv sync                                    # creates .venv and installs the locked dependencies

cp .env_sample .env                        # then put your real keys in .env

mkdir -p data/CV
cp examples/sample_CV.txt data/CV/CV.txt   # or put your own CV here as data/CV/CV.txt or data/CV/CV.pdf

uv run python langgraph_agent.py
```

- **Your CV** goes in `data/CV/` as `CV.txt`, `CV.md` or `CV.pdf`. An existing cover letter is optional (`data/CV/cover_letter.txt`, `.md` or `.pdf`). `examples/sample_CV.txt` is a made-up CV you can use to try the program (`examples/sample_CV_butcher.txt` is one for a job outside IT).
- **Free-form notes** are optional: `data/CV/notes.txt` (or `.md`, `.pdf`) can hold anything about your career in any shape, such as a rough CV draft without structure, bullet notes, or caveats ("exam booked, not passed yet"). The writers use them as your own facts, like the CV, and the check for invented claims counts them as evidence. Notes alone are enough to start: without a CV the program builds one from your notes and portfolio.
- **A portfolio** is optional: if you have one, put your own projects and past jobs in `data/portfolio/portfolio.txt`, one item per paragraph (separated by a blank line). Only that file is read, and only if you created it; without it the program works from your CV and notes alone (it logs "No portfolio"). The repository ships no portfolio: `examples/sample_portfolio.txt` only shows the format and is never read.
- **The job ad**: when asked, paste the ad text or a path to a file. `job_description/sample_job_ad.txt` is a made-up ad for trying it out (`job_description/sample_job_ad_butcher.txt` goes with the butcher CV).
- **Costs**: every step after you paste the job ad makes paid OpenAI and Tavily calls. A portfolio up to 12,000 characters is given to the writers whole; only a larger one is indexed (one paid embedding call) so its most relevant items can be retrieved.
- The `data/` and `job_description/` folders are in `.gitignore`, so your own CV and ads are not committed by accident.
- **Recording a session**: `uv run python langgraph_agent.py --trace-dir DIR` writes every step, every model call (prompt, response, tokens, cost) and every question and reply to `DIR` (see [Testing mode and evaluation](#testing-mode-and-evaluation)). API keys are never written.

### What candidate information the program can take

| Input | Where | Formats | Used by |
|---|---|---|---|
| CV | `data/CV/CV.*`, or pasted / a file path when asked (only if there is no CV and no notes) | `.txt`, `.md`, `.pdf` (text PDFs; scanned PDFs have no text) | CV and cover letter writers, critique, invented-claims check |
| Earlier cover letter | `data/CV/cover_letter.*` | `.txt`, `.md`, `.pdf` | both writers as source facts (the first letter for a new job is written fresh for that job, using the old one only for facts and voice; a request you make about it on the first letter is applied as you ask, and the letter is also readdressed to this employer, role and addressee), critique, invented-claims check |
| Free-form notes | `data/CV/notes.*` | `.txt`, `.md`, `.pdf` | CV and cover letter writers, critique, invented-claims check |
| Portfolio (optional, your own) | `data/portfolio/portfolio.txt` | plain text, one item per paragraph | the writers (the whole file up to 12,000 characters, otherwise the items retrieved for the ad), the critique and the invented-claims check |
| Work-arrangement preference (remote, hybrid, on-site, no preference) | say it in the chat ("I only want remote jobs"); `--work-preference` in testing mode | writers state it only if you gave it; the program tells you when the job's arrangement conflicts with it. Without one you count as open to any arrangement: remote, hybrid, on-site or relocation is never presented as a conflict or weakness, and no specific commitment (e.g. relocating to a named city) is invented |
| Anything you say in the chat | typed or pasted during the session | text | your requests steer the CV and letter (they are applied as you ask, also in your first message); the writers, the critique and the invented-claims check treat what you say about yourself as your own facts |

Features
--------

- **Portfolio**: Your portfolio (`data/portfolio/portfolio.txt`, optional) goes to the writers whole; a large one (over 12,000 characters) is searched for the items relevant to the job description (RAG).
- **Output Language**: The CV and cover letter are written in the language of the job ad, even when the job is in another country; ask for another language ("write the cover letter in English") to change it.
- **Relevance Verification** (large portfolios only): a two-stage filter (vector similarity + LLM verification) keeps only relevant items.
- **Automatic Refinement**: A draft is redrafted automatically while it has a real defect: a claim your material does not support, a critical issue, or a requirement of the ad you meet but the draft does not show. The budget is small: one redraft by default, and at most one more if the draft still makes unsupported claims or the last redraft showed more of the ad's requirements (see [Configuration](#configuration)).
- **Honesty checks at the end**: if the chosen draft still has unsupported claims, one removal-only pass deletes them and the check runs again; any claim that remains is listed to you word for word. After the PDF is made, its text is compared with the checked text, and you are told if they differ. A document that still has placeholders such as `[EMAIL]` makes the program ask you once for the missing details before the PDF.
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
- **SearchAgent**: Queries Tavily (only when the ad names the company), summarizes the results, and extracts the work arrangement, location limits and culture signals from the ad and the results.
- **ExperienceRetrievalAgent**: Reads your portfolio if you supplied one; gives it to the writers whole, or retrieves the relevant items of a large one (RAG with relevance verification).
- **RouterAgent**: Uses conversation context and LLM reasoning to decide the next action (generate CV, cover letter, prompt user, exit). Intelligently handles sequential document generation when both are requested.
- **CVWriterAgent / CoverLetterWriterAgent**: Generate tailored drafts using LangChain chat models with iterative refinement based on critique feedback.
- **CritiqueAgent**: Evaluates generated documents for quality, ATS compatibility, and job alignment, marks each of the ad's requirements as shown, a gap, or missing from the draft, and gives improvement instructions. It runs on a model different from the writers' (`critique` in the config). Its grounding check, a separate call (`grounding` in the config, also on a non-writer model), lists claims about you that your material does not support and claims about the employer that the ad and company research do not support.
- **CVPDFGenerator / CoverLetterPDFGenerator**: Convert generated text documents to professional PDFs using LLM-generated LaTeX. Supports style customization, error fixing, and automatic versioning.
- **LatexFixer**: LLM-based agent that automatically fixes LaTeX compilation errors by analyzing error logs and generating corrected LaTeX code.
- **UserInputAgent**: Collects free-form responses when more clarity or feedback is required.

All agents share a common `State` TypedDict managed by LangGraph, which keeps track of messages, extracted info, generated artifacts, critique feedback, and version history. The result is a cohesive AI agentic system where multiple specialized models collaborate.

Prerequisites
-------------

- Python 3.12 or newer (`pyproject.toml` requires `>=3.12`; `.python-version` pins 3.12).
- [uv](https://docs.astral.sh/uv/) to install the dependencies from `uv.lock`.
- An OpenAI API key (paid, pay-per-use). The shipped `config/agent_models.yaml` uses `gpt-6-luna` for every agent except the critic's two calls (the quality review and the invented-claims check), which use `gpt-5.6-luna` (a session on the sample files is estimated at about 4 to 6 US cents); you can switch agents to other models there. To try the program without paid keys, see [Running with a local model (Ollama)](#running-with-a-local-model-ollama).
- A Tavily API key for company research (paid, with a limited free tier). Without one, company research is skipped.
- A LaTeX distribution for PDF generation, with the `pdflatex` command available: MacTeX or BasicTeX on macOS (`brew install --cask basictex`), TeX Live on Linux (`sudo apt install texlive-latex-extra`), or MiKTeX on Windows.
- macOS/Linux (Windows works but paths/logging instructions assume POSIX).

Project Setup
-------------

```bash
git clone https://github.com/mjordanic/CV_editor.git
cd CV_editor

# create .venv and install the exact dependency versions from uv.lock
uv sync
```

Graphviz is **not** needed to run the program. It is only needed if you want to redraw the workflow picture (see [Drawing the workflow graph](#drawing-the-workflow-graph)).

Environment Variables
---------------------

Copy the sample file and fill in your keys (or set the variables in your shell):

```bash
cp .env_sample .env
```

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

`dotenv` loads these before any agent imports, so keep `.env` up to date.

Running the Agent
-----------------

```bash
uv run python langgraph_agent.py
```

(or `source .venv/bin/activate` and then `python langgraph_agent.py`)

`uv run python langgraph_agent.py --help` lists the options without starting a run. `uv run pytest` runs the offline tests (no API keys needed).

The program will:

1. Load user CV/cover-letter data from `data/CV/`.
2. Load your portfolio from `data/portfolio/portfolio.txt`, if you put one there.
3. Ask you to paste the target job description (or provide a file path).
4. Fetch company insights via Tavily.
5. Give the writers your portfolio (or, for a large one, its items relevant to the job).
6. Chat with you about what to generate next.
7. Produce tailored documents and save them to `data/generated_CVs/`.

While streaming, the CLI may pause with prompts such as “Please paste the job description…” or “Please provide your input:”. Paste multiline text and finish with `Ctrl+D`, two empty lines in a row, or `END` on its own line.

**Tip:** For long job descriptions that might exceed terminal buffer limits, you can save the text to a file (e.g., `job_description/sample_job_ad.txt`) and simply paste the file path when prompted.

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

1. **Document ingestion**: `DocumentReaderAgent` loads `data/CV/CV.(txt|md|pdf)`, `data/CV/cover_letter.(txt|md|pdf)` and `data/CV/notes.(txt|md|pdf)`. If there is neither a CV nor notes there (or they have no text, e.g. a scanned PDF), it asks you to paste your CV (a rough draft is fine) or give a path to it.
2. **Job description capture**: `JobDescriptionAgent` interrupts for pasted text or file path (e.g., `job_description/sample_job_ad.txt`), extracts company name, work type, requirements, etc. An empty answer or a path to a missing file is asked again. If the ad has neither a job title nor any requirements, you are asked once to add details or reply `continue`.
3. **Company research**: if the ad names the company, `SearchAgent` builds a Tavily query and summarizes the results with an LLM (results about another company are not used). From the job ad and the results together it extracts the work arrangement (remote / hybrid / on-site), location or country limits and culture signals, shown on the console and given to the writers; anything neither source states stays "not stated" and is never invented. If you stated a work-arrangement preference that the job does not match, the assistant tells you; without a stated preference you are treated as open to the job's arrangement.
4. **Portfolio**: `ExperienceRetrievalAgent` gives the writers your portfolio (`data/portfolio/portfolio.txt`, if you supplied one) whole. Only a portfolio over 12,000 characters is searched for items relevant to the job description, with vector similarity (threshold defined in `config/agent_models.yaml`) and an LLM to verify relevance.
5. **Routing decision**: `RouterAgent` considers chat history, existing outputs, and user feedback to pick the next node.
6. **Generation**:
   - `CVWriterAgent` creates or updates a CV using the selected LLM.
   - `CoverLetterWriterAgent` creates or updates a cover letter (consistent with the freshly generated CV).
   - The first draft of each document, when you have not asked for anything specific, is written fresh for this ad, with your existing CV or earlier letter used only as source material. A request of yours (also in your first message) is applied as you ask, keeping the rest.
   - Every draft is written with your original CV, earlier letter, notes, portfolio and what you said in the chat in view, as source facts next to the document being edited.
7. **Quality assessment and refinement**:
   - `CritiqueAgent` evaluates generated documents on three dimensions: content quality, ATS compatibility, and job alignment.
   - Provides a quality score (0-100) and actionable improvement instructions.
   - All document iterations are saved to version history with their quality scores as draft files.
   - If the draft has a real defect (a claim your material does not support, a critical issue, or a requirement of the ad you meet but the draft does not show), it is refined automatically, up to `workflow.max_refinements` times (1 in the shipped config). After that it gets another pass only while it still makes unsupported claims or the last redraft showed more of the ad's requirements, and never more than `workflow.max_refinements_hard_cap` (2) times in total. Set `max_refinements: 0` to turn automatic refinement off. Requirements you do not meet are listed to you as gaps; they are never faked.
   - The critic's suggestions are checked against your earlier requests for that document (a suggestion that conflicts with one is dropped); the removal of unsupported claims is never dropped.
8. **Final selection**: Of the drafts written for the current request (drafts from earlier requests are not considered), the one with the fewest unsupported claims is kept, then the one that shows the most of the ad's requirements; between drafts still tied, the critic compares the last two side by side. If the chosen draft still has unsupported claims, one removal-only pass deletes them; any claim that remains is listed to you. If the final text still has placeholders such as `[EMAIL]`, you are asked once for the details (or which to leave out) before the PDF is made.
9. **PDF Generation**: 
   - `CVPDFGenerator` and `CoverLetterPDFGenerator` convert the final text documents to professional PDFs.
   - The LLM generates complete LaTeX documents based on the text content and any style preferences.
   - LaTeX compilation errors are automatically fixed using the `LatexFixer` agent (up to a configurable number of iterations).
   - Each generation creates a draft version (`generated_CV_draft_N.pdf`), with the latest always available as `generated_CV.pdf`.
   - The text of the compiled PDF is compared word by word with the checked plain text. If they differ, the document is converted once more and the closer PDF is kept; if it still differs, you are told which words differ.
10. **User feedback loop**: 
   - The router intelligently extracts content and style feedback from user messages.
   - You can provide feedback for both CV and cover letter in a single message.
   - Content feedback modifies the text, while style feedback modifies PDF appearance.
   - If both documents are requested, the system generates them sequentially without additional prompts.
   - `UserInputAgent` interrupts the graph when clarification is needed, allowing you to provide instructions that start a new draft and refinement cycle (earlier drafts stay on disk).

Logging and Debugging
---------------------

- **Structured logs**: Stored in `logs/cv_editor_<timestamp>.log`.
- **Debug transcripts**: `debug/cv_editor_<timestamp>.log` captures prompts, state snapshots, and generated text segments for auditing.
- **Graph visualization**: Not drawn on a normal run. See [Drawing the workflow graph](#drawing-the-workflow-graph).

Configuration
-------------

The system uses a centralized YAML configuration file (`config/agent_models.yaml`) to manage AI models and workflow settings. This approach provides a single source of truth for all configuration, making it easy to adjust behavior without code changes.

**Key Configuration Options:**
- **Agent Models**: Each agent (CV writer, cover letter writer, router, critique, etc.) can be configured with specific models and temperatures.
- **Refinement**: `workflow.max_refinements` sets how many automatic refinement rounds a draft with a real defect gets (shipped: `1`; `0` turns refinement off). `workflow.max_refinements_hard_cap` (shipped: `2`) allows extra rounds only while the draft still makes unsupported claims or the last redraft showed more of the ad's requirements, and is never exceeded.
- **LaTeX Fixer**: Configure `max_latex_fix_iterations` in the `workflow` section to control how many times the system attempts to fix LaTeX compilation errors automatically (shipped value: 2; the code falls back to 1 if the key is missing).
- **Critic models**: the critic makes two separate calls, the quality review (`agents.critique`) and the invented-claims check (`agents.grounding`). Both run on a model different from the writers' so the critic does not share their blind spots (shipped: `gpt-5.6-luna` for both, roughly +$0.02-0.03 per session).
- **RAG Settings**: Configure vector similarity thresholds and relevance verification settings for experience retrieval (used only for a portfolio over 12,000 characters).

All agents load configuration at startup from the YAML file. No code changes needed to adjust models or workflow parameters.

### Testing mode and evaluation

`evals/` runs the program end to end from prepared inputs and scores the results with an independent model. See [evals/README.md](evals/README.md).

- **Testing mode** (`python -m evals.run_case`): one session from files you give (any subset of CV, portfolio, earlier cover letter and notes, plus the job ad), with a scripted user answering the questions and prepared web-search results replacing live Tavily calls (`CV_EDITOR_SEARCH_FIXTURE`). Every run writes a folder with the exact inputs, the config and model of every agent, every graph step and router decision, every model call with prompt, response, tokens and cost, the critique scores and invented-claims results of every draft, the final texts and PDFs, timings and errors.
- **Recording search results** (`python -m evals.search_fixtures record`, or `CV_EDITOR_SEARCH_RECORD=file.json` on a normal run): saves live Tavily answers in the fixture format, so a real company is searched once and replayed after that.
- **Evaluator** (`python -m evals.evaluate RUN_DIR`): scores one run with `openai:gpt-6.1-sol` (configurable), blind to the model and inputs that produced it: invented content with quotes, completeness, fit to the ad, CV standards, automatic screening (ATS) and the cover letter.
- **Fixtures** (`evals/fixtures/`): eight anonymized or invented job ads, each with company pages and a search fixture, and thirteen fully fictional candidates, each with their own portfolio. Six ads and eleven candidates are in IT; a butcher (English ad for a Zagreb supermarket) and a home-care assistant (German ad and CV) are outside it. Every evaluation input is tracked here, so anyone can repeat the tests.
- **Matrix** (`python -m evals.run_matrix --out DIR`): runs every case in `evals/matrix.json` (persona × ad × input set × model, some with a scripted revision turn), then the evaluator on each, with a spending cap. `--repeats N` and `--rescore K` measure the spread between runs and between scorings; `--truth-check` also checks each run against the persona's recorded gaps and traps (`evals/score_truth.py`), and `python -m evals.summarize DIR` reports means and spread. Inputs come only from `evals/fixtures/`.

### Running with a local model (Ollama)

For testing without paid API calls you can run every agent on a local [Ollama](https://ollama.com) model:

```bash
ollama pull gemma4:e4b            # or another model, see below
ollama pull nomic-embed-text      # optional: retrieval for a portfolio over 12,000 characters (RAG); without it RAG is skipped
CV_EDITOR_MODEL=ollama:gemma4:e4b uv run python langgraph_agent.py
```

- No OpenAI or Tavily key is needed. Company research is skipped without a Tavily key.
- Local models get a 16k-token context (`CV_EDITOR_OLLAMA_NUM_CTX` to change it) and run with thinking off.
- Expect it to be slow and rougher than the hosted model. On an M1 with 16 GB, a full session on the sample files took about 30 minutes with `gemma4:e4b`. That model also kept invented skills that the critique had flagged, and wrote LaTeX that did not compile, so no PDFs were produced. Use local runs to test the workflow, not to judge the output.
- On 16 GB of memory, models up to about 8 GB on disk fit next to the operating system: `gemma4:e4b-it-qat` (6.1 GB), `qwen3:8b` (5.2 GB) or `gemma4:12b-it-qat` (7.2 GB, best quality of these but slowest).

Customization Tips
------------------

- **Models**: Change model IDs or temperatures in `config/agent_models.yaml`. `temperature: null` uses the model's default (`gpt-6-luna` accepts no other value). Ensure the chosen model is available in your OpenAI organization. `CV_EDITOR_MODEL=<provider:model>` replaces the writer-side models for one run without editing the file, never the critic's. The critic is meant to run on a different model from the writers', so `CV_EDITOR_CRITIC_MODEL=<provider:model>` replaces the quality review's model (`critique`) and `CV_EDITOR_GROUNDING_MODEL=<provider:model>` replaces the fact check's (`grounding`). A local `CV_EDITOR_MODEL` (an `ollama:` model) is the exception: it also replaces the critic's models unless their own variables are set, so a local run sends nothing to OpenAI and needs no OpenAI key; the log then says the checker shares the writer's model.
- **Prompt tone**: Update the system/human prompt strings in each agent to reflect a different writing style or formatting rules.
- **PDF Templates**: Base LaTeX templates are in `templates/` directory. The LLM uses these as starting points but can modify them completely based on your style preferences.
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

- **Graphviz / `pygraphviz` build errors**: Only relevant for `uv sync --extra graph`; see [Drawing the workflow graph](#drawing-the-workflow-graph). A plain `uv sync` does not need Graphviz.
- **Tracing interrupts**: If the CLI seems stuck, make sure you finished input with `Ctrl+D` or typed `END` on its own line.
- **LaTeX compilation errors**: If PDF generation fails repeatedly, check the logs for LaTeX error messages. The system will attempt to fix errors automatically, but some issues may require manual template adjustments.
- **Missing pdflatex**: Ensure `pdflatex` is in your PATH. On macOS, install with `brew install --cask basictex` (or `brew install texlive`); on Linux, `sudo apt install texlive-latex-extra`. Verify with `which pdflatex`.

Drawing the workflow graph
--------------------------

![LangGraph workflow](images/graph_visualization.png)

The picture above shows how the agents are wired together. A normal run does not touch it. To redraw it after changing the graph, install the Graphviz system library and the optional `graph` extra, then run with `--draw-graph` (this overwrites `images/graph_visualization.png`):

```bash
# macOS (Homebrew)
brew install graphviz
CFLAGS="-I$(brew --prefix graphviz)/include" LDFLAGS="-L$(brew --prefix graphviz)/lib" uv sync --extra graph

# Debian/Ubuntu
sudo apt install graphviz graphviz-dev
uv sync --extra graph

uv run python langgraph_agent.py --draw-graph
```

Questions or ideas? Open an issue on GitHub.

Licence
-------

MIT, see [LICENSE](LICENSE). Made by Mislav Jordanic.


TODO list
---------
- ~~**PDF generation agent**: Add an agent that uses LaTeX to generate professional PDF versions of CVs and cover letters.~~ ✅ **Completed**: CVPDFGenerator and CoverLetterPDFGenerator implemented with customizable LaTeX templates.
- **Enhanced candidate information system**: The candidate's CV, cover letter, free-form notes and portfolio can be given (see [What candidate information the program can take](#what-candidate-information-the-program-can-take)). This should be improved to:
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

