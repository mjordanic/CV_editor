# How CV Editor works

The technical companion to the [README](../README.md): the workflow step by step, the agents, the files the
program reads and writes, configuration, environment variables and customisation. For how well it works, see
[How well does it work?](../README.md#how-well-does-it-work) in the README and the evaluation in
[`evals/README.md`](../evals/README.md).

## Contents

1. [The workflow, step by step](#the-workflow-step-by-step)
2. [The agents](#the-agents)
3. [Files and folders](#files-and-folders)
4. [Configuration](#configuration)
5. [Environment variables](#environment-variables)
6. [Customisation](#customisation)
7. [Logging and debugging](#logging-and-debugging)

The workflow is a LangGraph graph. The picture in the README's
[Drawing the workflow graph](../README.md#drawing-the-workflow-graph) section shows how the steps are wired; it is
redrawn from the code with `--draw-graph`.

## The workflow, step by step

1. **Document ingestion.** `DocumentReaderAgent` loads `data/CV/CV.(txt|md|pdf)`,
   `data/CV/cover_letter.(txt|md|pdf)` and `data/CV/notes.(txt|md|pdf)`. If there is neither a CV nor notes there
   (or they have no text, e.g. a scanned PDF), it asks you to paste your CV (a rough draft is fine) or give a path
   to it.
2. **Job description capture.** `JobDescriptionAgent` pauses the graph for the pasted ad text or a file path (e.g.
   `job_description/sample_job_ad.txt`) and extracts the company name, work type, requirements, etc. An empty answer
   or a path to a missing file is asked again. If the ad has neither a job title nor any requirements, you are asked
   once to add details or reply `continue` (the `clarify_job_ad` step).
3. **Company research.** If the ad names the company, `SearchAgent` builds a Tavily query and summarises the results
   with a model (results about another company are not used). Only that query goes to Tavily, never the CV. Without a
   `TAVILY_API_KEY`, or when the search agent runs on a local model, no web search is made and this step is skipped
   with a log line. From the job ad and the results together it extracts the work arrangement (remote / hybrid /
   on-site), location or country limits and culture signals, shows them on the console and gives them to the writers;
   anything neither source states stays "not stated" and is never invented. If you stated a work-arrangement
   preference that the job does not match, the assistant tells you; without a stated preference you are treated as
   open to the job's arrangement.
4. **Portfolio.** `ExperienceRetrievalAgent` gives the writers your portfolio (`data/portfolio/portfolio.txt`, if
   you supplied one) whole. Only a portfolio over 12,000 characters is indexed (one paid embedding call, or a local
   Ollama embedding model on a local run) and searched for the items relevant to the job description: a two-stage
   filter of vector similarity (threshold in `config/agent_models.yaml`) and a model that verifies relevance
   (retrieval-augmented generation, RAG). Without a portfolio it logs "No portfolio".
5. **Routing decision.** `RouterAgent` reads the chat history, the existing outputs and your feedback, and picks the
   next step: write the CV, write the cover letter, change a PDF's appearance, ask you something, or exit. If you ask
   for both documents in one message, it writes them one after the other without asking again in between.
6. **Generation.**
   - `CVWriterAgent` creates or updates a CV; `CoverLetterWriterAgent` creates or updates a cover letter
     (consistent with the freshly generated CV).
   - The documents are written in the language of the job ad, even when the job is in another country; ask for
     another language ("write the cover letter in English") to change it.
   - The first draft of each document, when you have not asked for anything specific, is written fresh for this ad,
     with your existing CV or earlier letter used only as source material. The first letter for a new job uses the
     old letter only for facts and voice and is readdressed to this employer, role and addressee. A request of yours
     (also in your first message) is applied as you ask, keeping the rest.
   - Every draft is written with your original CV, earlier letter, notes, portfolio and what you said in the chat in
     view, as source facts next to the document being edited.
7. **Quality assessment and refinement.**
   - `CritiqueAgent` reviews each draft on three dimensions: content quality, ATS (applicant tracking system)
     compatibility, and job alignment. It gives a quality score (0-100), marks each of the ad's requirements as
     shown, a gap, or missing from the draft, and gives improvement instructions. It runs on a model different from
     the writers' (`critique` in the config).
   - Its grounding check, a separate call (`grounding` in the config, also on a non-writer model), lists claims
     about you that your material does not support and claims about the employer that the ad and company research
     do not support.
   - Every draft is saved as a text file (`data/generated_CVs/generated_CV_draft_N.txt` and the cover letter's
     equivalent), so earlier versions stay on disk. Its quality score is kept in the session's draft history and
     the log, not in the file.
   - If the draft has a real defect (a claim your material does not support, a critical issue, or a requirement of
     the ad you meet but the draft does not show), it is redrafted automatically, up to `workflow.max_refinements`
     times (1 in the shipped config). After that it gets another pass only while it still makes unsupported claims
     or the last redraft showed more of the ad's requirements, and never more than
     `workflow.max_refinements_hard_cap` (2) times in total. Set `max_refinements: 0` to turn automatic refinement
     off. The critic's 0-100 score is shown but does not decide. Requirements you do not meet are listed to you as
     gaps; they are never faked.
   - The critic's suggestions are checked against your earlier requests for that document (a suggestion that
     conflicts with one is dropped); the removal of unsupported claims is never dropped.
   - A cover letter draft that repeats the CV (in whole or in large part) is sent back for a redraft before the
     critic sees it, within the same budget. The check needs no model call: it measures how much of the CV's wording
     the letter repeats (`agents/letter_cv_check.py`).
8. **Final selection and honesty checks.**
   - Only the drafts written for the current request count (drafts from earlier requests are not considered).
   - A cover letter that repeats the CV is kept only if every draft does, and then the CV part is cut out of it
     automatically (or, if that is not possible, you are warned).
   - Otherwise the draft with the fewest unsupported claims is kept (a draft whose invented-claims check could not
     run ranks below checked ones), then the one that shows the most of the ad's requirements; between drafts still
     tied, the critic compares the last two side by side.
   - If the chosen draft still has unsupported claims, one removal-only pass deletes them and the check runs again;
     any claim that remains is listed to you word for word, and if the check could not run on the final draft you
     are told so.
   - If the final text still has placeholders such as `[EMAIL]`, you are asked once for the details (or which to
     leave out) before the PDF is made (the `ask_cv_missing_details` and `ask_cover_letter_missing_details` steps).
9. **PDF generation.**
   - `CVPDFGenerator` and `CoverLetterPDFGenerator` have a model write a complete LaTeX document (preamble and
     content, starting from the templates in `templates/`) from the final text and any style preferences, then
     compile it with `pdflatex`.
   - Compilation errors are fixed by the `LatexFixer` agent, which reads the error log and rewrites the LaTeX, up to
     `max_latex_fix_iterations` times (2 in the shipped config).
   - The text of the compiled PDF is compared word by word with the checked plain text, numbers exactly. If they
     differ, the document is converted once more and the closer PDF is kept (changed numbers count most); if it
     still differs, you are told which words and numbers differ.
   - Each generation creates a draft version (`generated_CV_draft_N.pdf`), and the latest is always copied to
     `generated_CV.pdf` and `generated_cover_letter.pdf`.
10. **Your feedback.**
    - The router separates content feedback (changes the text) from style feedback (changes only the PDF's
      appearance), for the CV and the cover letter separately, so one message can carry both: "Add my research stay
      at Berkeley. Make titles blue and bolded."
    - An appearance-only request ("make it more compact", "use minimal spacing") changes the LaTeX and leaves the
      text as it is, without writing the content again. Earlier style requests are kept when you make a new one.
    - `UserInputAgent` pauses the graph when it needs your input. A new instruction starts a new draft and
      refinement cycle with a fresh redraft budget (earlier drafts stay on disk).

All steps share one `State` (a TypedDict in `langgraph_agent.py`) that LangGraph passes between them: messages,
extracted job and company information, generated documents, critique feedback and version history.

## The agents

| agent | step | what it does |
|---|---|---|
| `MasterAgent` (`langgraph_agent.py`) | all | builds the graph, streams it, and resumes it with your answers when it pauses for input |
| `DocumentReaderAgent` | 1 | loads your CV, earlier cover letter and notes from disk |
| `JobDescriptionAgent` | 2 | asks for the job ad and extracts structured fields from it |
| `SearchAgent` | 3 | queries Tavily (only when the ad names the company), summarises the results, extracts the work arrangement, location limits and culture signals |
| `ExperienceRetrievalAgent` | 4 | gives the writers your portfolio, or the relevant items of a large one (RAG with relevance verification) |
| `RouterAgent` | 5, 10 | decides the next step from the conversation; handles writing both documents in turn |
| `CVWriterAgent`, `CoverLetterWriterAgent` | 6 | write and redraft the documents (LangChain chat models) |
| `CritiqueAgent` | 7 | quality review and grounding (invented-claims) check, on a different model from the writers |
| `CVPDFGenerator`, `CoverLetterPDFGenerator` | 9 | model-written LaTeX, style changes, versioned PDFs |
| `LatexFixer` | 9 | repairs LaTeX that does not compile |
| `UserInputAgent` | 10 | collects free-form answers when the graph needs them |

## Files and folders

| path | content |
|---|---|
| `data/CV/` | your CV, earlier cover letter and notes (`CV.*`, `cover_letter.*`, `notes.*`; `.txt`, `.md` or text `.pdf`) |
| `data/portfolio/portfolio.txt` | your optional portfolio, one item per paragraph |
| `data/generated_CVs/` | every draft (`generated_CV_draft_N.*`, the cover letter likewise) and the latest PDFs, `generated_CV.pdf` and `generated_cover_letter.pdf` |
| `job_description/` | job ads you saved as files; two made-up samples are tracked |
| `logs/`, `debug/` | log files (see [Logging and debugging](#logging-and-debugging)) |

`data/` and `job_description/` are in `.gitignore` (the two sample ads are tracked explicitly), so your own CV and
ads are not committed by accident.

## Configuration

`config/agent_models.yaml` holds every agent's model and the workflow settings; every agent reads it at startup, so
no code change is needed to adjust them.

- **Agent models.** Each agent (CV writer, cover letter writer, router, critique, etc.) has a model and a
  temperature. The shipped file uses `gpt-6-luna` for every agent except the critic's two calls, which use
  `gpt-5.6-luna`. `temperature: null` uses the model's default (`gpt-6-luna` accepts no other value). Make sure the
  model you choose is available in your OpenAI organisation; any provider LangChain's `init_chat_model` supports
  works, e.g. `ollama:...`.
- **Critic models.** The critic makes two separate calls, the quality review (`agents.critique`) and the
  invented-claims check (`agents.grounding`). Both run on a model different from the writers' so the critic does not
  share their blind spots. Over the 49 app sessions of the 2026-11-11 pilot and core rounds, the critic's
  `gpt-5.6-luna` cost about $0.019 of the $0.028 session mean (`summary.json`, cost by model).
- **Refinement.** `workflow.max_refinements` sets how many automatic redrafts a draft with a real defect gets
  (shipped: `1`; `0` turns refinement off). `workflow.max_refinements_hard_cap` (shipped: `2`) allows extra rounds
  only while the draft still makes unsupported claims or the last redraft showed more of the ad's requirements, and
  is never exceeded.
- **LaTeX fixer.** `workflow.max_latex_fix_iterations` (shipped: `2`; the code falls back to 1 if the key is
  missing) is how many times a LaTeX compilation error is repaired automatically.
- **RAG settings.** Vector similarity threshold and relevance verification for experience retrieval (used only for a
  portfolio over 12,000 characters).

## Environment variables

Put these in `.env` (copied from `.env_sample`) or set them in your shell. `python-dotenv` loads `.env` before any
agent is built.

| variable | meaning |
|---|---|
| `OPENAI_API_KEY` | needed for the shipped OpenAI models. Without it, and without a local `CV_EDITOR_MODEL`, the program stops at startup with `OPENAI_API_KEY is not set` and how to set it |
| `TAVILY_API_KEY` | optional; company research is skipped without it, and on a local model |
| `ANTHROPIC_API_KEY` | for the evaluation tools when a Claude model grades (`evals/README.md`), and for the program when one of its models is an `anthropic:` model (checked at startup like `OPENAI_API_KEY`) |
| `CV_EDITOR_MODEL=<provider:model>` | replaces the writer-side models for one run without editing the config, never the critic's |
| `CV_EDITOR_CRITIC_MODEL=<provider:model>` | replaces the quality review's model (`critique`) |
| `CV_EDITOR_GROUNDING_MODEL=<provider:model>` | replaces the fact check's model (`grounding`) |
| `CV_EDITOR_TEMPERATURE` | a number (with a decimal point, e.g. `0.2`), or `default` to send none; applies only to the agents whose model one of the three variables above replaced. Anything else stops the program at startup with a message |
| `CV_EDITOR_OLLAMA_NUM_CTX` | context size for local Ollama models (default 16384) |
| `CV_EDITOR_EMBEDDING_MODEL` | the Ollama embedding model for a large portfolio on a local run (default `nomic-embed-text`; hosted runs use OpenAI's `text-embedding-3-small`). If it is not pulled, indexing fails with a warning and RAG is skipped |
| `CV_EDITOR_SEARCH_FIXTURE=file.json` | replays prepared company-search results instead of calling Tavily (testing mode) |
| `CV_EDITOR_SEARCH_RECORD=file.json` | saves the live Tavily answers in the fixture format |

A local `CV_EDITOR_MODEL` (an `ollama:` model) is the exception to "never the critic's": it also replaces the
critic's models unless `CV_EDITOR_CRITIC_MODEL` or `CV_EDITOR_GROUNDING_MODEL` is set, so a local run sends nothing
to OpenAI and needs no OpenAI key; the log then says the checker shares the writer's model. A local run also makes no
Tavily search, even when `TAVILY_API_KEY` is set.

## Customisation

- **Models.** See [Configuration](#configuration) and [Environment variables](#environment-variables).
- **Prompt tone.** Edit the system and human prompt strings in each agent for a different writing style or
  formatting rules.
- **PDF templates.** The base LaTeX templates are in `templates/`. The model uses them as starting points but can
  change them completely to follow your style requests ("Make all titles dark blue and bolded", "Use minimal spacing
  and make it more compact", "Change the font to Arial and increase margins").
- **State fields.** Extend `State` in `langgraph_agent.py` to carry more data between steps.
- **Another search provider.** Replace `SearchAgent.search_tavily` (`agents/search.py`) with another API, keeping
  the `company_info` shape.

## Logging and debugging

- **Logs.** `logs/cv_editor_<timestamp>.log` holds every line, including other libraries' debug output. The console
  shows the program's own lines (its questions arrive as `<timestamp> - __main__ - INFO -` lines) and other
  libraries' warnings and errors.
- **Debug transcripts.** `debug/cv_editor_<timestamp>.log` holds prompts, state snapshots and generated text for
  auditing. Running the tests also creates files here.
- **Recording a session.** `uv run python langgraph_agent.py --trace-dir DIR` writes every step (`events.jsonl`),
  every model call with prompt, response, tokens and cost (`llm_calls.jsonl`), `summary.json` and `final_state.json`
  to `DIR`. API keys are never written. The evaluation's testing mode uses the same recording
  (`evals/README.md`, section 1).
- **Graph picture.** Not drawn on a normal run; see
  [Drawing the workflow graph](../README.md#drawing-the-workflow-graph).
