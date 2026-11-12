from tavily import TavilyClient
from typing import Optional
from datetime import datetime, timezone
import os
import json
import logging
from langchain.chat_models import init_chat_model
from config.config_loader import is_local_model, local_model_kwargs
from langchain_core.prompts import ChatPromptTemplate
from debug_utils import write_to_debug
from agents.work_context import NOT_STATED, WORK_CONTEXT_PROMPT, WorkContext
from agents.job_description import field_value

logger = logging.getLogger(__name__)

# Prepared web-search results (testing mode). A fixture file holds the raw Tavily answers for one
# company, so a session can be replayed without live searches; see evals/README.md.
SEARCH_FIXTURE_ENV = "CV_EDITOR_SEARCH_FIXTURE"   # replay: read answers from this file, never call Tavily
SEARCH_RECORD_ENV = "CV_EDITOR_SEARCH_RECORD"     # record: call Tavily and save each answer to this file
FIXTURE_VERSION = 1


def load_search_fixture(path: str) -> dict:
    """Read a search fixture: {"fixture_version", "company", "searches": [{"query", "response"}]}."""
    with open(path, "r", encoding="utf-8") as f:
        fixture = json.load(f)
    if not isinstance(fixture, dict) or not isinstance(fixture.get("searches"), list) or not fixture["searches"]:
        raise ValueError(f"{path} is not a search fixture (it needs a non-empty 'searches' list)")
    return fixture


def append_to_search_fixture(path: str, query: str, response: dict, company: Optional[str] = None) -> None:
    """Add one live search answer to a fixture file (created if missing)."""
    fixture = {"fixture_version": FIXTURE_VERSION, "company": company, "source": "tavily", "searches": []}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            fixture = json.load(f)
    fixture.setdefault("searches", []).append({
        "query": query,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "response": response,
    })
    if company and not fixture.get("company"):
        fixture["company"] = company
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=2, ensure_ascii=False)


QUERY_TEMPLATE = (
    "Find the most relevant information about the following company, including company culture, values, and working environment: "
    "\n {company_information}\n"
)

COMPANY_DESCRIPTION_SYSTEM_PROMPT = (
    "You are a helpful assistant that creates concise company descriptions based on search results."
)

COMPANY_DESCRIPTION_HUMAN_PROMPT = (
    "The company named in the job ad is: {company_name}.\n\n"
    "Based on the following search results, create a short (max 2000 characters) description "
    "about that company, focusing on what the company does, its industry, its values, and any notable "
    "characteristics or values. This description will be used to tailor a CV and cover letter for a job "
    "application so make sure to include all the relevant information about core values and company culture and working environment. "
    "Use only results that are about this company. If none of them are, answer only: "
    "\"Could not identify the company in the search results.\"\n\n"
    "Here are the search results: \n\n{search_content}"
)

NO_COMPANY_INFO = "No information found about the company."
NO_COMPANY_NAME = "The job ad names no company, so no company research was done."

# Characters of page text kept per search result. The app's summary and the evaluator read the same
# text (search_result_texts), so the evaluator never judges against facts the writers could not see.
CHARS_PER_RESULT = 3000
MAX_SEARCH_CONTENT_CHARS = 15000


def search_result_texts(results: dict | None, chars_per_result: int = CHARS_PER_RESULT) -> list[dict]:
    """
    The text of each search result: its full page text (Tavily raw_content) trimmed to
    chars_per_result, or the short snippet when there is no page text. Returns dicts with
    'title', 'url' and 'text', in result order, skipping results without text.
    """
    out = []
    for r in (results or {}).get("results") or []:
        body = (r.get("raw_content") or r.get("content") or "").strip()
        if body:
            out.append({"title": r.get("title", ""), "url": r.get("url", ""), "text": body[:chars_per_result]})
    return out

def build_search_query(job_description_info: dict) -> str:
    """The Tavily query for the company named in the extracted job ad fields."""
    company_info_dict = {
        "name": job_description_info.get("company_name"),
        "website": job_description_info.get("company_website"),
        "location": job_description_info.get("location"),
        "industry": job_description_info.get("industry")
    }
    info_parts = []
    for key in ["name", "website", "location", "industry"]:
        if company_info_dict.get(key):
            info_parts.append(f"{key.title()}: {company_info_dict[key]}")
    company_info_str = "; ".join(info_parts) if info_parts else "No additional information provided"
    # limit the length of the company information because Tavily API has a limit of 400 characters
    return QUERY_TEMPLATE.format(company_information=company_info_str[:250])


class SearchAgent:
    """Agent responsible for performing company research via Tavily and summarizing results."""
    def __init__(
        self,
        model: str = "openai:gpt-5-nano",
        temperature: float = 0.0,
        fixture_path: Optional[str] = None,
        record_path: Optional[str] = None,
    ):
        """
        Initialize the search agent and supporting LLM/Tavily clients.

        Args:
            model: The LLM model identifier to use
            temperature: Temperature setting for the LLM
            fixture_path: Replay prepared search results from this fixture file instead of calling
                Tavily (default: $CV_EDITOR_SEARCH_FIXTURE).
            record_path: Call Tavily and save every answer to this fixture file
                (default: $CV_EDITOR_SEARCH_RECORD).

        Returns:
            None
        """
        logger.info("Initializing SearchAgent...")
        
        self.llm = init_chat_model(model, temperature=temperature, **local_model_kwargs(model))
        logger.debug(f"SearchAgent LLM initialized - model: {model}, temperature: {temperature}")
        self.fixture_path = fixture_path or os.getenv(SEARCH_FIXTURE_ENV) or None
        self.record_path = record_path or os.getenv(SEARCH_RECORD_ENV) or None
        self.fixture = load_search_fixture(self.fixture_path) if self.fixture_path else None
        if self.fixture is not None:
            logger.info(f"Testing mode: company research replays {self.fixture_path} (no live web search)")
            self.tavily_client = None
            return
        # A local model means the run sends nothing to a cloud service, so no web search either,
        # even when a TAVILY_API_KEY is set (e.g. copied from .env_sample).
        if is_local_model(model):
            self.tavily_client = None
            self.skip_reason = "a local model is in use, so no web search is made"
            logger.info(f"Company research is skipped: {self.skip_reason}")
            return
        # Initialize TavilyClient lazily to ensure environment variables are loaded
        # Without a key company research is skipped, not fatal.
        api_key = os.getenv("TAVILY_API_KEY")
        self.tavily_client = TavilyClient(api_key=api_key) if api_key else None
        if self.tavily_client is None:
            self.skip_reason = "TAVILY_API_KEY is not set"
            logger.warning("TAVILY_API_KEY is not set - company research will be skipped")

    def search_tavily(self, query: str):
        """
        Execute a Tavily search for company research.

        Args:
            query: Natural-language search string describing the target company.

        Returns:
            dict: Raw Tavily response containing search results payload.
        """
        if self.fixture is not None:
            # Replay: the answer recorded for this exact query, else the first one (the query is
            # built from the model's reading of the ad, so it can differ slightly between runs).
            searches = self.fixture["searches"]
            match = next((s for s in searches if s.get("query") == query), searches[0])
            logger.info(f"Replaying prepared search results from {self.fixture_path} (exact query match: {match.get('query') == query})")
            return match.get("response") or {}
        if self.tavily_client is None:
            raise RuntimeError("TAVILY_API_KEY is not set")
        logger.info(f"Searching Tavily with query: {query}")
        results = self.tavily_client.search(
            query=query, 
            search_depth="advanced", 
            max_results=5, 
            include_images=False,
            include_raw_content="text"
        )
        if self.record_path:
            append_to_search_fixture(self.record_path, query, results)
            logger.info(f"Recorded the search answer to {self.record_path}")
        num_results = len(results.get('results', [])) if results else 0
        logger.info(f"Tavily search completed - found {num_results} results")
        logger.debug(f"Search results keys: {list(results.keys()) if results else 'None'}")
        return results

    def _extract_search_content(self, results: dict) -> str:
        """
        Combine the page text of the Tavily results (see search_result_texts) into a single string.

        Args:
            results: Tavily response dictionary returned by `search_tavily`.

        Returns:
            str: Concatenated textual content extracted from individual results.
        """
        return "\n\n".join(r["text"] for r in search_result_texts(results))

    def process_search_results(self, results: dict, company_name: Optional[str] = None):
        """
        Summarize Tavily results into a concise company description.

        Args:
            results: Tavily response dictionary returned by `search_tavily`.

        Returns:
            Any: LangChain response object whose `content` holds the description text.
        """
        logger.info("Processing search results to generate company description")
        combined_content = self._extract_search_content(results)
        
        if not combined_content:
            logger.warning("No search content extracted - returning default message")
            return NO_COMPANY_INFO
        
        logger.debug(f"Extracted search content length: {len(combined_content)} characters")
        
        # Create prompt for generating company description
        prompt = ChatPromptTemplate.from_messages([
            ("system", COMPANY_DESCRIPTION_SYSTEM_PROMPT),
            ("human", COMPANY_DESCRIPTION_HUMAN_PROMPT)
        ])
        
        # Generate description using OpenAI
        chain = prompt | self.llm
        llm_input = {
            "search_content": combined_content[:MAX_SEARCH_CONTENT_CHARS],  # limit content to avoid token limits
            "company_name": company_name or "(not given)",
        }
        logger.info("Calling LLM to generate company description...")
        logger.debug(f"LLM input - search_content length: {len(llm_input['search_content'])} characters")
        
        response = chain.invoke(llm_input)
        
        response_content = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"LLM response received - company description length: {len(response_content)} characters")
        logger.debug(f"Company description preview: {response_content[:200]}...")
        
        return response

    def extract_work_context(self, job_ad: str, results: dict) -> WorkContext:
        """
        Work arrangement, location limits and culture signals from the job ad and the search
        results together (one model call). Fields neither text states are 'not stated'.
        """
        chain = ChatPromptTemplate.from_messages([("human", WORK_CONTEXT_PROMPT)]) | self.llm.with_structured_output(WorkContext)
        return chain.invoke({
            "job_ad": (job_ad or "(no job ad text)")[:6000],
            "search_content": self._extract_search_content(results or {})[:MAX_SEARCH_CONTENT_CHARS] or "(no search results)",
        })

    def run(self, state):
        """
        Run search with company information from state and update state with company info.

        Args:
            state: LangGraph state dictionary containing `job_description_info`.

        Returns:
            dict: Partial state update with `company_info` and assistant message.
        """
        logger.info("SearchAgent.run() called")
        
        job_description_info = state.get("job_description_info")
        if not job_description_info:
            raise ValueError("No job_description_info in state - cannot perform search")

        
        logger.debug(f"Job description info extracted - company: {job_description_info.get('company_name')}, website: {job_description_info.get('company_website')}, location: {job_description_info.get('location')}, industry: {job_description_info.get('industry')}")
        
        formatted_query = build_search_query(job_description_info)
        logger.debug(f"Formatted query: {formatted_query}")

        # Execute search. Company research is optional context: if the web search or its summary
        # fails (network, rate limit, quota), carry on without it instead of ending the session
        # after the user has already pasted the job ad.
        company_name = field_value(job_description_info.get("company_name"))
        search_source = None  # set only when a search is made (recorded fixture or Tavily)
        if not company_name:
            # Without a name the query is only a location or "No additional information provided",
            # and the summary of whatever comes back reached the writers as "Company Information".
            logger.info("The job ad names no company - skipping company research")
            results = {}
            company_description = NO_COMPANY_NAME
            formatted_query = None
        elif getattr(self, "fixture", None) is None and getattr(self, "skip_reason", None):
            logger.info(f"Company research skipped: {self.skip_reason}")
            results = {}
            company_description = NO_COMPANY_INFO
            formatted_query = None
        else:
            search_source = f"fixture:{self.fixture_path}" if getattr(self, "fixture", None) is not None else "tavily"
            try:
                results = self.search_tavily(formatted_query) or {}
                company_description = self.process_search_results(results, company_name)
            except Exception as e:
                logger.warning(f"Company research failed ({type(e).__name__}: {e}) - continuing without company information")
                results = {}
                company_description = NO_COMPANY_INFO
        # Arrangement and culture come from the ad and the search results together, so a failed
        # search still leaves what the ad says.
        try:
            work_context = self.extract_work_context(job_description_info.get("job_description") or "", results).model_dump()
        except Exception as e:
            logger.warning(f"Work arrangement extraction failed ({type(e).__name__}: {e}) - using the ad's work type only")
            ad_type = job_description_info.get("work_type")
            work_context = WorkContext(arrangement=ad_type if ad_type in ("remote", "hybrid", "on-site") else NOT_STATED).model_dump()
        
        # Extract search content for debug logging
        extracted_search_content = self._extract_search_content(results)
        company_description_str = (
            company_description.content if hasattr(company_description, "content") else str(company_description)
        )
        
        # Write debug info to file
        debug_content = ""
        debug_content += "EXTRACTED SEARCH CONTENT:\n"
        debug_content += "-" * 80 + "\n"
        debug_content += extracted_search_content if extracted_search_content else "(no search content)"
        debug_content += "\n\n"
        debug_content += "COMPANY DESCRIPTION:\n"
        debug_content += "-" * 80 + "\n"
        debug_content += company_description_str if company_description_str else "(no description)"
        debug_content += "\n\n"
        debug_content += "WORK ARRANGEMENT AND CULTURE:\n"
        debug_content += "-" * 80 + "\n"
        debug_content += json.dumps(work_context, ensure_ascii=False, indent=2)
        debug_content += "\n\n"
        
        write_to_debug(debug_content, "SEARCH AGENT DEBUG INFO")
        logger.info("Search debug info written to debug file")
        
        # Format company info for state
        company_info = {
            "company_description": company_description_str,
            "work_context": work_context,
            "search_results": results,  # Store raw search results if needed
            "search_query": formatted_query,
            "search_source": search_source,
        }
        
        logger.info(f"Company info prepared - arrangement: {work_context['arrangement']}, description_length: {len(company_description_str)}")
        culture = "; ".join(work_context.get("culture_signals") or []) or NOT_STATED
        summary = (f"Work arrangement (from the ad and company pages): {work_context['arrangement']}. "
                   f"Location or country limits: {work_context.get('location_limits') or NOT_STATED}. "
                   f"Culture signals: {culture}.")
        logger.info(summary)  # shown on the console, like the job ad summary
        
        return {
            "company_info": company_info,
            "messages": state.get("messages", []) + [{
                "role": "assistant",
                "content": f"Company information retrieved. {summary}"
            }],
        }
