from typing import Optional, Literal
import logging
import json
from langchain.chat_models import init_chat_model
from config.config_loader import local_model_kwargs
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.types import interrupt
from debug_utils import write_to_debug
from agents.document_reader import ask_for_document

logger = logging.getLogger(__name__)


JOB_DESCRIPTION_EXTRACTION_SYSTEM_PROMPT = (
    "You are a helpful assistant that extracts structured information from job descriptions. "
    "Analyze the job description carefully and extract all relevant details about the company and position. Be terse and concise."
)

JOB_DESCRIPTION_EXTRACTION_HUMAN_PROMPT = (
    "Extract the following information from the job description:\n"
    "- Company name\n"
    "- Company website/web page\n"
    "- Location (city, state, country)\n"
    "- Industry\n"
    "- Whether the job is remote, hybrid, or on-site. If not specified, return 'unknown'.\n"
    "- Candidate minimal requirements (qualifications, skills, experience, education, etc.)\n"
    "- The ad's must-have requirements and its nice-to-have requirements, each as a list of short items in the ad's own terms\n"
    "- The language the job description is written in (e.g. English, German)\n"
    "- Any other relevant company or position details\n\n"
    "Job Description:\n\n{job_description}"
)


class JobDescriptionInfo(BaseModel):
    """
    Structured information extracted from a job description.

    The ad's own text is not a field here: structured output invited the model to type the whole ad
    again (expensive output tokens on every run). run() adds it to the dict as 'job_description'.
    """
    company_name: Optional[str] = Field(
        None,
        description="The name of the company posting the job"
    )
    company_website: Optional[str] = Field(
        None,
        description="The company's website URL or web page"
    )
    location: Optional[str] = Field(
        None,
        description="Job location (city, state, country)"
    )
    industry: Optional[str] = Field(
        None,
        description="The industry the company operates in"
    )
    work_type: Literal["remote", "hybrid", "on-site", "unknown"] = Field(
        "unknown",
        description="Whether the job is remote, hybrid, or on-site, or unknown if not specified"
    )
    job_title: Optional[str] = Field(
        None,
        description="The job title or position name"
    )

    candidate_minimal_requirements: Optional[str] = Field(
        None,
        description="The minimum requirements, qualifications, skills, experience, and education needed for the position"
    )
    must_haves: list[str] = Field(
        default_factory=list,
        description="The requirements the ad says a candidate must meet, one short item each, in the ad's own terms. Empty if the ad states none."
    )
    nice_to_haves: list[str] = Field(
        default_factory=list,
        description="The requirements the ad calls preferred, a plus or nice to have, one short item each. Empty if none."
    )
    language: Optional[str] = Field(
        None,
        description="The language the job description is written in, as an English name (e.g. 'English', 'German')"
    )
    additional_info: Optional[str] = Field(
        None,
        description="Any other relevant information about the company or position"
    )



def output_language_instruction(job_description_info: Optional[dict], document: str) -> str:
    """
    Tell a writer which language to write the document in.

    Without this a light model sometimes wrote the letter in the language of the job's country
    instead of the language of the ad and the user's CV (an English ad for a Zagreb job got a
    Croatian cover letter). The user's own request for another language still wins.
    """
    language = ((job_description_info or {}).get("language") or "").strip()
    target = f"{language}, the language of the job ad" if language else "the language the job ad is written in"
    return (
        f"Write the {document} in {target}, even if the employer or job is in another country. "
        f"Use another language only if the user's instructions (current or previous) ask for it."
    )


class JobDescriptionAgent:
    """Agent for extracting structured information from job descriptions."""
    
    def __init__(self, model: str = "openai:gpt-5-nano", temperature: float = 0.0):
        """
        Initialize the job description agent and supporting LLM.

        Args:
            model: The LLM model identifier to use
            temperature: Temperature setting for the LLM

        Returns:
            None
        """
        logger.info("Initializing JobDescriptionAgent...")
        
        self.llm = init_chat_model(model, temperature=temperature, **local_model_kwargs(model))
        logger.debug(f"JobDescriptionAgent LLM initialized - model: {model}, temperature: {temperature}")
    
    def extract_info(self, job_description: str) -> JobDescriptionInfo:
        """
        Extract structured information from a job description.
        
        Args:
            job_description: The full text of the job description
            
        Returns:
            JobDescriptionInfo: Structured information extracted from the job description
        """
        logger.info(f"Extracting job description info - input length: {len(job_description)} characters")
        logger.debug(f"Job description preview: {job_description[:200]}...")
        
        # Create prompt for extracting job description information
        prompt = ChatPromptTemplate.from_messages([
            ("system", JOB_DESCRIPTION_EXTRACTION_SYSTEM_PROMPT),
            ("human", JOB_DESCRIPTION_EXTRACTION_HUMAN_PROMPT)
        ])
        
        # Use structured output to ensure proper return type
        structured_llm = self.llm.with_structured_output(JobDescriptionInfo)
        chain = prompt | structured_llm
        
        llm_input = {"job_description": job_description}
        logger.info("Calling LLM to extract job description information...")
        logger.debug(f"LLM input - job_description length: {len(job_description)}")
        
        # Retried: a parse error or rate limit here would end the session right after the user pasted the ad.
        response = chain.with_retry(stop_after_attempt=3).invoke(llm_input)

        logger.info(f"LLM response received - company: {response.company_name}, job_title: {response.job_title}, work_type: {response.work_type}")
        logger.debug(f"LLM response - location: {response.location}, industry: {response.industry}, requirements_length: {len(response.candidate_minimal_requirements) if response.candidate_minimal_requirements else 0}")
        
        return response
    
    def run(self, state):
        """
        Main method to extract job description information from user input.
        
        This method uses LangGraph's interrupt() to pause execution and wait for the user
        to provide a job description. It checks if job_description_info already exists in 
        state to avoid reprocessing.
        
        Args:
            state: The state dictionary containing messages
            
        Returns:
            dict: Updated state with extracted job description information
        """
        logger.info("JobDescriptionAgent.run() called")
        
        pending_details = (state.get("job_ad_extra_details") or "").strip()
        if state.get("job_description_info") and not pending_details:
            logger.info("Job description already processed - skipping extraction")
            return {
                "messages": [{
                    "role": "assistant",
                    "content": "Job description has already been processed. Use the extracted information to proceed."
                }]
            }

        if pending_details:
            # The user answered the follow-up question for an ad with almost no details
            # (clarify_job_ad): extract again from the ad plus what they added.
            previous = (state.get("job_description_info") or {}).get("job_description") or ""
            job_description_text = f"{previous}\n\n{pending_details}".strip()
            logger.info(f"Re-extracting the job description with {len(pending_details)} characters added by the user")
        else:
            # Pause until the user gives a job ad with text (pasted, or a path to a readable file);
            # an empty reply or a wrong path is answered with a new question, not an empty ad.
            logger.info("Requesting job description from user via interrupt")
            job_description_text = ask_for_document(
                "Please paste the job description you'd like me to analyze. (filepath or text)", "job ad"
            )
            logger.info(f"Job description received - length: {len(job_description_text)} characters")

        # Extract information from job description
        extracted_info = self.extract_info(job_description_text)
        
        logger.info("Job description extraction completed successfully")
        
        # Format the response
        info_summary = f"""I've extracted the following information from the job description:

**Company:** {extracted_info.company_name or 'Not specified'}
**Website:** {extracted_info.company_website or 'Not specified'}
**Location:** {extracted_info.location or 'Not specified'}
**Industry:** {extracted_info.industry or 'Not specified'}
**Work Type:** {extracted_info.work_type.title()}
**Job Title:** {extracted_info.job_title or 'Not specified'}
**Candidate Minimal Requirements:** {extracted_info.candidate_minimal_requirements or 'Not specified'}
"""
        if extracted_info.must_haves:
            info_summary += "**Must-haves:** " + "; ".join(extracted_info.must_haves) + "\n"
        if extracted_info.nice_to_haves:
            info_summary += "**Nice-to-haves:** " + "; ".join(extracted_info.nice_to_haves) + "\n"
        if extracted_info.additional_info:
            info_summary += f"\n**Additional Information:**\n{extracted_info.additional_info}"
        
        # Write debug info to file
        debug_content = ""
        debug_content += "FULL JOB DESCRIPTION:\n"
        debug_content += "-" * 80 + "\n"
        debug_content += job_description_text
        debug_content += "\n\n"
        
        debug_content += "EXTRACTED SUMMARY:\n"
        debug_content += "-" * 80 + "\n"
        debug_content += info_summary
        debug_content += "\n\n"
        
        write_to_debug(debug_content, "JOB DESCRIPTION DEBUG INFO")
        logger.info("Debug info written to debug file")
        
        # Store extracted info in state for use by other nodes
        updated_state = {
            "messages": state.get("messages", []) + [{"role": "assistant", "content": info_summary}],
            # The ad's own text is added here, not echoed by the model (see JobDescriptionInfo).
            "job_description_info": {**extracted_info.model_dump(), "job_description": job_description_text},
            "job_ad_extra_details": None,
            "current_node": "job_description"
        }
        
        return updated_state



# Answers a light model gives for a field it could not fill.
_EMPTY_FIELD_ANSWERS = {"", "none", "n/a", "na", "null", "unknown", "not specified", "not provided", "not stated", "unspecified"}


def field_value(value) -> str:
    """An extracted text field, or "" when the model answered with a placeholder such as 'unknown'."""
    text = (value or "").strip() if isinstance(value, str) else ""
    return "" if text.strip(" .").lower() in _EMPTY_FIELD_ANSWERS else text



def job_ad_lacks_details(job_description_info: dict | None) -> bool:
    """True when extraction found neither a job title nor any requirements (e.g. "We're hiring!")."""
    info = job_description_info or {}
    return not field_value(info.get("job_title")) and not field_value(info.get("candidate_minimal_requirements"))


def route_after_job_description(state) -> str:
    """Ask once for more details when the ad has almost none; otherwise go on to company research."""
    if job_ad_lacks_details(state.get("job_description_info")) and not state.get("job_ad_clarification_asked"):
        return "clarify_job_ad"
    return "research_company_context"


def clarify_job_ad(state):
    """
    Ask the user, once, for more details about a job ad that has no job title and no requirements.

    Whatever they reply is added to the ad and extracted again (analyze_job_description); there is
    no keyword to type, so this works in any language. A reply with no details (e.g. "continue")
    just leaves the extraction as thin as before, and the session goes on: this is asked only once.
    """
    reply = str(interrupt({
        "message": (
            "This job ad has very little detail: I could not find a job title or any requirements, "
            "so the CV and cover letter can only be tailored loosely. Paste the full ad or add the "
            "job title and main requirements, or reply 'continue' to go on with the ad as it is."
        ),
        "required": True,
    }) or "").strip()
    update = {"job_ad_clarification_asked": True, "current_node": "clarify_job_ad"}
    if reply:
        # Stored with the ad (job_description_info), not as a chat message: the router would
        # otherwise read a reply such as "continue" as a request about the documents.
        update["job_ad_extra_details"] = reply
    return update


def route_after_clarify_job_ad(state) -> str:
    return "analyze_job_description" if (state.get("job_ad_extra_details") or "").strip() else "research_company_context"
