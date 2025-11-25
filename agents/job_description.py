from typing import Optional, Literal
import logging
import os
import json
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.types import interrupt
from debug_utils import write_to_debug

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
    "- Any other relevant company or position details\n\n"
    "Job Description:\n\n{job_description}"
)


class JobDescriptionInfo(BaseModel):
    """Structured information extracted from a job description."""
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
    job_description: Optional[str] = Field(
        None,
        description="The full original job description text"
    )
    additional_info: Optional[str] = Field(
        None,
        description="Any other relevant information about the company or position"
    )



class JobDescriptionAgent:
    """Agent for extracting structured information from job descriptions."""
    
    def __init__(self, model: str = "openai:gpt-5-nano", temperature: float = 0):
        """
        Initialize the job description agent and supporting LLM.

        Args:
            model: The LLM model identifier to use
            temperature: Temperature setting for the LLM

        Returns:
            None
        """
        logger.info("Initializing JobDescriptionAgent...")
        self.llm = init_chat_model(model, temperature=temperature)
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
        
        response = chain.invoke(llm_input)
        
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
        
        # Check if we've already processed a job description
        if state.get("job_description_info"):
            logger.info("Job description already processed - skipping extraction")
            # Already have job description info, return it without reprocessing
            return {
                "messages": [{
                    "role": "assistant",
                    "content": "Job description has already been processed. Use the extracted information to proceed."
                }]
            }
        
        # Pause execution and wait for user to provide job description
        logger.info("Requesting job description from user via interrupt")
        job_description_text = interrupt({"message": "Please paste the job description you'd like me to analyze.", "required": True})
        
        logger.info(f"Job description received via interrupt - length: {len(str(job_description_text))} characters")
        logger.debug(f"Job description preview: {str(job_description_text)[:200]}...")
        
        # Ensure we have a string
        job_description_text = str(job_description_text).strip()
        
        # Extract information from job description
        extracted_info = self.extract_info(job_description_text)
        
        # Store the full job description text in the extracted info
        extracted_info.job_description = job_description_text
        
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
            "job_description_info": extracted_info.model_dump(),
            "current_node": "job_description"
        }
        
        return updated_state

