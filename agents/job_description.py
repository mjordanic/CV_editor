from typing import Optional, Literal
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.types import interrupt


JOB_DESCRIPTION_EXTRACTION_SYSTEM_PROMPT = (
    "You are a helpful assistant that extracts structured information from job descriptions. "
    "Analyze the job description carefully and extract all relevant details about the company and position."
)

JOB_DESCRIPTION_EXTRACTION_HUMAN_PROMPT = (
    "Extract the following information from the job description:\n"
    "- Company name\n"
    "- Company website/web page\n"
    "- Location (city, state, country)\n"
    "- Industry\n"
    "- Whether the job is remote, hybrid, or on-site\n"
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
    
    def __init__(self):
        self.llm = init_chat_model("openai:gpt-5-nano", temperature=0)
    
    def extract_info(self, job_description: str) -> JobDescriptionInfo:
        """
        Extract structured information from a job description.
        
        Args:
            job_description: The full text of the job description
            
        Returns:
            JobDescriptionInfo: Structured information extracted from the job description
        """
        # Create prompt for extracting job description information
        prompt = ChatPromptTemplate.from_messages([
            ("system", JOB_DESCRIPTION_EXTRACTION_SYSTEM_PROMPT),
            ("human", JOB_DESCRIPTION_EXTRACTION_HUMAN_PROMPT)
        ])
        
        # Use structured output to ensure proper return type
        structured_llm = self.llm.with_structured_output(JobDescriptionInfo)
        chain = prompt | structured_llm
        response = chain.invoke({"job_description": job_description})
        
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
        # Check if we've already processed a job description
        if state.get("job_description_info"):
            # Already have job description info, return it without reprocessing
            return {
                "messages": [{
                    "role": "assistant",
                    "content": "Job description has already been processed. Use the extracted information to proceed."
                }]
            }
        
        # Pause execution and wait for user to provide job description
        # The interrupt() call will pause the graph and return the user's input when resumed
        job_description_text = interrupt({
            "message": "Please paste the job description you'd like me to analyze.",
            "required": True
        })
        
        # Validate that we received substantial content
        if not job_description_text or len(str(job_description_text).strip()) < 50:
            # If content is too short, interrupt again asking for more
            job_description_text = interrupt({
                "message": "Please paste the complete job description. The description should be substantial (at least a few sentences).",
                "required": True
            })
        
        # Ensure we have a string
        job_description_text = str(job_description_text).strip()
        
        # Extract information from job description
        extracted_info = self.extract_info(job_description_text)
        
        # Store the full job description text in the extracted info
        extracted_info.job_description = job_description_text
        
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
        
        # Store extracted info in state for use by other nodes
        updated_state = {
            "messages": [{"role": "assistant", "content": info_summary}],
            "job_description_info": extracted_info.model_dump()
        }
        
        return updated_state

