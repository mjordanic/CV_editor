from typing import Optional
import logging
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
import os

logger = logging.getLogger(__name__)


COVER_LETTER_GENERATION_SYSTEM_PROMPT = (
    "You are an expert cover letter writer specializing in creating compelling, personalized cover letters "
    "that match job descriptions and company cultures. Your cover letters are engaging, professional, and "
    "effectively communicate why the candidate is a perfect fit for the role."
)

COVER_LETTER_GENERATION_HUMAN_PROMPT = (
    "Create a professional, tailored cover letter based on the following information:\n\n"
    "**Candidate Information:**\n{candidate_cv}\n{candidate_cover_letter}\n\n"
    "**Job Description:**\n{job_description}\n\n"
    "**Company Information:**\n{company_info}\n\n"
    "{modification_instructions}\n\n"
    "Instructions:\n"
    "- Tailor the cover letter to the specific job and company\n"
    "- Address the hiring manager or use a professional greeting\n"
    "- Start with a strong opening that captures attention\n"
    "- Highlight relevant skills and experiences that match the job requirements\n"
    "- Show enthusiasm for the role and company\n"
    "- Reference specific details from the job description and company information\n"
    "- Explain why you're a good fit for the position\n"
    "- Keep it concise (ideally 3-4 paragraphs, one page)\n"
    "- Use professional, engaging language\n"
    "- End with a strong closing and call to action\n"
    "- If a previous cover letter is provided, use it as a base and enhance it\n"
    "- If no cover letter is provided, create a new one from scratch"
)


class CoverLetterWriterAgent:
    """Agent for generating tailored cover letters based on job descriptions and company information."""
    
    def __init__(self, output_folder: str = "CV"):
        """
        Initialize the CoverLetterWriterAgent.
        
        Args:
            output_folder: Folder where generated cover letters will be saved
        """
        self.llm = init_chat_model("openai:gpt-5-nano", temperature=0.7)
        self.output_folder = output_folder
        # Ensure output folder exists
        os.makedirs(self.output_folder, exist_ok=True)
    
    def _format_company_info(self, company_info: Optional[dict]) -> str:
        """Format company information for the prompt."""
        if not company_info:
            return "No additional company information available."
        
        info_parts = []
        if company_info.get("company_description"):
            info_parts.append(f"Description: {company_info['company_description']}")
        if company_info.get("remote_work"):
            info_parts.append(f"Remote Work Policy: {company_info['remote_work']}")
        
        return "\n".join(info_parts) if info_parts else "No additional company information available."
    
    def _format_job_description(self, job_description_info: Optional[dict]) -> str:
        """Format job description information for the prompt."""
        if not job_description_info:
            return "No job description information available."
        
        parts = []
        if job_description_info.get("job_title"):
            parts.append(f"Job Title: {job_description_info['job_title']}")
        if job_description_info.get("company_name"):
            parts.append(f"Company: {job_description_info['company_name']}")
        if job_description_info.get("location"):
            parts.append(f"Location: {job_description_info['location']}")
        if job_description_info.get("candidate_minimal_requirements"):
            parts.append(f"\nRequirements:\n{job_description_info['candidate_minimal_requirements']}")
        if job_description_info.get("job_description"):
            parts.append(f"\nFull Job Description:\n{job_description_info['job_description']}")
        
        return "\n".join(parts)
    
    def _get_modification_instructions(self, user_feedback: Optional[str], has_existing_cover_letter: bool) -> str:
        """Get modification instructions based on user feedback."""
        if not user_feedback or not user_feedback.strip():
            if has_existing_cover_letter:
                return "Please review the existing cover letter and make improvements based on the job description and company information."
            return ""
        
        return f"**User Feedback/Modification Request:**\n{user_feedback}\n\nPlease incorporate these changes into the cover letter."
    
    def generate_cover_letter(
        self,
        candidate_cv: Optional[str],
        candidate_cover_letter: Optional[str],
        job_description_info: Optional[dict],
        company_info: Optional[dict],
        user_feedback: Optional[str] = None
    ) -> str:
        """
        Generate a tailored cover letter.
        
        Args:
            candidate_cv: Existing CV text (if available)
            candidate_cover_letter: Existing cover letter text (if available)
            job_description_info: Extracted job description information
            company_info: Company information from search
            user_feedback: Optional feedback for modifications
            
        Returns:
            str: Generated cover letter text
        """
        # Format inputs
        candidate_cv_text = candidate_cv or "No CV provided."
        candidate_cover_letter_text = candidate_cover_letter or "No existing cover letter provided. Create a new cover letter."
        job_desc_text = self._format_job_description(job_description_info)
        company_info_text = self._format_company_info(company_info)
        modification_instructions = self._get_modification_instructions(user_feedback, bool(candidate_cover_letter))
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", COVER_LETTER_GENERATION_SYSTEM_PROMPT),
            ("human", COVER_LETTER_GENERATION_HUMAN_PROMPT)
        ])
        
        # Generate cover letter
        chain = prompt | self.llm
        response = chain.invoke({
            "candidate_cv": candidate_cv_text,
            "candidate_cover_letter": candidate_cover_letter_text,
            "job_description": job_desc_text,
            "company_info": company_info_text,
            "modification_instructions": modification_instructions
        })
        
        return response.content if hasattr(response, 'content') else str(response)
    
    def save_cover_letter(self, cover_letter_text: str, filename: str = "generated_cover_letter.txt") -> str:
        """
        Save the generated cover letter to a file.
        
        Args:
            cover_letter_text: The cover letter text to save
            filename: Name of the file to save
            
        Returns:
            str: Path to the saved file
        """
        file_path = os.path.join(self.output_folder, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(cover_letter_text)
        return file_path
    
    def run(self, state):
        """
        Main method to generate and save a cover letter.
        
        Args:
            state: The state dictionary containing candidate_text, job_description_info, company_info, and user_feedback
            
        Returns:
            dict: Updated state with generated cover letter
        """
        candidate_text = state.get("candidate_text", {})
        candidate_cv = candidate_text.get("cv") if candidate_text else None
        # Use generated cover letter if it exists (for modifications), otherwise use original
        generated_cover_letter = state.get("generated_cover_letter")
        candidate_cover_letter = generated_cover_letter if generated_cover_letter else (candidate_text.get("cover_letter") if candidate_text else None)
        job_description_info = state.get("job_description_info")
        company_info = state.get("company_info")
        user_feedback = state.get("user_feedback")
        
        # Generate cover letter
        cover_letter_text = self.generate_cover_letter(
            candidate_cv=candidate_cv,
            candidate_cover_letter=candidate_cover_letter,
            job_description_info=job_description_info,
            company_info=company_info,
            user_feedback=user_feedback
        )
        
        # Save cover letter
        file_path = self.save_cover_letter(cover_letter_text)
        
        return {
            "generated_cover_letter": cover_letter_text,
            "messages": [{
                "role": "assistant",
                "content": f"Cover letter has been generated and saved to {file_path}. Please review it and let me know if you'd like any modifications."
            }]
        }

