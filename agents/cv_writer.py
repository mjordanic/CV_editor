from typing import Optional
import logging
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
import os
from debug_utils import write_to_debug

logger = logging.getLogger(__name__)


CV_GENERATION_SYSTEM_PROMPT = (
    "You are an expert CV writer specializing in creating tailored, professional CVs that match job descriptions. "
    "Your CVs are well-structured, highlight relevant skills and experiences, and are optimized for ATS (Applicant Tracking Systems). "
    "You create compelling CVs that stand out while remaining truthful and accurate."
)

CV_GENERATION_HUMAN_PROMPT = (
    "Create a professional, tailored CV based on the following information:\n\n"
    "**Candidate Information:**\n{candidate_cv}\n\n"
    "**Job Description:**\n{job_description}\n\n"
    "**Company Information:**\n{company_info}\n\n"
    "Make sure to follow these modification instructions: {modification_instructions}\n\n"
    "User requested the following modifications in the previous messages, make sure not to overwrite them or contradict them unless they are explicitly stated in the modification instructions: {previous_modification_instructions}\n\n"
    "General guidelines:\n"
    "- Tailor the CV to match the job requirements and company culture\n"
    "- Highlight relevant skills, experiences, and achievements\n"
    "- Use clear, professional language\n"
    "- Structure the CV in a standard format (Contact Info, Professional Summary, Experience, Education, Skills)\n"
    "- Ensure the CV is ATS-friendly\n"
    "- Keep it concise (ideally 1-2 pages)\n"
    "- Make sure all information is accurate and truthful\n"
    "- If the candidate's CV is provided, use it as a base and enhance it for this specific role\n"
    "- If no candidate CV is provided, create a professional CV based on the job requirements"
)


class CVWriterAgent:
    """Agent for generating tailored CVs based on job descriptions and company information."""
    
    def __init__(self, output_folder: str = "generated_CVs"):
        """
        Initialize the CVWriterAgent.
        
        Args:
            output_folder: Folder where generated CVs will be saved

        Returns:
            None
        """
        logger.info(f"Initializing CVWriterAgent with output_folder: {output_folder}")
        self.llm = init_chat_model("openai:gpt-5-nano", temperature=0.7)
        logger.debug("CVWriterAgent LLM initialized")
        self.output_folder = output_folder
        # Ensure output folder exists
        os.makedirs(self.output_folder, exist_ok=True)
        logger.debug(f"Output folder ensured: {output_folder}")

        self.previous_modification_instructions = []
    
    def _format_company_info(self, company_info: Optional[dict]) -> str:
        """
        Format structured company details into a readable string for prompting.

        Args:
            company_info: Dictionary containing company description and remote policy keys.

        Returns:
            str: Human-readable description of the organization or default fallback text.
        """
        if not company_info:
            return "No additional company information available."
        
        info_parts = []
        if company_info.get("company_description"):
            info_parts.append(f"Description: {company_info['company_description']}")
        if company_info.get("remote_work"):
            info_parts.append(f"Remote Work Policy: {company_info['remote_work']}")
        
        return "\n".join(info_parts) if info_parts else "No additional company information available."
    
    def _format_job_description(self, job_description_info: Optional[dict]) -> str:
        """
        Compile the extracted job description fields into a prompt-ready string.

        Args:
            job_description_info: Dictionary produced by `JobDescriptionAgent`.

        Returns:
            str: Concise job overview highlighting title, company, requirements, and description.
        """
        if not job_description_info:
            return "No job description information available."
        
        parts = []
        if job_description_info.get("job_title"):
            parts.append(f"Job Title: {job_description_info['job_title']}")
        if job_description_info.get("company_name"):
            parts.append(f"Company: {job_description_info['company_name']}")
        if job_description_info.get("candidate_minimal_requirements"):
            parts.append(f"\nRequirements:\n{job_description_info['candidate_minimal_requirements']}")
        if job_description_info.get("job_description"):
            parts.append(f"\nFull Job Description:\n{job_description_info['job_description']}")
        
        return "\n".join(parts)
    
    def _get_modification_instructions(self, user_feedback: Optional[str], has_existing_cv: bool) -> str:
        """
        Build instructions that describe how the LLM should modify an existing CV.

        Args:
            user_feedback: Raw feedback text provided by the user, if any.
            has_existing_cv: Flag indicating whether a prior CV exists to be edited.

        Returns:
            str: Instructional text injected into the prompt or an empty string if none needed.
        """
        # Always prioritize user feedback when provided
        if user_feedback and user_feedback.strip():
            if has_existing_cv:
                return f"**User Feedback/Modification Request:**\n{user_feedback}\n\nPlease review the existing CV and incorporate these changes into it."
            return f"**User Feedback/Modification Request:**\n{user_feedback}\n\nPlease incorporate these changes into the CV."
        
        # Only use hardcoded message when there's no user feedback
        if has_existing_cv:
            return "Please review the existing CV and make improvements based on the job description and company information."
        
        return ""
    
    def _format_previous_modification_instructions(self) -> str:
        """
        Format the list of previous modification instructions into a readable string.

        Returns:
            str: Formatted string of previous instructions, or empty string if none exist.
        """
        if not self.previous_modification_instructions:
            return "No previous modification instructions."
        
        formatted = []
        for idx, instruction in enumerate(self.previous_modification_instructions, 1):
            if instruction and instruction.strip():
                formatted.append(f"{idx}. {instruction.strip()}")
        
        if not formatted:
            return "No previous modification instructions."
        
        return "\n".join(formatted)
    
    def generate_cv(
        self,
        candidate_cv: Optional[str],
        job_description_info: Optional[dict],
        company_info: Optional[dict],
        modification_instructions: Optional[str] = None,
        previous_modification_instructions_formatted: Optional[str] = None
    ) -> str:
        """
        Generate a tailored CV.
        
        Args:
            candidate_cv: Existing CV text (if available)
            job_description_info: Extracted job description information
            company_info: Company information from search
            user_feedback: Optional feedback for modifications
            
        Returns:
            str: Generated CV text
        """
        # Format inputs
        candidate_cv_text = candidate_cv or "No existing CV provided. Create a professional CV based on the job requirements."
        job_desc_text = self._format_job_description(job_description_info)
        company_info_text = self._format_company_info(company_info)
        
        logger.info(f"Generating CV - candidate_cv_length: {len(candidate_cv_text) if candidate_cv_text else 0}, has_modification_instructions: {bool(modification_instructions)}")
        logger.debug(f"Job description length: {len(job_desc_text)}, company info length: {len(company_info_text)}")
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", CV_GENERATION_SYSTEM_PROMPT),
            ("human", CV_GENERATION_HUMAN_PROMPT)
        ])
        
        # Generate CV
        chain = prompt | self.llm
        llm_input = {
            "candidate_cv": candidate_cv_text,
            "job_description": job_desc_text,
            "company_info": company_info_text,
            "modification_instructions": modification_instructions,
            "previous_modification_instructions": previous_modification_instructions_formatted
        }
        logger.info("Calling LLM to generate CV...")
        logger.debug(f"LLM input - candidate_cv length: {len(candidate_cv_text)}, job_description length: {len(job_desc_text)}, company_info length: {len(company_info_text)}, modification_instructions length: {len(modification_instructions)}")
        
        response = chain.invoke(llm_input)
        
        cv_content = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"LLM response received - CV generated, length: {len(cv_content)} characters")
        logger.debug(f"CV preview: {cv_content[:200]}...")
        
        return cv_content
    
    def save_cv(self, cv_text: str, filename: str = "generated_CV.txt") -> str:
        """
        Save the generated CV to a file.
        
        Args:
            cv_text: The CV text to save
            filename: Name of the file to save
            
        Returns:
            str: Path to the saved file
        """
        file_path = os.path.join(self.output_folder, filename)
        logger.info(f"Saving CV to file: {file_path}")
        logger.debug(f"CV text length: {len(cv_text)} characters")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(cv_text)
        logger.info(f"CV saved successfully to {file_path}")
        return file_path
    
    def run(self, state):
        """
        Main method to generate and save a CV.
        
        Args:
            state: The state dictionary containing candidate_text, job_description_info, company_info, and user_feedback
            
        Returns:
            dict: Updated state with generated CV
        """
        logger.info("CVWriterAgent.run() called")
        
        candidate_text = state.get("candidate_text", {})
        # Use generated CV if it exists (for modifications), otherwise use original
        generated_cv = state.get("generated_cv")
        candidate_cv = generated_cv if generated_cv else (candidate_text.get("cv") if candidate_text else None)
        job_description_info = state.get("job_description_info")
        company_info = state.get("company_info")
        user_feedback = state.get("user_feedback")
        
        logger.debug(f"State extracted - has_candidate_cv: {candidate_cv is not None}, has_job_desc: {job_description_info is not None}, has_company_info: {company_info is not None}, has_feedback: {user_feedback is not None}")
        
        modification_instructions = self._get_modification_instructions(user_feedback, bool(candidate_cv))
        previous_modification_instructions_formatted = self._format_previous_modification_instructions()
        
        # Generate CV
        cv_text = self.generate_cv(
            candidate_cv=candidate_cv,
            job_description_info=job_description_info,
            company_info=company_info,
            modification_instructions=modification_instructions,
            previous_modification_instructions_formatted=previous_modification_instructions_formatted
        )

        # Store feedback for future iterations (only if non-empty)
        if user_feedback and user_feedback.strip():
            self.previous_modification_instructions.append(user_feedback.strip())
        
        # Save CV
        file_path = self.save_cv(cv_text)
        
        # Write to debug file
        debug_content = ""
        debug_content += f"GENERATED CV (saved to: {file_path}):\n"
        debug_content += "-" * 80 + "\n"
        debug_content += f"MODIFICATION INSTRUCTIONS:\n{modification_instructions}\n\n"
        debug_content += f"PREVIOUS MODIFICATION INSTRUCTIONS:\n{previous_modification_instructions_formatted}\n\n"
        debug_content += "-" * 80 + "\n"
        debug_content += cv_text
        debug_content += "\n\n"
        
        write_to_debug(debug_content, "CV WRITER DEBUG INFO")
        logger.info("CV generation and saving completed successfully")
        
        return {
            "generated_cv": cv_text,
            "messages": state.get("messages", []) + [{
                "role": "assistant",
                "content": f"CV has been generated and saved to {file_path}. Please review it and let me know if you'd like any modifications."
            }],
            "current_node": "cv_writer"
        }

