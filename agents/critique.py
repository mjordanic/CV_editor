from typing import Optional, Literal
import logging
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from debug_utils import write_to_debug

logger = logging.getLogger(__name__)


CRITIQUE_SYSTEM_PROMPT = (
    "You are an expert CV and cover letter reviewer specializing in quality assessment, "
    "ATS (Applicant Tracking System) optimization, and job alignment analysis. "
    "Your PRIMARY GOAL is to ensure the document passes AI-based screening and is strictly aligned with the job advertisement. "
    "Your role is to evaluate documents objectively and provide clear, actionable improvement instructions. "
    "You assess content quality, ATS compatibility, alignment with job requirements, and overall professionalism. "
    "Check specifically for keyword matching and semantic alignment with the job description. "
    "CRITICAL: When evaluating a CV, provide improvement instructions ONLY for the CV. Do NOT suggest moving content to a cover letter or any other document "
    "or generating a cover letter. Focus solely on improving the CV document itself. "
    "Likewise, when evaluating a cover letter, provide improvement instructions ONLY for the cover letter. Do NOT suggest moving content to a CV or any other document "
    "or generating a CV. Focus solely on improving the cover letter document itself. "
    "IMPORTANT: Be conservative in your assessment. Only suggest improvements for genuine issues. "
    "If a document is already good (quality score 85+), only suggest critical issues or very minor enhancements. "
    "Do NOT be overly perfectionist - focus on meaningful improvements that significantly impact quality, ATS compatibility, or job alignment."
    "\n\nTRUTHFULNESS VERIFICATION:\n"
    "You MUST cross-reference the generated document with the 'Original Candidate Information'.\n"
    "If you detect any potential hallucinations (skills or experiences in the generated doc that are NOT in the original info), "
    "you MUST flag them as 'Critical Issues'.\n"
    "Example: If generated CV says 'Expert in Kubernetes' but original CV only mentions 'Docker', flag this as a potential hallucination."
)

CRITIQUE_HUMAN_PROMPT = (
    "Evaluate the following document and provide improvement instructions:\n\n"
    "**Document Type:** {document_type}\n\n"
    "**Document Content:**\n{document_content}\n\n"
    "**Job Description:**\n{job_description}\n\n"
    "**Company Information:**\n{company_info}\n\n"
    "**Original Candidate Information:**\n{candidate_info}\n\n"
    "Please evaluate the document on:\n"
    "1. **Content Quality**: Structure, clarity, professionalism, impact, and completeness\n"
    "2. **ATS Compatibility**: Formatting, keywords, structure, parsing-friendliness, and **AI Screening Probability**\n"
    "3. **Job Alignment**: How well the document matches job requirements, company culture, and role expectations (**Keyword Matching**)\n"
    "4. **Overall Assessment**: Overall quality score and critical issues\n\n"
    "CRITICAL REQUIREMENTS FOR IMPROVEMENT INSTRUCTIONS:\n"
    "- Be VERY specific and actionable. Each instruction should clearly state WHAT to change, WHERE to change it, and HOW to change it.\n"
    "- Use exact locations (e.g., 'In the Professional Summary section, change...' or 'In the Experience section under [Company Name], add...')\n"
    "- Provide concrete examples when possible (e.g., 'Change \"managed team\" to \"led a cross-functional team of 5 members\"')\n"
    "- Distinguish between CRITICAL issues (must fix) and MINOR improvements (nice to have)\n"
    "- If quality score is 85 or above, only suggest critical issues or very minor enhancements\n"
    "- If the document is already excellent (90+) or there are no meaningful improvements, return an EMPTY STRING (\"\") for improvement_instructions\n"
    "- Do NOT suggest vague improvements like 'improve clarity' - be specific about what to change\n"
    "- Focus on improvements that significantly impact quality, ATS compatibility, or job alignment\n"
    "- CRITICAL: Return empty string \"\" if no improvements are needed - do NOT write 'No improvements needed' or similar text"
)


class CritiqueResponse(BaseModel):
    """Response model for critique evaluation."""
    quality_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Overall quality score from 0-100, where 100 is perfect"
    )
    content_quality_feedback: str = Field(
        ...,
        description="Feedback on content quality, structure, clarity, and professionalism"
    )
    ats_compatibility_feedback: str = Field(
        ...,
        description="Feedback on ATS compatibility, formatting, keywords, and structure"
    )
    job_alignment_feedback: str = Field(
        ...,
        description="Feedback on how well the document aligns with job requirements and company culture"
    )
    improvement_instructions: str = Field(
        default="",
        description="Clear, actionable instructions for improving the document. These should be VERY specific and implementable. "
        "Each instruction must clearly state WHAT to change, WHERE to change it, and HOW to change it. "
        "Use exact locations (e.g., 'In the Professional Summary section, change...'). "
        "CRITICAL: Return an EMPTY STRING (\"\") if no improvements are needed. Do NOT write 'No improvements needed' or any other text - just return empty string."
    )
    critical_issues: str = Field(
        default="",
        description="Any critical issues that must be addressed (empty if none)"
    )


class CritiqueAgent:
    """Agent for evaluating CV and cover letter quality and providing improvement instructions."""
    
    def __init__(self, model: str = "openai:gpt-5-mini", temperature: float = 0.3, quality_threshold: int = 85):
        """
        Initialize the CritiqueAgent.

        Args:
            model: The LLM model identifier to use
            temperature: Temperature setting for the LLM
            quality_threshold: Minimum quality score below which refinement is recommended (default: 85)

        Returns:
            None
        """
        logger.info("Initializing CritiqueAgent...")
        
        self.llm = init_chat_model(model, temperature=temperature)
        self.quality_threshold = quality_threshold
        logger.debug(f"CritiqueAgent LLM initialized - model: {model}, temperature: {temperature}, quality_threshold: {quality_threshold}")
    
    def _format_job_description(self, job_description_info: Optional[dict]) -> str:
        """
        Format job description information for critique evaluation.

        Args:
            job_description_info: Dictionary produced by `JobDescriptionAgent`.

        Returns:
            str: Formatted job description information.
        """
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
        
        return "\n".join(parts) if parts else "No job description information available."
    
    def _format_company_info(self, company_info: Optional[dict]) -> str:
        """
        Format company information for critique evaluation.

        Args:
            company_info: Dictionary containing company description and remote policy.

        Returns:
            str: Formatted company information.
        """
        if not company_info:
            return "No additional company information available."
        
        info_parts = []
        if company_info.get("company_description"):
            info_parts.append(f"Description: {company_info['company_description']}")
        if company_info.get("remote_work"):
            info_parts.append(f"Remote Work Policy: {company_info['remote_work']}")
        
        return "\n".join(info_parts) if info_parts else "No additional company information available."
    
    def _format_candidate_info(self, candidate_text: Optional[dict]) -> str:
        """
        Format original candidate information for context.

        Args:
            candidate_text: Dictionary containing original CV and cover letter.

        Returns:
            str: Formatted candidate information.
        """
        if not candidate_text:
            return "No original candidate information available."
        
        parts = []
        if candidate_text.get("cv"):
            cv_text = candidate_text["cv"]
            if cv_text:
                parts.append(f"Original CV: {cv_text[:500]}...")  # Truncate for context
            else:
                parts.append("Original CV: Not provided")
        else:
            parts.append("Original CV: Not provided")
        
        if candidate_text.get("cover_letter"):
            cl_text = candidate_text["cover_letter"]
            if cl_text:
                parts.append(f"Original Cover Letter: {cl_text[:500]}...")  # Truncate for context
            else:
                parts.append("Original Cover Letter: Not provided")
        else:
            parts.append("Original Cover Letter: Not provided")
        
        return "\n".join(parts) if parts else "No original candidate information available."
    
    def critique_cv(
        self,
        cv_content: str,
        job_description_info: Optional[dict],
        company_info: Optional[dict],
        candidate_text: Optional[dict]
    ) -> CritiqueResponse:
        """
        Evaluate a CV and provide improvement instructions.

        Args:
            cv_content: The CV content to evaluate
            job_description_info: Extracted job description information
            company_info: Company information from search
            candidate_text: Original candidate documents for context

        Returns:
            CritiqueResponse: Evaluation results with improvement instructions
        """
        logger.info("Critiquing CV...")
        logger.debug(f"CV length: {len(cv_content)} characters")
        
        job_desc_text = self._format_job_description(job_description_info)
        company_info_text = self._format_company_info(company_info)
        candidate_info_text = self._format_candidate_info(candidate_text)
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", CRITIQUE_SYSTEM_PROMPT),
            ("human", CRITIQUE_HUMAN_PROMPT)
        ])
        
        # Use structured output
        structured_llm = self.llm.with_structured_output(CritiqueResponse)
        chain = prompt | structured_llm
        
        # Prepare LLM input
        llm_input = {
            "document_type": "CV",
            "document_content": cv_content,
            "job_description": job_desc_text,
            "company_info": company_info_text,
            "candidate_info": candidate_info_text
        }
        
        logger.info("Calling LLM for CV critique...")
        logger.debug(f"LLM input - CV length: {len(cv_content)}, job_desc length: {len(job_desc_text)}")
        
        response = chain.invoke(llm_input)
        
        logger.info(f"CV critique completed - quality_score: {response.quality_score}")
        logger.debug(f"Improvement instructions length: {len(response.improvement_instructions)}")
        
        return response
    
    def critique_cover_letter(
        self,
        cover_letter_content: str,
        cv_content: Optional[str],
        job_description_info: Optional[dict],
        company_info: Optional[dict],
        candidate_text: Optional[dict]
    ) -> CritiqueResponse:
        """
        Evaluate a cover letter and provide improvement instructions.

        Args:
            cover_letter_content: The cover letter content to evaluate
            cv_content: The generated CV (for context and consistency)
            job_description_info: Extracted job description information
            company_info: Company information from search
            candidate_text: Original candidate documents for context

        Returns:
            CritiqueResponse: Evaluation results with improvement instructions
        """
        logger.info("Critiquing cover letter...")
        logger.debug(f"Cover letter length: {len(cover_letter_content)} characters")
        
        job_desc_text = self._format_job_description(job_description_info)
        company_info_text = self._format_company_info(company_info)
        candidate_info_text = self._format_candidate_info(candidate_text)
        
        # Include CV content in document content for context
        document_content = f"**Associated CV:**\n{cv_content}\n\n**Cover Letter:**\n{cover_letter_content}" if cv_content else cover_letter_content
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", CRITIQUE_SYSTEM_PROMPT),
            ("human", CRITIQUE_HUMAN_PROMPT)
        ])
        
        # Use structured output
        structured_llm = self.llm.with_structured_output(CritiqueResponse)
        chain = prompt | structured_llm
        
        # Prepare LLM input
        llm_input = {
            "document_type": "Cover Letter",
            "document_content": document_content,
            "job_description": job_desc_text,
            "company_info": company_info_text,
            "candidate_info": candidate_info_text
        }
        
        logger.info("Calling LLM for cover letter critique...")
        logger.debug(f"LLM input - cover letter length: {len(cover_letter_content)}, job_desc length: {len(job_desc_text)}")
        
        response = chain.invoke(llm_input)
        
        logger.info(f"Cover letter critique completed - quality_score: {response.quality_score}")
        logger.debug(f"Improvement instructions length: {len(response.improvement_instructions)}")
        
        return response
    
    def run_cv(self, state):
        """
        Critique CV document.

        Args:
            state: The state dictionary containing generated CV and context

        Returns:
            dict: Updated state with CV critique feedback and improvement instructions
        """
        logger.info("CritiqueAgent.run_cv() called")
        
        generated_cv = state.get("generated_cv")
        if not generated_cv:
            logger.warning("No CV to critique - skipping")
            return {
                "messages": state.get("messages", []),
                "current_node": "critique_cv",
                "cv_needs_refinement": False
            }
        
        job_description_info = state.get("job_description_info")
        company_info = state.get("company_info")
        candidate_text = state.get("candidate_text")
        
        logger.info("Critiquing generated CV...")
        cv_critique_result = self.critique_cv(
            cv_content=generated_cv,
            job_description_info=job_description_info,
            company_info=company_info,
            candidate_text=candidate_text
        )
        
        state_updates = {
            "messages": state.get("messages", []),
            "current_node": "critique_cv",
            "cv_critique": {
                "quality_score": cv_critique_result.quality_score,
                "content_quality_feedback": cv_critique_result.content_quality_feedback,
                "ats_compatibility_feedback": cv_critique_result.ats_compatibility_feedback,
                "job_alignment_feedback": cv_critique_result.job_alignment_feedback,
                "improvement_instructions": cv_critique_result.improvement_instructions,
                "critical_issues": cv_critique_result.critical_issues
            }
        }
        
        # Update CV history
        cv_history = state.get("cv_history", [])
        # Determine iteration number (if not already set in state, infer from history length)
        iteration = len(cv_history) + 1
        
        history_entry = {
            "iteration": iteration,
            "content": generated_cv,
            "score": cv_critique_result.quality_score,
            "critique": cv_critique_result.dict()
        }
        
        # Append to history
        state_updates["cv_history"] = cv_history + [history_entry]
        logger.info(f"Added CV iteration {iteration} to history with score {cv_critique_result.quality_score}")
        
        # Write to debug file
        debug_content = ""
        debug_content += "CV CRITIQUE RESULTS:\n"
        debug_content += "-" * 80 + "\n"
        debug_content += f"Quality Score: {cv_critique_result.quality_score}/100\n\n"
        debug_content += f"Content Quality Feedback:\n{cv_critique_result.content_quality_feedback}\n\n"
        debug_content += f"ATS Compatibility Feedback:\n{cv_critique_result.ats_compatibility_feedback}\n\n"
        debug_content += f"Job Alignment Feedback:\n{cv_critique_result.job_alignment_feedback}\n\n"
        if cv_critique_result.critical_issues:
            debug_content += f"Critical Issues:\n{cv_critique_result.critical_issues}\n\n"
        debug_content += f"Improvement Instructions:\n{cv_critique_result.improvement_instructions}\n\n"
        debug_content += "-" * 80 + "\n"
        
        write_to_debug(debug_content, "CV CRITIQUE DEBUG INFO")
        
        # Determine if refinement is needed based on quality threshold and critical issues
        quality_score = cv_critique_result.quality_score
        has_critical_issues = bool(cv_critique_result.critical_issues and cv_critique_result.critical_issues.strip())
        # Check if improvement instructions exist (non-empty string)
        has_improvement_instructions = bool(
            cv_critique_result.improvement_instructions and 
            cv_critique_result.improvement_instructions.strip()
        )
        
        # Only refine if:
        # 1. Quality score is below threshold, OR
        # 2. There are critical issues that must be addressed
        should_refine = (quality_score < self.quality_threshold or has_critical_issues) and has_improvement_instructions
        
        if should_refine:
            state_updates["cv_needs_refinement"] = True
            state_updates["cv_critique_improvement_instructions"] = cv_critique_result.improvement_instructions
            state_updates["cv_needs_critique"] = False  # Reset flag
            logger.info(f"CV critique identified improvements - quality_score: {quality_score}, has_critical_issues: {has_critical_issues}, marking for refinement")
        else:
            state_updates["cv_needs_refinement"] = False
            state_updates["cv_needs_critique"] = False  # Reset flag
            state_updates["cv_critique_improvement_instructions"] = None # Clear instructions if no refinement needed
            if quality_score >= self.quality_threshold and not has_critical_issues:
                logger.info(f"CV critique: Quality score {quality_score} is above threshold {self.quality_threshold} and no critical issues - skipping refinement")
            else:
                logger.info("CV critique found no significant improvements needed")
        
        # Store previous quality score for comparison (to allow 2 iterations if quality improves)
        state_updates["cv_previous_quality_score"] = quality_score
        
        # Always show critique feedback to user for transparency
        critique_message = f"**CV Quality Assessment:**\n"
        critique_message += f"Quality Score: {quality_score}/100\n\n"
        critique_message += f"**Content Quality:**\n{cv_critique_result.content_quality_feedback}\n\n"
        critique_message += f"**ATS Compatibility:**\n{cv_critique_result.ats_compatibility_feedback}\n\n"
        critique_message += f"**Job Alignment:**\n{cv_critique_result.job_alignment_feedback}\n\n"
        if has_critical_issues:
            critique_message += f"**Critical Issues:**\n{cv_critique_result.critical_issues}\n\n"
        if has_improvement_instructions:
            critique_message += f"**Improvement Suggestions:**\n{cv_critique_result.improvement_instructions}\n\n"
        if should_refine:
            critique_message += "The CV will be automatically refined based on these suggestions."
        else:
            critique_message += "The CV quality is good. No automatic refinement needed."
        
        # Add critique feedback to messages for user visibility
        state_updates["messages"] = state.get("messages", []) + [{
            "role": "assistant",
            "content": critique_message
        }]
        
        logger.info("CritiqueAgent.run_cv() completed successfully")
        return state_updates
    
    def run_cover_letter(self, state):
        """
        Critique cover letter document.

        Args:
            state: The state dictionary containing generated cover letter and context

        Returns:
            dict: Updated state with cover letter critique feedback and improvement instructions
        """
        logger.info("CritiqueAgent.run_cover_letter() called")
        
        generated_cover_letter = state.get("generated_cover_letter")
        if not generated_cover_letter:
            logger.warning("No cover letter to critique - skipping")
            return {
                "messages": state.get("messages", []),
                "current_node": "critique_cover_letter",
                "cover_letter_needs_refinement": False
            }
        
        generated_cv = state.get("generated_cv")
        job_description_info = state.get("job_description_info")
        company_info = state.get("company_info")
        candidate_text = state.get("candidate_text")
        
        logger.info("Critiquing generated cover letter...")
        cover_letter_critique_result = self.critique_cover_letter(
            cover_letter_content=generated_cover_letter,
            cv_content=generated_cv,  # Include CV for context
            job_description_info=job_description_info,
            company_info=company_info,
            candidate_text=candidate_text
        )
        
        state_updates = {
            "messages": state.get("messages", []),
            "current_node": "critique_cover_letter",
            "cover_letter_critique": {
                "quality_score": cover_letter_critique_result.quality_score,
                "content_quality_feedback": cover_letter_critique_result.content_quality_feedback,
                "ats_compatibility_feedback": cover_letter_critique_result.ats_compatibility_feedback,
                "job_alignment_feedback": cover_letter_critique_result.job_alignment_feedback,
                "improvement_instructions": cover_letter_critique_result.improvement_instructions,
                "critical_issues": cover_letter_critique_result.critical_issues
            }
        }
        
        # Update Cover Letter history
        cover_letter_history = state.get("cover_letter_history", [])
        # Determine iteration number (if not already set in state, infer from history length)
        iteration = len(cover_letter_history) + 1
        
        history_entry = {
            "iteration": iteration,
            "content": generated_cover_letter,
            "score": cover_letter_critique_result.quality_score,
            "critique": cover_letter_critique_result.dict()
        }
        
        # Append to history
        state_updates["cover_letter_history"] = cover_letter_history + [history_entry]
        logger.info(f"Added cover letter iteration {iteration} to history with score {cover_letter_critique_result.quality_score}")
        
        
        # Write to debug file
        debug_content = ""
        debug_content += "COVER LETTER CRITIQUE RESULTS:\n"
        debug_content += "-" * 80 + "\n"
        debug_content += f"Quality Score: {cover_letter_critique_result.quality_score}/100\n\n"
        debug_content += f"Content Quality Feedback:\n{cover_letter_critique_result.content_quality_feedback}\n\n"
        debug_content += f"ATS Compatibility Feedback:\n{cover_letter_critique_result.ats_compatibility_feedback}\n\n"
        debug_content += f"Job Alignment Feedback:\n{cover_letter_critique_result.job_alignment_feedback}\n\n"
        if cover_letter_critique_result.critical_issues:
            debug_content += f"Critical Issues:\n{cover_letter_critique_result.critical_issues}\n\n"
        debug_content += f"Improvement Instructions:\n{cover_letter_critique_result.improvement_instructions}\n\n"
        debug_content += "-" * 80 + "\n"
        
        write_to_debug(debug_content, "COVER LETTER CRITIQUE DEBUG INFO")
        
        # Determine if refinement is needed based on quality threshold and critical issues
        quality_score = cover_letter_critique_result.quality_score
        has_critical_issues = bool(cover_letter_critique_result.critical_issues and cover_letter_critique_result.critical_issues.strip())
        # Check if improvement instructions exist (non-empty string)
        has_improvement_instructions = bool(
            cover_letter_critique_result.improvement_instructions and 
            cover_letter_critique_result.improvement_instructions.strip()
        )
        
        # Only refine if:
        # 1. Quality score is below threshold, OR
        # 2. There are critical issues that must be addressed
        should_refine = (quality_score < self.quality_threshold or has_critical_issues) and has_improvement_instructions
        
        if should_refine:
            state_updates["cover_letter_needs_refinement"] = True
            state_updates["cover_letter_critique_improvement_instructions"] = cover_letter_critique_result.improvement_instructions
            state_updates["cover_letter_needs_critique"] = False  # Reset flag
            logger.info(f"Cover letter critique identified improvements - quality_score: {quality_score}, has_critical_issues: {has_critical_issues}, marking for refinement")
        else:
            state_updates["cover_letter_needs_refinement"] = False
            state_updates["cover_letter_needs_critique"] = False  # Reset flag
            state_updates["cover_letter_critique_improvement_instructions"] = None # Clear instructions if no refinement needed
            if quality_score >= self.quality_threshold and not has_critical_issues:
                logger.info(f"Cover letter critique: Quality score {quality_score} is above threshold {self.quality_threshold} and no critical issues - skipping refinement")
            else:
                logger.info("Cover letter critique found no significant improvements needed")
        
        # Store previous quality score for comparison (to allow 2 iterations if quality improves)
        state_updates["cover_letter_previous_quality_score"] = quality_score
        
        # Always show critique feedback to user for transparency
        critique_message = f"**Cover Letter Quality Assessment:**\n"
        critique_message += f"Quality Score: {quality_score}/100\n\n"
        critique_message += f"**Content Quality:**\n{cover_letter_critique_result.content_quality_feedback}\n\n"
        critique_message += f"**ATS Compatibility:**\n{cover_letter_critique_result.ats_compatibility_feedback}\n\n"
        critique_message += f"**Job Alignment:**\n{cover_letter_critique_result.job_alignment_feedback}\n\n"
        if has_critical_issues:
            critique_message += f"**Critical Issues:**\n{cover_letter_critique_result.critical_issues}\n\n"
        if has_improvement_instructions:
            critique_message += f"**Improvement Suggestions:**\n{cover_letter_critique_result.improvement_instructions}\n\n"
        if should_refine:
            critique_message += "The cover letter will be automatically refined based on these suggestions."
        else:
            critique_message += "The cover letter quality is good. No automatic refinement needed."
        
        # Add critique feedback to messages for user visibility
        state_updates["messages"] = state.get("messages", []) + [{
            "role": "assistant",
            "content": critique_message
        }]
        
        logger.info("CritiqueAgent.run_cover_letter() completed successfully")
        return state_updates
    


