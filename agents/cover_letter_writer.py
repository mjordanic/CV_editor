from typing import Optional
import logging
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import os
from debug_utils import write_to_debug

logger = logging.getLogger(__name__)


COVER_LETTER_GENERATION_SYSTEM_PROMPT = (
    "You are an expert cover letter writer specializing in creating compelling, personalized cover letters "
    "that match job descriptions and company cultures. Your cover letters are engaging, professional, and "
    "effectively communicate why the candidate is a perfect fit for the role. "
    "IMPORTANT: You must provide two separate outputs: 1) A clean cover letter document with NO explanations or notes, "
    "and 2) Separate notes and explanations for the user about what you did and why. "
    "CRITICAL PRESERVATION RULE: When modification instructions are provided and an existing cover letter is provided, "
    "you MUST preserve ALL existing content, structure, and formatting that is NOT explicitly mentioned in the modification instructions. "
    "Only make the specific changes requested - do NOT rewrite, restructure, or add content unless explicitly requested."
)

COVER_LETTER_GENERATION_HUMAN_PROMPT = (
    "Create a professional, tailored cover letter based on the following information:\n\n"
    "**Existing Cover Letter (Base Document):**\n{candidate_cover_letter}\n\n"
    "**Candidate CV (for context):**\n{candidate_cv}\n\n"
    "**Job Description:**\n{job_description}\n\n"
    "**Company Information:**\n{company_info}\n\n"
    "**Modification Instructions:**\n{modification_instructions}\n\n"
    "**Previous Modification Instructions (preserve these):**\n{previous_modification_instructions}\n\n"
    "CRITICAL: When modification instructions are provided, you MUST:\n"
    "1. **PRESERVE the existing cover letter structure and content** - Keep all paragraphs, formatting, and information that is NOT mentioned in the modification instructions\n"
    "2. **ONLY make the specific changes requested** - Do NOT add, remove, or modify anything that is not explicitly mentioned in the modification instructions\n"
    "3. **Do NOT rewrite or restructure** - Only apply the exact changes requested, preserving everything else as-is\n"
    "4. **Do NOT hallucinate or add new content** unless explicitly requested in the modification instructions\n"
    "5. **Maintain consistency** - Keep the same tone, style, and format as the existing cover letter\n\n"
    "If NO modification instructions are provided (empty), then you can create or enhance the cover letter based on the job description and company information.\n\n"
    "General guidelines (apply only when no specific modification instructions are provided):\n"
    "- Tailor the cover letter to the specific job and company\n"
    "- Address the hiring manager or use a professional greeting\n"
    "- Start with a strong opening that captures attention\n"
    "- Highlight relevant skills and experiences that match the job requirements\n"
    "- Show enthusiasm for the role and company\n"
    "- Reference specific details from the job description and company information\n"
    "- Explain why you're a good fit for the position\n"
    "- Keep it concise (ideally 3-4 paragraphs, one page)\n"
    "- Use professional, engaging language\n"
    "- End with a strong closing and call to action\n\n"
    "CRITICAL: Provide your response in two parts:\n"
    "1. **Cover Letter Content**: The actual cover letter document - clean, professional, with NO explanations, notes, or meta-commentary. Just the cover letter itself.\n"
    "2. **Notes**: Separate explanations for the user about what specific changes you made to the cover letter based on the modification instructions. "
    "If modification instructions were provided, list ONLY the changes you made. Do NOT explain general tailoring unless no modification instructions were provided."
)


class CoverLetterGenerationResponse(BaseModel):
    """Response model for cover letter generation with separate cover letter content and notes."""
    cover_letter_content: str = Field(
        ...,
        description="The actual cover letter document - clean, professional, with NO explanations, notes, or meta-commentary. Just the cover letter itself."
    )
    notes: str = Field(
        ...,
        description="Explanations for the user about what changes were made, why they were made, and any important considerations. This helps the user understand the tailoring approach. It can be empty if no changes were made or if there is nothing specific to explain."
    )


class CoverLetterWriterAgent:
    """Agent for generating tailored cover letters based on job descriptions and company information."""
    
    def __init__(
        self, 
        output_folder: str = "data/generated_CVs", 
        model: str = "openai:gpt-5-mini", 
        temperature: float = 0.2,
        filter_model: str = "openai:gpt-5-nano",
        filter_temperature: float = 0.0
    ):
        """
        Initialize the CoverLetterWriterAgent.
        
        Args:
            output_folder: Folder where generated cover letters will be saved
            model: The LLM model identifier to use for cover letter generation
            temperature: Temperature setting for the main LLM
            filter_model: The LLM model for filtering operations
            filter_temperature: Temperature for filtering LLM

        Returns:
            None
        """
        logger.info(f"Initializing CoverLetterWriterAgent with output_folder: {output_folder}")
        
        # Main LLM for cover letter generation
        self.llm = init_chat_model(model, temperature=temperature)
        logger.debug(f"CoverLetterWriterAgent LLM initialized - model: {model}, temperature: {temperature}")
        
        # Separate LLM for filtering operations
        self.filter_llm = init_chat_model(filter_model, temperature=filter_temperature)
        logger.debug(f"CoverLetterWriterAgent filter LLM initialized - model: {filter_model}, temperature: {filter_temperature}")
        
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
        if job_description_info.get("location"):
            parts.append(f"Location: {job_description_info['location']}")
        if job_description_info.get("candidate_minimal_requirements"):
            parts.append(f"\nRequirements:\n{job_description_info['candidate_minimal_requirements']}")
        if job_description_info.get("job_description"):
            parts.append(f"\nFull Job Description:\n{job_description_info['job_description']}")
        
        return "\n".join(parts)
    
    def _get_modification_instructions(self, user_feedback: Optional[str], has_existing_cover_letter: bool) -> str:
        """
        Build instructions describing how to revise an existing cover letter.

        Args:
            user_feedback: Raw feedback text provided by the user, if any.
            has_existing_cover_letter: Flag indicating whether a prior cover letter exists.

        Returns:
            str: Instructional text injected into the prompt or an empty string if none needed.
        """
        # Always prioritize user feedback when provided
        if user_feedback and user_feedback.strip():
            if has_existing_cover_letter:
                return f"**CRITICAL: Apply ONLY these specific changes to the existing cover letter. Preserve everything else exactly as-is:**\n{user_feedback}\n\nYou MUST:\n- Make ONLY the changes explicitly requested above\n- Preserve all other content, structure, and formatting\n- Do NOT add, remove, or modify anything not mentioned in the instructions\n- Do NOT rewrite or restructure the cover letter unless explicitly requested"
            return f"**User Feedback/Modification Request:**\n{user_feedback}\n\nPlease incorporate these changes into the cover letter."
        
        # Only use hardcoded message when there's no user feedback
        if has_existing_cover_letter:
            return "Please review the existing cover letter and make improvements based on the job description and company information."
        
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
    
    def _filter_critique_instructions_by_user_preferences(
        self, 
        critique_instructions: str
    ) -> str:
        """
        Filter critique instructions against user preferences from previous_modification_instructions.
        If user previously stated preferences (e.g., "don't include education"), 
        omit conflicting critique suggestions.

        Args:
            critique_instructions: Critique improvement instructions to filter

        Returns:
            str: Filtered critique instructions that respect user preferences
        """
        if not critique_instructions or not critique_instructions.strip():
            return critique_instructions
        
        if not self.previous_modification_instructions:
            return critique_instructions
        
        # Get formatted user preferences using existing method
        user_preferences_text = self._format_previous_modification_instructions()
        
        if user_preferences_text == "No previous modification instructions.":
            return critique_instructions
        
        # Use LLM to filter critique instructions based on user preferences
        # This ensures we respect user's explicit preferences even if critique suggests otherwise
        filter_prompt = f"""You are filtering critique improvement instructions based on user preferences.

**User Preferences (from previous feedback):**
{user_preferences_text}

**Critique Improvement Instructions:**
{critique_instructions}

**Task:**
Filter the critique instructions to respect user preferences. 
- If critique suggests something that conflicts with user preferences, OMIT that suggestion
- If critique suggests something that aligns with user preferences, KEEP it
- If user explicitly stated they don't want something (e.g., "don't include education", "no education section"), 
  remove any critique suggestions that would add or modify that element
- Keep all other valid critique suggestions that don't conflict
- If a critique instruction contains multiple parts and only some conflict, keep the non-conflicting parts and remove only the conflicting parts

Return ONLY the filtered critique instructions. If all suggestions conflict, return empty string.
Do NOT add explanations or notes - just return the filtered instructions. 
CRITICAL: You MUST NOT add any new content, suggestions, or changes that are not explicitly requested in the critique instructions. 
You can ONLY remove parts of the critique instructions that conflict with user preferences. Do not add, modify, or rephrase anything beyond removing conflicting parts."""

        try:
            logger.info("Filtering critique instructions against user preferences...")
            response = self.filter_llm.invoke(filter_prompt)
            filtered_instructions = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            
            if filtered_instructions and filtered_instructions.lower() not in ["none", "no changes needed", ""]:
                logger.info(f"Filtered critique instructions - original length: {len(critique_instructions)}, filtered length: {len(filtered_instructions)}")
                return filtered_instructions
            else:
                logger.info("All critique instructions were filtered out due to user preferences")
                return ""
        except Exception as e:
            logger.warning(f"Error filtering critique instructions: {e}. Using original instructions.")
            return critique_instructions
    
    def generate_cover_letter(
        self,
        candidate_cv: Optional[str],
        candidate_cover_letter: Optional[str],
        job_description_info: Optional[dict],
        company_info: Optional[dict],
        modification_instructions: Optional[str] = None,
        previous_modification_instructions_formatted: Optional[str] = None
    ) -> tuple[str, str]:
        """
        Generate a tailored cover letter with separate content and notes.
        
        Args:
            candidate_cv: Existing CV text (if available)
            candidate_cover_letter: Existing cover letter text (if available)
            job_description_info: Extracted job description information
            company_info: Company information from search
            modification_instructions: Optional modification instructions
            previous_modification_instructions_formatted: Optional formatted previous modification instructions
            
        Returns:
            tuple[str, str]: Tuple of (cover letter content, notes)
        """
        # Format inputs
        candidate_cv_text = candidate_cv or "No CV provided."
        candidate_cover_letter_text = candidate_cover_letter or "No existing cover letter provided. Create a new cover letter."
        job_desc_text = self._format_job_description(job_description_info)
        company_info_text = self._format_company_info(company_info)
        
        logger.info(f"Generating cover letter - candidate_cv_length: {len(candidate_cv_text) if candidate_cv_text else 0}, candidate_cover_letter_length: {len(candidate_cover_letter_text) if candidate_cover_letter_text else 0}, has_modification_instructions: {bool(modification_instructions)}")
        logger.debug(f"Job description length: {len(job_desc_text)}, company info length: {len(company_info_text)}")
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", COVER_LETTER_GENERATION_SYSTEM_PROMPT),
            ("human", COVER_LETTER_GENERATION_HUMAN_PROMPT)
        ])
        
        # Use structured output
        structured_llm = self.llm.with_structured_output(CoverLetterGenerationResponse)
        chain = prompt | structured_llm
        
        llm_input = {
            "candidate_cv": candidate_cv_text,
            "candidate_cover_letter": candidate_cover_letter_text,
            "job_description": job_desc_text,
            "company_info": company_info_text,
            "modification_instructions": modification_instructions or "",
            "previous_modification_instructions": previous_modification_instructions_formatted or ""
        }
        logger.info("Calling LLM to generate cover letter with structured output...")
        logger.debug(f"LLM input - candidate_cv length: {len(candidate_cv_text)}, candidate_cover_letter length: {len(candidate_cover_letter_text)}, job_description length: {len(job_desc_text)}, company_info length: {len(company_info_text)}, modification_instructions length: {len(modification_instructions) if modification_instructions else 0}")
        
        response = chain.invoke(llm_input)
        
        cover_letter_content = response.cover_letter_content
        notes = response.notes
        
        logger.info(f"LLM response received - cover letter generated, length: {len(cover_letter_content)} characters, notes length: {len(notes)} characters")
        logger.debug(f"Cover letter preview: {cover_letter_content[:200]}...")
        logger.debug(f"Notes preview: {notes[:200]}...")
        
        return cover_letter_content, notes
    
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
        logger.info(f"Saving cover letter to file: {file_path}")
        logger.debug(f"Cover letter text length: {len(cover_letter_text)} characters")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(cover_letter_text)
        logger.info(f"Cover letter saved successfully to {file_path}")
        return file_path
    
    def save_notes(self, notes: str, filename: str = "generated_cover_letter_notes.txt") -> str:
        """
        Save the cover letter generation notes to a file.
        
        Args:
            notes: The notes text to save
            filename: Name of the file to save
            
        Returns:
            str: Path to the saved file
        """
        file_path = os.path.join(self.output_folder, filename)
        logger.info(f"Saving cover letter notes to file: {file_path}")
        logger.debug(f"Notes length: {len(notes)} characters")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(notes)
        logger.info(f"Cover letter notes saved successfully to {file_path}")
        return file_path
    
    def run(self, state):
        """
        Main method to generate and save a cover letter.
        
        Args:
            state: The state dictionary containing candidate_text, job_description_info, company_info, and user_feedback
            
        Returns:
            dict: Updated state with generated cover letter
        """
        logger.info("CoverLetterWriterAgent.run() called")
        
        candidate_text = state.get("candidate_text", {})
        generated_cv = state.get("generated_cv")
        candidate_cv = generated_cv if generated_cv else (candidate_text.get("cv") if candidate_text else None)
        # Use generated cover letter if it exists (for modifications), otherwise use original
        generated_cover_letter = state.get("generated_cover_letter")
        candidate_cover_letter = generated_cover_letter if generated_cover_letter else (candidate_text.get("cover_letter") if candidate_text else None)
        job_description_info = state.get("job_description_info")
        company_info = state.get("company_info")
        user_feedback = state.get("user_feedback")
        
        # Check for critique improvement instructions (prioritize over user feedback for automatic refinement)
        cover_letter_critique_improvement_instructions = state.get("cover_letter_critique_improvement_instructions")
        is_refinement = state.get("cover_letter_needs_refinement", False)
        
        logger.debug(f"State extracted - has_candidate_cv: {candidate_cv is not None}, has_candidate_cover_letter: {candidate_cover_letter is not None}, has_job_desc: {job_description_info is not None}, has_company_info: {company_info is not None}, has_feedback: {user_feedback is not None}, is_refinement: {is_refinement}")
        
        # If this is a refinement based on critique, use critique instructions
        # Otherwise, use user feedback as before
        if is_refinement and cover_letter_critique_improvement_instructions:
            # Filter critique instructions against user preferences
            filtered_critique_instructions = self._filter_critique_instructions_by_user_preferences(
                cover_letter_critique_improvement_instructions
            )
            
            if filtered_critique_instructions:
                if candidate_cover_letter:
                    modification_instructions = f"**CRITICAL: Apply ONLY these specific improvements to the existing cover letter. Preserve everything else exactly as-is:**\n{filtered_critique_instructions}\n\nYou MUST:\n- Make ONLY the changes explicitly requested above\n- Preserve all other content, structure, and formatting\n- Do NOT add, remove, or modify anything not mentioned in the instructions\n- Do NOT rewrite or restructure the cover letter unless explicitly requested"
                else:
                    modification_instructions = f"**Critique-based Improvement Instructions:**\n{filtered_critique_instructions}\n\nPlease apply these improvements to enhance the cover letter quality, ATS compatibility, and job alignment."
                logger.info("Using filtered critique improvement instructions for cover letter refinement (respecting user preferences)")
            else:
                # All critique instructions were filtered out due to user preferences
                # Skip refinement and return early - no changes needed
                logger.info("All critique instructions filtered out due to user preferences - skipping refinement")
                return {
                    "generated_cover_letter": state.get("generated_cover_letter"),  # Keep existing cover letter
                    "cover_letter_needs_critique": False,  # Don't critique again
                    "cover_letter_needs_refinement": False,  # Reset refinement flag
                    "cover_letter_refinement_count": state.get("cover_letter_refinement_count", 0),  # Keep count
                    "current_node": "cover_letter_writer",
                    "messages": state.get("messages", []) + [{
                        "role": "assistant",
                        "content": "Cover letter refinement skipped: All critique suggestions were filtered out based on your previous preferences."
                    }]
                }
        else:
            modification_instructions = self._get_modification_instructions(user_feedback, bool(candidate_cover_letter))
        
        previous_modification_instructions_formatted = self._format_previous_modification_instructions()
        
        # Generate cover letter and notes
        cover_letter_text, cover_letter_notes = self.generate_cover_letter(
            candidate_cv=candidate_cv,
            candidate_cover_letter=candidate_cover_letter,
            job_description_info=job_description_info,
            company_info=company_info,
            modification_instructions=modification_instructions,
            previous_modification_instructions_formatted=previous_modification_instructions_formatted
        )

        # Store feedback for future iterations (only if non-empty and not from critique)
        if user_feedback and user_feedback.strip() and not is_refinement:
            self.previous_modification_instructions.append(user_feedback.strip())
        
        # Determine iteration number for draft saving
        cover_letter_history = state.get("cover_letter_history", [])
        iteration = len(cover_letter_history) + 1
        
        # Save cover letter and notes as drafts
        draft_filename = f"generated_cover_letter_draft_{iteration}.txt"
        cover_letter_file_path = self.save_cover_letter(cover_letter_text, filename=draft_filename)
        
        notes_draft_filename = f"generated_cover_letter_notes_draft_{iteration}.txt"
        notes_file_path = self.save_notes(cover_letter_notes, filename=notes_draft_filename)
        
        # Write to debug file
        debug_content = ""
        debug_content += f"GENERATED COVER LETTER (saved to: {cover_letter_file_path}):\n"
        debug_content += "-" * 80 + "\n"
        if is_refinement:
            debug_content += "REFINEMENT MODE: Applying critique-based improvements\n"
        debug_content += f"MODIFICATION INSTRUCTIONS:\n{modification_instructions}\n\n"
        debug_content += f"PREVIOUS MODIFICATION INSTRUCTIONS:\n{previous_modification_instructions_formatted}\n\n"
        debug_content += "-" * 80 + "\n"
        debug_content += cover_letter_text
        debug_content += "\n\n"
        debug_content += f"COVER LETTER NOTES (saved to: {notes_file_path}):\n"
        debug_content += "-" * 80 + "\n"
        debug_content += cover_letter_notes
        debug_content += "\n\n"
        
        write_to_debug(debug_content, "COVER LETTER WRITER DEBUG INFO")
        logger.info("Cover letter generation and saving completed successfully")
        
        # Mark cover letter as needing critique after generation/refinement
        # Increment refinement count if this was a refinement
        cover_letter_refinement_count = state.get("cover_letter_refinement_count", 0)
        if is_refinement:
            cover_letter_refinement_count += 1
            logger.info(f"Cover letter refinement completed - refinement count: {cover_letter_refinement_count}")
        
        state_update = {
            "generated_cover_letter": cover_letter_text,
            "cover_letter_needs_critique": True,  # Always critique after generation/refinement
            "cover_letter_needs_refinement": False,  # Reset refinement flag
            "cover_letter_refinement_count": cover_letter_refinement_count,  # Update refinement count
            "current_node": "cover_letter_writer"
        }
        
        # Include notes in messages so router can process and present them to the user
        # For both initial generation and refinements, add notes to messages
        # The router will handle presenting them appropriately to the user
        message_content = f"Cover letter has been generated and saved to {cover_letter_file_path}.\n\n"
        message_content += f"**Notes and Explanations:**\n{cover_letter_notes}\n\n"
        
        if not is_refinement:
            message_content += "Please review the cover letter and let me know if you'd like any modifications."
        else:
            message_content += "Cover letter has been refined based on critique feedback."
            logger.info("Cover letter refinement completed - notes included in messages for router to process")
        
        state_update["messages"] = state.get("messages", []) + [{
            "role": "assistant",
            "content": message_content
        }]
        
        return state_update

    def finalize_best_version(self, state):
        """
        Select the best version from history and save it as the final output.
        
        Args:
            state: The state dictionary containing cover_letter_history
            
        Returns:
            dict: Updated state with final cover letter
        """
        logger.info("CoverLetterWriterAgent.finalize_best_version() called")
        
        cover_letter_history = state.get("cover_letter_history", [])
        if not cover_letter_history:
            logger.warning("No cover letter history found. Cannot finalize best version.")
            return {"current_node": "finalize_cover_letter"}
            
        # Find the version with the highest score
        # If scores are equal, prefer the later version (higher iteration number)
        best_version = max(cover_letter_history, key=lambda x: (x.get("score", 0), x.get("iteration", 0)))
        
        logger.info(f"Selected best version: Iteration {best_version.get('iteration')} with score {best_version.get('score')}")
        
        final_cover_letter_text = best_version.get("content")
        
        # Save as the final generated_cover_letter.txt
        final_path = self.save_cover_letter(final_cover_letter_text, filename="generated_cover_letter.txt")
        
        message = f"**Finalization Complete:**\nSelected version from iteration {best_version.get('iteration')} (Score: {best_version.get('score')}) as the best version.\nSaved to {final_path}"
        
        return {
            "generated_cover_letter": final_cover_letter_text,
            "current_node": "finalize_cover_letter",
            "messages": state.get("messages", []) + [{
                "role": "assistant",
                "content": message
            }]
        }

