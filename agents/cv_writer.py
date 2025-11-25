from typing import Optional
import logging
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import os
from debug_utils import write_to_debug

logger = logging.getLogger(__name__)


CV_GENERATION_SYSTEM_PROMPT = (
    "You are an expert CV writer specializing in creating tailored, professional CVs that match job descriptions. "
    "Your CVs are well-structured, highlight relevant skills and experiences, and are optimized for ATS (Applicant Tracking Systems). "
    "You create compelling CVs that stand out while remaining truthful and accurate. "
    "CRITICAL: You MUST ONLY generate a CV document. You MUST NOT generate a cover letter or any other document, even if instructions mention it. "
    "If instructions mention moving content to a cover letter, ignore that part and focus only on improving the CV. "
    "IMPORTANT: You must provide two separate outputs: 1) A clean CV document with NO explanations or notes, "
    "and 2) Separate notes and explanations for the user about what you did and why. "
    "CRITICAL PRESERVATION RULE: When modification instructions are provided and an existing CV is provided, "
    "you MUST preserve ALL existing content, structure, and formatting that is NOT explicitly mentioned in the modification instructions. "
    "Only make the specific changes requested - do NOT rewrite, restructure, or add content unless explicitly requested."
)

CV_GENERATION_HUMAN_PROMPT = (
    "Create a professional, tailored CV based on the following information:\n\n"
    "**Existing CV (Base Document):**\n{candidate_cv}\n\n"
    "**Job Description:**\n{job_description}\n\n"
    "**Company Information:**\n{company_info}\n\n"
    "**Modification Instructions:**\n{modification_instructions}\n\n"
    "**Previous Modification Instructions (preserve these):**\n{previous_modification_instructions}\n\n"
    "CRITICAL: When modification instructions are provided, you MUST:\n"
    "1. **PRESERVE the existing CV structure and content** - Keep all sections, formatting, and information that is NOT mentioned in the modification instructions\n"
    "2. **ONLY make the specific changes requested** - Do NOT add, remove, or modify anything that is not explicitly mentioned in the modification instructions\n"
    "3. **Do NOT rewrite or restructure** - Only apply the exact changes requested, preserving everything else as-is (UNLESS creative liberty is explicitly granted in the instructions)\n"
    "4. **Do NOT hallucinate or add new content** unless explicitly requested in the modification instructions\n"
    "5. **Maintain consistency** - Keep the same tone, style, and format as the existing CV (UNLESS creative liberty is explicitly granted)\n\n"
    "If NO modification instructions are provided (empty), then you can create or enhance the CV based on the job description and company information.\n\n"
    "General guidelines (apply only when no specific modification instructions are provided):\n"
    "- Tailor the CV to match the job requirements and company culture\n"
    "- Highlight relevant skills, experiences, and achievements\n"
    "- Use clear, professional language\n"
    "- Structure the CV in a standard format (Contact Info, Professional Summary, Experience, Education, Skills)\n"
    "- Ensure the CV is ATS-friendly\n"
    "- Keep it concise (ideally 1-2 pages)\n"
    "- Make sure all information is accurate and truthful\n\n"
    "CRITICAL REQUIREMENTS:\n"
    "1. **ONLY generate a CV**: You must ONLY create or modify a CV document. Do NOT generate a cover letter or any other document, even if instructions mention it. "
    "If instructions mention cover letter changes, ignore those parts and focus solely on CV improvements.\n"
    "2. **CV Content**: The actual CV document - clean, professional, with NO explanations, notes, or meta-commentary. Just the CV itself.\n"
    "3. **Notes**: Separate explanations for the user about what specific changes you made to the CV based on the modification instructions. "
    "If modification instructions were provided, list ONLY the changes you made. Do NOT explain general tailoring unless no modification instructions were provided."
)


class CVGenerationResponse(BaseModel):
    """Response model for CV generation with separate CV content and notes."""
    cv_content: str = Field(
        ...,
        description="The actual CV document ONLY - clean, professional, with NO explanations, notes, or meta-commentary. Just the CV itself. MUST NOT include a cover letter or any other document."
    )
    notes: str = Field(
        ...,
        description="Explanations for the user about what changes were made, why they were made, and any important considerations. This helps the user understand the tailoring approach. It can be empty if no changes were made or if there is nothing specific to explain."
    )


class CVWriterAgent:
    """Agent for generating tailored CVs based on job descriptions and company information."""
    
    def __init__(self, output_folder: str = "generated_CVs", model: str = "openai:gpt-5-nano", temperature: float = 0.2):
        """
        Initialize the CVWriterAgent.
        
        Args:
            output_folder: Folder where generated CVs will be saved
            model: The LLM model identifier to use
            temperature: Temperature setting for the LLM

        Returns:
            None
        """
        logger.info(f"Initializing CVWriterAgent with output_folder: {output_folder}")
        self.llm = init_chat_model(model, temperature=temperature)
        logger.debug(f"CVWriterAgent LLM initialized - model: {model}, temperature: {temperature}")
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
                return f"**CRITICAL: Apply ONLY these specific changes to the existing CV. Preserve everything else exactly as-is:**\n{user_feedback}\n\nYou MUST:\n- Make ONLY the changes explicitly requested above\n- Preserve all other content, structure, and formatting\n- Do NOT add, remove, or modify anything not mentioned in the instructions\n- Do NOT rewrite or restructure the CV unless explicitly requested"
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
            response = self.llm.invoke(filter_prompt)
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
    
    def generate_cv(
        self,
        candidate_cv: Optional[str],
        job_description_info: Optional[dict],
        company_info: Optional[dict],
        modification_instructions: Optional[str] = None,
        previous_modification_instructions_formatted: Optional[str] = None
    ) -> tuple[str, str]:
        """
        Generate a tailored CV with separate content and notes.
        
        Args:
            candidate_cv: Existing CV text (if available)
            job_description_info: Extracted job description information
            company_info: Company information from search
            modification_instructions: Optional modification instructions
            previous_modification_instructions_formatted: Optional formatted previous modification instructions
            
        Returns:
            tuple[str, str]: Tuple of (CV content, notes)
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
        
        # Use structured output
        structured_llm = self.llm.with_structured_output(CVGenerationResponse)
        chain = prompt | structured_llm
        
        llm_input = {
            "candidate_cv": candidate_cv_text,
            "job_description": job_desc_text,
            "company_info": company_info_text,
            "modification_instructions": modification_instructions or "",
            "previous_modification_instructions": previous_modification_instructions_formatted or ""
        }
        logger.info("Calling LLM to generate CV with structured output...")
        logger.debug(f"LLM input - candidate_cv length: {len(candidate_cv_text)}, job_description length: {len(job_desc_text)}, company_info length: {len(company_info_text)}, modification_instructions length: {len(modification_instructions) if modification_instructions else 0}")
        
        response = chain.invoke(llm_input)
        
        cv_content = response.cv_content
        notes = response.notes
        
        logger.info(f"LLM response received - CV generated, length: {len(cv_content)} characters, notes length: {len(notes)} characters")
        logger.debug(f"CV preview: {cv_content[:200]}...")
        logger.debug(f"Notes preview: {notes[:200]}...")
        
        return cv_content, notes
    
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
    
    def save_notes(self, notes: str, filename: str = "generated_CV_notes.txt") -> str:
        """
        Save the CV generation notes to a file.
        
        Args:
            notes: The notes text to save
            filename: Name of the file to save
            
        Returns:
            str: Path to the saved file
        """
        file_path = os.path.join(self.output_folder, filename)
        logger.info(f"Saving CV notes to file: {file_path}")
        logger.debug(f"Notes length: {len(notes)} characters")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(notes)
        logger.info(f"CV notes saved successfully to {file_path}")
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
        
        # Check for critique improvement instructions (prioritize over user feedback for automatic refinement)
        cv_critique_improvement_instructions = state.get("cv_critique_improvement_instructions")
        is_refinement = state.get("cv_needs_refinement", False)
        
        logger.debug(f"State extracted - has_candidate_cv: {candidate_cv is not None}, has_job_desc: {job_description_info is not None}, has_company_info: {company_info is not None}, has_feedback: {user_feedback is not None}, is_refinement: {is_refinement}")
        
        # If this is a refinement based on critique, use critique instructions
        # Otherwise, use user feedback as before
        if is_refinement and cv_critique_improvement_instructions:
            # Filter critique instructions against user preferences
            filtered_critique_instructions = self._filter_critique_instructions_by_user_preferences(
                cv_critique_improvement_instructions
            )
            
            if filtered_critique_instructions:
                if candidate_cv:
                    modification_instructions = f"**CRITICAL: Apply ONLY these specific improvements to the existing CV. Preserve everything else exactly as-is:**\n{filtered_critique_instructions}\n\nYou MUST:\n- Make ONLY the changes explicitly requested above\n- Preserve all other content, structure, and formatting\n- Do NOT add, remove, or modify anything not mentioned in the instructions\n- Do NOT rewrite or restructure the CV unless explicitly requested"
                else:
                    modification_instructions = f"**Critique-based Improvement Instructions:**\n{filtered_critique_instructions}\n\nPlease apply these improvements to enhance the CV quality, ATS compatibility, and job alignment."
                logger.info("Using filtered critique improvement instructions for CV refinement (respecting user preferences)")
            else:
                # All critique instructions were filtered out due to user preferences
                # Skip refinement and return early - no changes needed
                logger.info("All critique instructions filtered out due to user preferences - skipping refinement")
                return {
                    "generated_cv": state.get("generated_cv"),  # Keep existing CV
                    "cv_needs_critique": False,  # Don't critique again
                    "cv_needs_refinement": False,  # Reset refinement flag
                    "cv_refinement_count": state.get("cv_refinement_count", 0),  # Keep count
                    "current_node": "cv_writer",
                    "messages": state.get("messages", []) + [{
                        "role": "assistant",
                        "content": "CV refinement skipped: All critique suggestions were filtered out based on your previous preferences."
                    }]
                }
        else:
            # Check if this is the first iteration (no generated CV yet, but we have a candidate CV)
            is_first_iteration = candidate_cv is not None and generated_cv is None
            
            if is_first_iteration and not user_feedback:
                # First iteration with existing CV: Grant creative liberty
                modification_instructions = (
                    "**CREATIVE LIBERTY GRANTED:**\n"
                    "Use the provided 'Existing CV' primarily as a source of information about the candidate's history, skills, and experience. "
                    "You are NOT bound to preserve its structure, formatting, or exact phrasing. "
                    "Your goal is to create the BEST possible CV tailored to the Job Description. "
                    "Feel free to reorder sections, rewrite bullet points for impact, and choose the most professional format. "
                    "However, do NOT hallucinate facts - only use the factual information provided in the Existing CV."
                )
                logger.info("First iteration detected - granting creative liberty to CV Writer")
            else:
                modification_instructions = self._get_modification_instructions(user_feedback, bool(candidate_cv))
        
        previous_modification_instructions_formatted = self._format_previous_modification_instructions()
        
        # Generate CV and notes
        cv_text, cv_notes = self.generate_cv(
            candidate_cv=candidate_cv,
            job_description_info=job_description_info,
            company_info=company_info,
            modification_instructions=modification_instructions,
            previous_modification_instructions_formatted=previous_modification_instructions_formatted
        )

        # Store feedback for future iterations (only if non-empty and not from critique)
        if user_feedback and user_feedback.strip() and not is_refinement:
            self.previous_modification_instructions.append(user_feedback.strip())
        
        # Determine iteration number for draft saving
        cv_history = state.get("cv_history", [])
        iteration = len(cv_history) + 1
        
        # Save CV and notes as drafts
        draft_filename = f"generated_CV_draft_{iteration}.txt"
        cv_file_path = self.save_cv(cv_text, filename=draft_filename)
        
        notes_draft_filename = f"generated_CV_notes_draft_{iteration}.txt"
        notes_file_path = self.save_notes(cv_notes, filename=notes_draft_filename)
        
        # Write to debug file
        debug_content = ""
        debug_content += f"GENERATED CV (saved to: {cv_file_path}):\n"
        debug_content += "-" * 80 + "\n"
        if is_refinement:
            debug_content += "REFINEMENT MODE: Applying critique-based improvements\n"
        debug_content += f"MODIFICATION INSTRUCTIONS:\n{modification_instructions}\n\n"
        debug_content += f"PREVIOUS MODIFICATION INSTRUCTIONS:\n{previous_modification_instructions_formatted}\n\n"
        debug_content += "-" * 80 + "\n"
        debug_content += cv_text
        debug_content += "\n\n"
        debug_content += f"CV NOTES (saved to: {notes_file_path}):\n"
        debug_content += "-" * 80 + "\n"
        debug_content += cv_notes
        debug_content += "\n\n"
        
        write_to_debug(debug_content, "CV WRITER DEBUG INFO")
        logger.info("CV generation and saving completed successfully")
        
        # Mark CV as needing critique after generation/refinement
        # Increment refinement count if this was a refinement
        cv_refinement_count = state.get("cv_refinement_count", 0)
        if is_refinement:
            cv_refinement_count += 1
            logger.info(f"CV refinement completed - refinement count: {cv_refinement_count}")
        
        state_update = {
            "generated_cv": cv_text,
            "cv_needs_critique": True,  # Always critique after generation/refinement
            "cv_needs_refinement": False,  # Reset refinement flag
            "cv_refinement_count": cv_refinement_count,  # Update refinement count
            "current_node": "cv_writer"
        }
        
        # Include notes in messages so router can process and present them to the user
        # For both initial generation and refinements, add notes to messages
        # The router will handle presenting them appropriately to the user
        message_content = f"CV has been generated and saved to {cv_file_path}.\n\n"
        message_content += f"**Notes and Explanations:**\n{cv_notes}\n\n"
        
        if not is_refinement:
            message_content += "Please review the CV and let me know if you'd like any modifications."
        else:
            message_content += "CV has been refined based on critique feedback."
            logger.info("CV refinement completed - notes included in messages for router to process")
        
        state_update["messages"] = state.get("messages", []) + [{
            "role": "assistant",
            "content": message_content
        }]
        
        return state_update

    def finalize_best_version(self, state):
        """
        Select the best version from history and save it as the final output.
        
        Args:
            state: The state dictionary containing cv_history
            
        Returns:
            dict: Updated state with final CV
        """
        logger.info("CVWriterAgent.finalize_best_version() called")
        
        cv_history = state.get("cv_history", [])
        if not cv_history:
            logger.warning("No CV history found. Cannot finalize best version.")
            return {"current_node": "finalize_cv"}
            
        # Find the version with the highest score
        # If scores are equal, prefer the later version (higher iteration number)
        best_version = max(cv_history, key=lambda x: (x.get("score", 0), x.get("iteration", 0)))
        
        logger.info(f"Selected best version: Iteration {best_version.get('iteration')} with score {best_version.get('score')}")
        
        final_cv_text = best_version.get("content")
        
        # Save as the final generated_CV.txt
        final_path = self.save_cv(final_cv_text, filename="generated_CV.txt")
        
        message = f"**Finalization Complete:**\nSelected version from iteration {best_version.get('iteration')} (Score: {best_version.get('score')}) as the best version.\nSaved to {final_path}"
        
        return {
            "generated_cv": final_cv_text,
            "current_node": "finalize_cv",
            "messages": state.get("messages", []) + [{
                "role": "assistant",
                "content": message
            }]
        }

