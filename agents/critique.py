from typing import Optional, Literal
import logging
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from debug_utils import write_to_debug
from agents.skill_check import unsupported_job_terms, unsupported_terms

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
    "Example: If generated CV says 'Expert in Kubernetes' but original CV only mentions 'Docker', flag this as a potential hallucination.\n"
    "NEVER suggest adding skills, tools, platforms, certifications or experience that are not in the 'Original Candidate Information', "
    "even when the job description asks for them. For a missing requirement, suggest highlighting the closest experience the candidate does have.\n"
    "Every critical issue MUST have a matching fix in improvement_instructions (e.g. 'In the Skills section, remove \"Azure\"')."
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
                # Full text: the critique checks the draft against it for invented skills, so a
                # truncated CV makes real skills look invented.
                parts.append(f"Original CV:\n{cv_text}")
            else:
                parts.append("Original CV: Not provided")
        else:
            parts.append("Original CV: Not provided")
        
        if candidate_text.get("cover_letter"):
            cl_text = candidate_text["cover_letter"]
            if cl_text:
                parts.append(f"Original Cover Letter:\n{cl_text}")
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
        return self._run("cv", state)

    def run_cover_letter(self, state):
        """
        Critique cover letter document.

        Args:
            state: The state dictionary containing generated cover letter and context

        Returns:
            dict: Updated state with cover letter critique feedback and improvement instructions
        """
        return self._run("cover_letter", state)

    def _run(self, kind: str, state):
        """
        Critique the latest draft of 'cv' or 'cover_letter' and decide whether it goes back for a redraft.

        Besides the LLM critique, two deterministic guards run here because a light critic model
        misses them: tools and skills the candidate never mentioned are flagged and turned into
        removal instructions (see agents/skill_check.py), and a draft identical to the last one of
        this cycle is not critiqued or stored again.
        """
        label = "CV" if kind == "cv" else "Cover letter"
        node = f"critique_{kind}"
        logger.info(f"CritiqueAgent._run({kind}) called")

        document = state.get(f"generated_{kind}")
        if not document:
            logger.warning(f"No {label.lower()} to critique - skipping")
            return {
                "messages": state.get("messages", []),
                "current_node": node,
                f"{kind}_needs_refinement": False
            }

        history = state.get(f"{kind}_history", []) or []
        cycle_start = state.get(f"{kind}_cycle_start") or 0
        if len(history) > cycle_start and history[-1].get("content") == document:
            # The writer returned the draft unchanged (e.g. every critique suggestion was filtered
            # out). It was critiqued already: storing it again would make a new "iteration" (and a
            # draft PDF) holding the previous draft's text.
            logger.info(f"{label} is unchanged since iteration {history[-1].get('iteration')} - not critiquing it again")
            return {
                "current_node": node,
                f"{kind}_needs_refinement": False,
                f"{kind}_needs_critique": False,
                f"{kind}_critique_improvement_instructions": None,
                "messages": state.get("messages", []) + [{
                    "role": "assistant",
                    "content": f"The {label.lower()} did not change in the last pass, so iteration {history[-1].get('iteration')} stands.",
                }],
            }

        if kind == "cv":
            result = self.critique_cv(
                cv_content=document,
                job_description_info=state.get("job_description_info"),
                company_info=state.get("company_info"),
                candidate_text=state.get("candidate_text")
            )
        else:
            result = self.critique_cover_letter(
                cover_letter_content=document,
                cv_content=state.get("generated_cv"),  # Include CV for context
                job_description_info=state.get("job_description_info"),
                company_info=state.get("company_info"),
                candidate_text=state.get("candidate_text")
            )

        instructions = _clean_text(result.improvement_instructions)
        critical_issues = _clean_text(result.critical_issues)

        if critical_issues and not instructions:
            # A critical issue with no instruction left the writer nothing to do, so the problem stayed.
            instructions = f"Fix these critical issues:\n{critical_issues}"

        invented = unsupported_terms(document, state)
        if invented:
            names = ", ".join(invented)
            logger.warning(f"{label} mentions tools/skills not found in the candidate's material: {names}")
            critical_issues = _join(critical_issues, (
                f"Mentions tools or skills not found in the candidate's CV, portfolio or messages: {names}."
            ))
            instructions = _join(instructions, (
                f"Remove every mention of {names}, wherever it appears. The candidate's CV and "
                f"portfolio do not mention them. Do not replace them with other tools the candidate does not list."
            ))
        if instructions:
            forbidden = unsupported_job_terms(state, instructions)
            if forbidden:
                instructions = _join(instructions, (
                    f"Do NOT add or keep these tools or skills: {', '.join(forbidden)}. "
                    f"The candidate's CV and portfolio do not mention them, even if the job asks for them."
                ))

        quality_score = result.quality_score
        critique_dict = {
            "quality_score": quality_score,
            "content_quality_feedback": result.content_quality_feedback,
            "ats_compatibility_feedback": result.ats_compatibility_feedback,
            "job_alignment_feedback": result.job_alignment_feedback,
            "improvement_instructions": instructions,
            "critical_issues": critical_issues,
            "unsupported_terms": invented,
        }
        state_updates = {
            "messages": state.get("messages", []),
            "current_node": node,
            f"{kind}_critique": critique_dict,
        }

        iteration = len(history) + 1
        state_updates[f"{kind}_history"] = history + [{
            "iteration": iteration,
            "content": document,
            "score": quality_score,
            "critique": critique_dict,
            "unsupported_terms": invented,
        }]
        logger.info(f"Added {label.lower()} iteration {iteration} to history with score {quality_score}")

        debug_content = f"{label.upper()} CRITIQUE RESULTS:\n"
        debug_content += "-" * 80 + "\n"
        debug_content += f"Quality Score: {quality_score}/100\n\n"
        debug_content += f"Content Quality Feedback:\n{result.content_quality_feedback}\n\n"
        debug_content += f"ATS Compatibility Feedback:\n{result.ats_compatibility_feedback}\n\n"
        debug_content += f"Job Alignment Feedback:\n{result.job_alignment_feedback}\n\n"
        if critical_issues:
            debug_content += f"Critical Issues:\n{critical_issues}\n\n"
        debug_content += f"Improvement Instructions:\n{instructions}\n\n"
        debug_content += "-" * 80 + "\n"
        write_to_debug(debug_content, f"{label.upper()} CRITIQUE DEBUG INFO")

        # Refine only if the score is below threshold or there are critical issues, and there is
        # something for the writer to do.
        has_critical_issues = bool(critical_issues)
        should_refine = (quality_score < self.quality_threshold or has_critical_issues) and bool(instructions)

        state_updates[f"{kind}_needs_refinement"] = should_refine
        state_updates[f"{kind}_needs_critique"] = False
        state_updates[f"{kind}_critique_improvement_instructions"] = instructions if should_refine else None
        if should_refine:
            logger.info(f"{label} critique identified improvements - quality_score: {quality_score}, has_critical_issues: {has_critical_issues}, marking for refinement")
        elif quality_score >= self.quality_threshold and not has_critical_issues:
            logger.info(f"{label} critique: Quality score {quality_score} is above threshold {self.quality_threshold} and no critical issues - skipping refinement")
        else:
            logger.info(f"{label} critique found no significant improvements needed")

        # Keep the previous draft's score next to the new one so the router can tell whether the
        # last redraft raised the score. The first draft of a new cycle has no previous draft.
        state_updates[f"{kind}_previous_quality_score"] = history[-1]["score"] if len(history) > cycle_start else None

        # Always show critique feedback to user for transparency
        critique_message = f"**{'CV' if kind == 'cv' else 'Cover Letter'} Quality Assessment:**\n"
        critique_message += f"Quality Score: {quality_score}/100\n\n"
        critique_message += f"**Content Quality:**\n{result.content_quality_feedback}\n\n"
        critique_message += f"**ATS Compatibility:**\n{result.ats_compatibility_feedback}\n\n"
        critique_message += f"**Job Alignment:**\n{result.job_alignment_feedback}\n\n"
        if has_critical_issues:
            critique_message += f"**Critical Issues:**\n{critical_issues}\n\n"
        if instructions:
            critique_message += f"**Improvement Suggestions:**\n{instructions}\n\n"
        if should_refine:
            critique_message += f"The {label.lower()} will be automatically refined based on these suggestions."
        else:
            critique_message += f"The {label.lower()} quality is good. No automatic refinement needed."
        state_updates["messages"] = state.get("messages", []) + [{
            "role": "assistant",
            "content": critique_message
        }]

        logger.info(f"CritiqueAgent._run({kind}) completed successfully")
        return state_updates


# Answers a light model gives when it means "nothing": the prompt asks for an empty string, and it
# sometimes returns the two quote characters themselves or a phrase.
_EMPTY_ANSWERS = {"", "none", "n/a", "na", "null", "nothing", "no improvements needed", "no changes needed",
                  "no critical issues", "no issues"}


def _clean_text(text: str | None) -> str:
    """Critique text with quote-only and 'none'-style answers turned into an empty string."""
    stripped = (text or "").strip()
    unquoted = stripped.strip("\"'`“”‘’ ").rstrip(".").strip().lower()
    return "" if unquoted in _EMPTY_ANSWERS else stripped


def _join(first: str, second: str) -> str:
    return f"{first}\n{second}" if first else second
