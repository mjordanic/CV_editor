from typing import Optional, Literal
import logging
from langchain.chat_models import init_chat_model
from config.config_loader import local_model_kwargs
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from debug_utils import write_to_debug

from agents.work_context import describe as describe_work_context, with_preference

logger = logging.getLogger(__name__)


CRITIQUE_SYSTEM_PROMPT = (
    "You are an expert CV and cover letter reviewer specializing in quality assessment, "
    "ATS (Applicant Tracking System) screening, and job alignment analysis. "
    "Your PRIMARY GOAL is that the document is the most convincing TRUTHFUL match to this job ad for a human recruiter, "
    "and that it uses the ad's own terms wherever the candidate's material supports them, so automatic (ATS) screening sees them too. "
    "ATS compatibility is a means to that goal, not the goal: never ask for an ad term the candidate's material does not support. "
    "Your role is to evaluate documents objectively and provide clear, actionable improvement instructions. "
    "You assess content quality, ATS compatibility, alignment with job requirements, and overall professionalism. "
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
    "\n\nGAPS ARE NOT DOCUMENT DEFECTS:\n"
    "A job requirement the candidate does not meet (a missing licence, certificate, qualification, years of experience "
    "or skill level) cannot be fixed by rewriting the document. Mention such gaps only in job_alignment_feedback, so the "
    "user sees them. Do NOT list them as critical issues, do NOT lower the score for them as if the writing were at fault, "
    "and do NOT write improvement instructions about them: no 'if true, add ...', no statements of willingness, "
    "availability or plans the candidate has not made, and no sentences that point the employer to what the candidate lacks."
)

CRITIQUE_HUMAN_PROMPT = (
    "Evaluate the following document and provide improvement instructions:\n\n"
    "**Document Type:** {document_type}\n\n"
    "**Document Content:**\n{document_content}\n\n"
    "**Job Description:**\n{job_description}\n\n"
    "**Company Information:**\n{company_info}\n\n"
    "**Original Candidate Information:**\n{candidate_info}\n\n"
    "**The Ad's Requirements (mark each one in 'requirements'):**\n{requirements}\n\n"
    "Please evaluate the document on:\n"
    "1. **Content Quality**: Structure, clarity, professionalism, impact, and completeness\n"
    "2. **ATS Compatibility**: Formatting, keywords, structure, parsing-friendliness, and **AI Screening Probability**\n"
    "3. **Job Alignment**: How well the document matches job requirements, company culture, and role expectations, in the ad's own terms where true\n"
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
    "- CRITICAL: Return empty string \"\" if no improvements are needed - do NOT write 'No improvements needed' or similar text\n\n"
    "REQUIREMENTS CHECK: for each requirement listed above, say whether this document shows it ('shown', with a short quote "
    "from the document), whether the candidate's material shows the candidate lacks it ('gap': not a document defect; leave "
    "it out of the instructions), or whether the candidate's material shows they have it but this document does not show it "
    "('missing': a real defect; add an improvement instruction to show it, quoting the candidate's material)."
)


class RequirementCheck(BaseModel):
    """One requirement of the ad, judged for one draft."""
    requirement: str = Field(..., description="The requirement, in short form, as listed")
    status: Literal["shown", "gap", "missing"] = Field(
        ...,
        description="shown: the document shows the candidate meets it; gap: the candidate's material shows they lack it "
        "(not a document defect); missing: the candidate's material shows they have it but the document does not show it"
    )
    quote: str = Field(default="", description="Short quote from the document (shown) or from the candidate's material (missing); empty for a gap")


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
    requirements: list[RequirementCheck] = Field(
        default_factory=list,
        description="One entry per requirement listed in the prompt (if none are listed, per main requirement of the job description)"
    )


GROUNDING_PROMPT = (
    "You check a {document_type} for claims that its sources do not support.\n\n"
    "**Candidate's own material** (the only evidence about the candidate; what the user said in this chat is the "
    "user's own statement about themselves and counts as their material):\n{candidate_info}\n\n"
    "**Job ad and company research** (the only evidence about the employer):\n{employer_info}\n\n"
    "**{document_type} to check:**\n{document_content}\n\n"
    "1. unsupported_claims: list every skill, tool, platform, qualification, certification, job, project, achievement, "
    "figure or level of experience that the {document_type} claims for the candidate but that the candidate's material "
    "does not support. "
    "Rephrasing, summarising and obvious inference are supported (e.g. 'built a React app' supports 'JavaScript'). "
    "Statements about the employer, the role or the job's requirements are not claims about the candidate; do not list them here. "
    "Statements that the candidate lacks something, or plans or is willing to obtain it, are not claims to have it; "
    "list such a statement only if neither the candidate's material nor the user's requests say it.\n"
    "2. unsupported_employer_claims: list every statement of fact about the employer (its products, history, funding, "
    "size, values, achievements, plans) that neither the job ad nor the company research states. Generic courtesy "
    "('I would welcome the chance to join your team') is not a factual claim.\n"
    "Quote each unsupported claim briefly, as it appears in the document. Return empty lists if everything is supported."
)


class GroundingResponse(BaseModel):
    """Claims in a draft that the candidate's own material (or, for the employer, the ad and research) does not support."""
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Each unsupported claim about the candidate, quoted briefly from the document. Empty if everything is supported."
    )
    unsupported_employer_claims: list[str] = Field(
        default_factory=list,
        description="Each statement of fact about the employer that neither the job ad nor the company research states, quoted briefly. Empty if none."
    )


class DraftComparison(BaseModel):
    """Which of two drafts of the same document is the better application for this ad."""
    better: Literal["A", "B"] = Field(..., description="The better draft")
    reason: str = Field(default="", description="One sentence")


COMPARE_PROMPT = (
    "Two drafts of the same {document_type} were written for the job ad below from the same candidate material. "
    "Both have been checked for invented claims. Which one is the more convincing truthful application for this ad, "
    "for a human recruiter and for automatic (ATS) screening? Judge relevance to the ad's requirements, clarity and "
    "completeness of the candidate's relevant real experience. Answer A or B.\n\n"
    "**Job ad:**\n{job_description}\n\n**Draft A:**\n{draft_a}\n\n**Draft B:**\n{draft_b}"
)


# Appended by the writers to every critique-driven redraft instruction, after the preference filter
# (so the filter can never drop it): a light critic sometimes proposes adding what the job asks for
# even when the candidate does not have it.
NO_NEW_CLAIMS_RULE = (
    "Do NOT add any skill, tool, qualification or experience that is not in the candidate's original CV, notes, "
    "portfolio or the user's own requests, even if the job asks for it."
)


class CritiqueAgent:
    """Agent for evaluating CV and cover letter quality and providing improvement instructions."""
    
    def __init__(
        self,
        model: str = "openai:gpt-5-mini",
        temperature: float = 0.3,
        grounding_model: Optional[str] = None,
        grounding_temperature: Optional[float] = None,
    ):
        """
        Initialize the CritiqueAgent.

        Args:
            model: The LLM model identifier for the critic (scores, requirement checks, draft comparison)
            temperature: Temperature setting for the critic
            grounding_model: Model for the invented-claims check. A different model from the writer's
                does not share its blind spots; defaults to the critic's model.
            grounding_temperature: Temperature for the grounding model

        Returns:
            None
        """
        logger.info("Initializing CritiqueAgent...")
        
        self.llm = init_chat_model(model, temperature=temperature, **local_model_kwargs(model))
        if grounding_model and grounding_model != model:
            self.grounding_llm = init_chat_model(grounding_model, temperature=grounding_temperature,
                                                 **local_model_kwargs(grounding_model))
        else:
            self.grounding_llm = self.llm
        logger.debug(f"CritiqueAgent LLM initialized - model: {model}, temperature: {temperature}, grounding model: {grounding_model or model}")
    
    def _format_job_description(self, job_description_info: Optional[dict]) -> str:
        """
        Format job description information for critique evaluation.

        Args:
            job_description_info: Dictionary produced by `JobDescriptionAgent`.

        Returns:
            str: Formatted job description information.
        """
        return format_job_description(job_description_info)
    
    def _format_company_info(self, company_info: Optional[dict]) -> str:
        """
        Format company information for critique evaluation.

        Args:
            company_info: Dictionary containing the company description and work context.

        Returns:
            str: Formatted company information.
        """
        if not company_info:
            return "No additional company information available."
        
        info_parts = []
        if company_info.get("company_description"):
            info_parts.append(f"Description: {company_info['company_description']}")
        info_parts.append(describe_work_context(company_info, company_info.get("work_preference")))
        
        return "\n".join(info_parts) if info_parts else "No additional company information available."
    
    def _format_candidate_info(self, candidate_text: Optional[dict]) -> str:
        """
        Format original candidate information for context.

        Args:
            candidate_text: Dictionary containing original CV and cover letter.

        Returns:
            str: Formatted candidate information.
        """
        return format_candidate_material(candidate_text)

    def _critique(self, document_type: str, document_content: str, job_description_info, company_info, candidate_text) -> CritiqueResponse:
        prompt = ChatPromptTemplate.from_messages([
            ("system", CRITIQUE_SYSTEM_PROMPT),
            ("human", CRITIQUE_HUMAN_PROMPT)
        ])
        chain = prompt | self.llm.with_structured_output(CritiqueResponse)
        llm_input = {
            "document_type": document_type,
            "document_content": document_content,
            "job_description": self._format_job_description(job_description_info),
            "company_info": self._format_company_info(company_info),
            "candidate_info": self._format_candidate_info(candidate_text),
            "requirements": format_requirements(job_description_info),
        }
        logger.info(f"Calling LLM for {document_type} critique...")
        # Retried: a parse error or rate limit must not end the session and lose the user's inputs.
        response = chain.with_retry(stop_after_attempt=3).invoke(llm_input)
        logger.info(f"{document_type} critique completed - quality_score: {response.quality_score}")
        return response
    
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
        return self._critique("CV", cv_content, job_description_info, company_info, candidate_text)
    
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
        # Include CV content in document content for context
        document_content = f"**Associated CV:**\n{cv_content}\n\n**Cover Letter:**\n{cover_letter_content}" if cv_content else cover_letter_content
        return self._critique("Cover Letter", document_content, job_description_info, company_info, candidate_text)
    
    def check_grounding(
        self,
        document_type: str,
        document_content: str,
        candidate_text: dict,
        employer_info: Optional[str] = None,
    ) -> Optional[list[str]]:
        """
        Ask the model, as a separate small task, which claims in the draft its sources do not
        support: claims about the candidate against the candidate's own material, and statements
        about the employer against the job ad and the company research. Works for any profession:
        nothing here knows what a skill looks like.

        Returns:
            list[str] | None: Unsupported claims, candidate claims first (empty when all are
            supported). None when the check could not run: an unchecked draft is not a clean one.
        """
        try:
            chain = ChatPromptTemplate.from_messages([("human", GROUNDING_PROMPT)]) | self.grounding_llm.with_structured_output(GroundingResponse)
            response = chain.with_retry(stop_after_attempt=2).invoke({
                "document_type": document_type,
                "document_content": document_content,
                "candidate_info": self._format_candidate_info(candidate_text),
                "employer_info": employer_info or "(no job ad or company research available)",
            })
        except Exception as e:  # a failed check must not end the session
            logger.warning(f"Grounding check failed ({type(e).__name__}: {e}) - the draft is marked as unchecked")
            return None
        claims = [c.strip() for c in (response.unsupported_claims or []) if _clean_text(c)]
        employer = [c.strip() for c in (getattr(response, "unsupported_employer_claims", None) or []) if _clean_text(c)]
        logger.info(f"Grounding check: {len(claims)} unsupported claim(s) about the candidate and "
                    f"{len(employer)} about the employer in the {document_type}")
        return claims + [f"{EMPLOYER_CLAIM_PREFIX}{c}" for c in employer if c not in claims]

    def compare_drafts(self, document_type: str, draft_a: str, draft_b: str, job_description_info: Optional[dict]) -> Optional[str]:
        """'A' or 'B': the better of two equally honest drafts, judged side by side (None if the call fails)."""
        try:
            chain = ChatPromptTemplate.from_messages([("human", COMPARE_PROMPT)]) | self.llm.with_structured_output(DraftComparison)
            response = chain.with_retry(stop_after_attempt=2).invoke({
                "document_type": document_type,
                "job_description": self._format_job_description(job_description_info),
                "draft_a": draft_a,
                "draft_b": draft_b,
            })
        except Exception as e:
            logger.warning(f"Draft comparison failed ({type(e).__name__}: {e}) - falling back to the scores")
            return None
        logger.info(f"Draft comparison: {response.better} is better ({response.reason})")
        return response.better

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

        A draft goes back while it has a real defect: an unsupported claim (from the separate
        grounding check), a critical issue, or a requirement of the ad that the candidate meets but
        the draft does not show. The critic's 0-100 score is recorded and shown, but it no longer
        decides: two independent scores from a light model are too noisy for that. A draft identical
        to the last one of this cycle is not critiqued or stored again.
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
                f"{kind}_claims_to_remove": None,
                "messages": state.get("messages", []) + [{
                    "role": "assistant",
                    "content": f"The {label.lower()} did not change in the last pass, so iteration {history[-1].get('iteration')} stands.",
                }],
            }

        candidate_text = candidate_material(state)
        company_info = with_preference(state.get("company_info"), state.get("work_preference"))
        if kind == "cv":
            result = self.critique_cv(
                cv_content=document,
                job_description_info=state.get("job_description_info"),
                company_info=company_info,
                candidate_text=candidate_text
            )
        else:
            result = self.critique_cover_letter(
                cover_letter_content=document,
                cv_content=state.get("generated_cv"),  # Include CV for context
                job_description_info=state.get("job_description_info"),
                company_info=company_info,
                candidate_text=candidate_text
            )

        instructions = _clean_text(result.improvement_instructions)
        critical_issues = _clean_text(result.critical_issues)

        if critical_issues and not instructions:
            # A critical issue with no instruction left the writer nothing to do, so the problem stayed.
            instructions = f"Fix these critical issues:\n{critical_issues}"

        requirements = [r.model_dump() for r in (getattr(result, "requirements", None) or [])]
        defects = [r for r in requirements if r["status"] == "missing"]
        gaps = [r["requirement"] for r in requirements if r["status"] == "gap"]
        if defects and not instructions:
            instructions = "Show these requirements of the ad, which the candidate's material shows they meet:\n" + "\n".join(
                f"- {r['requirement']}" + (f" (candidate's material: \"{r['quote']}\")" if r.get("quote") else "") for r in defects
            )

        unsupported = self.check_grounding(
            label if kind == "cv" else "cover letter", document, candidate_text,
            employer_info=employer_evidence(state),
        )
        # A check that could not run is recorded as such: the draft is ranked below checked ones
        # and, if it is still the final draft, the user is told it was not checked.
        grounding_checked = unsupported is not None
        unsupported = unsupported or []
        if unsupported:
            listed = "\n".join(f"- {claim}" for claim in unsupported)
            critical_issues = _join(critical_issues, f"Claims the sources (the candidate's own material; for the employer, the job ad and company research) do not support:\n{listed}")

        quality_score = result.quality_score
        # The critic's own suggestions, and separately the claims to remove: the writers filter only
        # the suggestions against the user's earlier requests, never the removals (see removal_instructions).
        critique_dict = {
            "quality_score": quality_score,
            "content_quality_feedback": result.content_quality_feedback,
            "ats_compatibility_feedback": result.ats_compatibility_feedback,
            "job_alignment_feedback": result.job_alignment_feedback,
            "improvement_instructions": instructions,
            "critical_issues": critical_issues,
            "unsupported_claims": unsupported,
            "grounding_checked": grounding_checked,
            "requirements": requirements,
            "defects": len(defects),
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
            "unsupported_claims": unsupported,
            "grounding_checked": grounding_checked,
            "defects": len(defects),
        }]
        logger.info(f"Added {label.lower()} iteration {iteration} to history with score {quality_score}, "
                    f"{len(defects)} requirement(s) not shown, {len(unsupported)} unsupported claim(s)")

        debug_content = f"{label.upper()} CRITIQUE RESULTS:\n"
        debug_content += "-" * 80 + "\n"
        debug_content += f"Quality Score: {quality_score}/100\n\n"
        debug_content += f"Content Quality Feedback:\n{result.content_quality_feedback}\n\n"
        debug_content += f"ATS Compatibility Feedback:\n{result.ats_compatibility_feedback}\n\n"
        debug_content += f"Job Alignment Feedback:\n{result.job_alignment_feedback}\n\n"
        if requirements:
            debug_content += "Requirements:\n" + "\n".join(f"- [{r['status']}] {r['requirement']}" for r in requirements) + "\n\n"
        if critical_issues:
            debug_content += f"Critical Issues:\n{critical_issues}\n\n"
        debug_content += f"Improvement Instructions:\n{instructions}\n\n"
        if unsupported:
            debug_content += "Claims to remove:\n" + "\n".join(f"- {c}" for c in unsupported) + "\n\n"
        debug_content += "-" * 80 + "\n"
        write_to_debug(debug_content, f"{label.upper()} CRITIQUE DEBUG INFO")

        # Redraft while there is a real defect and something for the writer to do.
        has_critical_issues = bool(critical_issues)
        has_defect = bool(unsupported) or has_critical_issues or bool(defects)
        should_refine = has_defect and bool(instructions or unsupported)

        state_updates[f"{kind}_needs_refinement"] = should_refine
        state_updates[f"{kind}_needs_critique"] = False
        state_updates[f"{kind}_critique_improvement_instructions"] = instructions if should_refine else None
        state_updates[f"{kind}_claims_to_remove"] = unsupported if should_refine else None
        if should_refine:
            logger.info(f"{label} critique found defects - unsupported claims: {len(unsupported)}, requirements not shown: {len(defects)}, critical issues: {has_critical_issues}; marking for refinement")
        else:
            logger.info(f"{label} critique found no defect the writer can fix - no redraft (score {quality_score})")

        # Keep the previous draft's defect count next to the new one, so the router can tell whether
        # the last redraft made progress. The first draft of a new cycle has no previous draft.
        previous = history[-1] if len(history) > cycle_start else None
        state_updates[f"{kind}_previous_defect_count"] = (previous.get("defects") if previous else None)

        # Always show critique feedback to user for transparency
        critique_message = f"**{'CV' if kind == 'cv' else 'Cover Letter'} Quality Assessment:**\n"
        critique_message += f"Quality Score: {quality_score}/100\n\n"
        critique_message += f"**Content Quality:**\n{result.content_quality_feedback}\n\n"
        critique_message += f"**ATS Compatibility:**\n{result.ats_compatibility_feedback}\n\n"
        critique_message += f"**Job Alignment:**\n{result.job_alignment_feedback}\n\n"
        if gaps:
            critique_message += "**What your material does not show for this job (left out, not faked):**\n" + "\n".join(f"- {g}" for g in gaps) + "\n\n"
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


EMPLOYER_CLAIM_PREFIX = "About the employer: "


def format_job_description(job_description_info: Optional[dict]) -> str:
    """The extracted job ad fields and full text, for the critic, the writers and the comparison."""
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


def format_requirements(job_description_info: Optional[dict]) -> str:
    """The ad's must-haves and nice-to-haves as the job-ad step extracted them, one per line."""
    info = job_description_info or {}
    lines = [f"- (must-have) {r}" for r in info.get("must_haves") or [] if str(r).strip()]
    lines += [f"- (nice-to-have) {r}" for r in info.get("nice_to_haves") or [] if str(r).strip()]
    return "\n".join(lines) if lines else "(none were extracted; use the main requirements of the job description)"


def format_candidate_material(candidate_text: Optional[dict]) -> str:
    """The candidate's own material, as the critic, the grounding check and the removal pass see it."""
    if not candidate_text:
        return "No original candidate information available."

    parts = []
    if candidate_text.get("cv"):
        # Full text: the critique checks the draft against it for invented skills, so a truncated
        # CV makes real skills look invented.
        parts.append(f"Original CV:\n{candidate_text['cv']}")
    else:
        parts.append("Original CV: Not provided")

    if candidate_text.get("cover_letter"):
        parts.append(f"Original Cover Letter:\n{candidate_text['cover_letter']}")
    else:
        parts.append("Original Cover Letter: Not provided")

    if candidate_text.get("notes"):
        # Free-form notes are the candidate's own statements, like the CV.
        parts.append(f"Candidate's Own Notes:\n{candidate_text['notes']}")

    if candidate_text.get("portfolio"):
        # Without it the critic calls real portfolio projects hallucinations and asks to remove them.
        parts.append(f"Candidate Portfolio:\n{candidate_text['portfolio']}")

    if candidate_text.get("work_preference"):
        parts.append(f"Candidate's Own Work-Arrangement Preference: {candidate_text['work_preference']}")

    if candidate_text.get("user_requests"):
        # What the user said in this session is the user's own statement, not an invention.
        parts.append(f"User's Own Requests and Statements in This Session:\n{candidate_text['user_requests']}")

    return "\n".join(parts)


def candidate_material(state) -> dict:
    """
    Everything that counts as the candidate's own material in a session: the uploaded CV, earlier
    cover letter and notes, the portfolio they supplied, their stated work preference, and every
    message the user wrote in the chat (requests and statements about themselves).
    """
    return {
        **(state.get("candidate_text") or {}),
        "portfolio": state.get("portfolio_text"),
        "user_requests": _user_requests(state.get("messages")),
        "work_preference": state.get("work_preference"),
    }


def employer_evidence(state) -> str:
    """The job ad and the company research: the only evidence for statements about the employer."""
    info = state.get("job_description_info") or {}
    company = (state.get("company_info") or {}).get("company_description") or "(no company research)"
    return f"Job ad:\n{info.get('job_description') or '(no job ad text)'}\n\nCompany research:\n{company}"


# Answers a light model gives when it means "nothing": the prompt asks for an empty string, and it
# sometimes returns the two quote characters themselves or a phrase.
_EMPTY_ANSWERS = {"", "none", "n/a", "na", "null", "nothing", "no improvements needed", "no changes needed",
                  "no critical issues", "no issues"}


def _clean_text(text: str | None) -> str:
    """Critique text with quote-only and 'none'-style answers turned into an empty string."""
    stripped = (text or "").strip()
    unquoted = stripped.strip("\"'`“”‘’ ").rstrip(".").strip().lower()
    return "" if unquoted in _EMPTY_ANSWERS else stripped


def _user_requests(messages) -> str:
    """The user's own messages in this session (dicts or LangChain messages), one per line."""
    lines = []
    for msg in messages or []:
        if isinstance(msg, dict):
            role, content = msg.get("role"), msg.get("content")
        else:
            role, content = {"human": "user"}.get(getattr(msg, "type", ""), ""), getattr(msg, "content", "")
        if role == "user" and isinstance(content, str) and content.strip():
            lines.append(f"- {content.strip()}")
    return "\n".join(lines)


def _join(first: str, second: str) -> str:
    return f"{first}\n{second}" if first else second
