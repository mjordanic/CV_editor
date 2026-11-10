from typing import Optional
import logging

from pydantic import BaseModel, Field

from agents.document_writer import TRUTHFULNESS_PROTOCOL, USER_STATEMENTS_LABEL, DocumentWriterAgent

logger = logging.getLogger(__name__)


COVER_LETTER_GENERATION_SYSTEM_PROMPT = (
    "You are an expert cover letter writer specializing in creating compelling, personalized cover letters "
    "that match job descriptions and company cultures. Your PRIMARY GOAL is a specific, TRUTHFUL letter for this job "
    "that convinces a human reader the candidate fits it, using the candidate's real experience and the ad's own terms. "
    "IMPORTANT: You must provide two separate outputs: 1) A clean cover letter document with NO explanations or notes, "
    "and 2) Separate notes and explanations for the user about what you did and why. "
    "CRITICAL PRESERVATION RULE: When modification instructions are provided and an existing cover letter is provided, "
    "you MUST preserve ALL existing content, structure, and formatting that is NOT explicitly mentioned in the modification instructions. "
    "Only make the specific changes requested - do NOT rewrite, restructure, or add content unless explicitly requested "
    "(or unless creative liberty is explicitly granted in the instructions)."
    + TRUTHFULNESS_PROTOCOL
)

COVER_LETTER_GENERATION_HUMAN_PROMPT = (
    "Create a professional, tailored cover letter based on the following information:\n\n"
    "**Existing Cover Letter (Base Document):**\n{base_document}\n{base_document_note}\n\n"
    "**Tailored CV for This Application (keep the letter consistent with it):**\n{tailored_cv}\n\n"
    "**Candidate's Original CV (source facts: the candidate's own, as uploaded):**\n{original_cv}\n\n"
    "**Candidate's Earlier Cover Letter (source facts only; written for a different job):**\n{original_cover_letter}\n\n"
    "**Candidate's Own Notes (free-form, possibly rough; facts here are the candidate's own, like the CV):**\n{candidate_notes}\n\n"
    "**Candidate's Portfolio (projects and past work the candidate supplied; their own facts):**\n{relevant_experience_context}\n\n"
    + USER_STATEMENTS_LABEL + "\n{user_statements}\n\n"
    "**Job Description:**\n{job_description}\n\n"
    "**Company Information:**\n{company_info}\n\n"
    "**Modification Instructions:**\n{modification_instructions}\n\n"
    "**Previous Modification Instructions (preserve these):**\n{previous_modification_instructions}\n\n"
    "**Output Language:** {output_language}\n\n"
    "CRITICAL: When modification instructions are provided, you MUST:\n"
    "1. **PRESERVE the existing cover letter structure and content** - Keep all paragraphs, formatting, and information that is NOT mentioned in the modification instructions (UNLESS creative liberty is explicitly granted in the instructions)\n"
    "2. **ONLY make the specific changes requested** - Do NOT add, remove, or modify anything that is not explicitly mentioned in the modification instructions\n"
    "3. **Do NOT rewrite or restructure** - Only apply the exact changes requested, preserving everything else as-is (UNLESS creative liberty is explicitly granted)\n"
    "4. **Do NOT hallucinate or add new content** unless explicitly requested in the modification instructions\n"
    "5. **Maintain consistency** - Keep the same tone, style, and format as the existing cover letter (UNLESS creative liberty is explicitly granted)\n\n"
    "If NO modification instructions are provided (empty), then you can create or enhance the cover letter based on the job description and company information.\n\n"
    "General guidelines (apply only when no specific modification instructions are provided, or creative liberty is granted):\n"
    "- Tailor the cover letter to this specific job and employer\n"
    "- Address the hiring manager or use a professional greeting\n"
    "- Start with a strong opening that captures attention\n"
    "- Highlight the candidate's real skills and experiences that match the job requirements\n"
    "- Show genuine interest in the role and company, using only reasons the candidate's material or the user gives, "
    "or facts the job description and company information state\n"
    "- Reference specific details from the job description and company information\n"
    "- Explain the fit with the candidate's real experience; do not dwell on or apologise for requirements they lack\n"
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


class CoverLetterWriterAgent(DocumentWriterAgent):
    """Agent for generating tailored cover letters based on job descriptions and company information."""

    kind = "cover_letter"
    label = "cover letter"
    file_stem = "generated_cover_letter"
    content_field = "cover_letter_content"
    response_model = CoverLetterGenerationResponse
    system_prompt = COVER_LETTER_GENERATION_SYSTEM_PROMPT
    human_prompt = COVER_LETTER_GENERATION_HUMAN_PROMPT

    def _original_document(self, candidate_text: dict) -> Optional[str]:
        return (candidate_text or {}).get("cover_letter")

    def _no_document_text(self) -> str:
        return "No existing cover letter provided. Create a new cover letter."

    def _base_document_note(self, mode: str, is_original: bool) -> str:
        if not is_original:
            return ""
        if mode == "fresh":
            return ("(This is the candidate's earlier cover letter, written for a different job and employer. "
                    "Use it only as a source of facts about the candidate and of their voice.)")
        if mode == "request":
            return ("(This is the candidate's earlier cover letter, written for a different job and employer. "
                    "Apply the user's request as asked, and update the employer, role and addressee to this job.)")
        return ""

    def _request_on_original_note(self) -> str:
        # The earlier letter names another employer, role and addressee: a user who asks for a
        # change to it on the first letter for this job wants it sent to this employer. The wish
        # itself is applied as asked (D2).
        return ("**The existing cover letter was written for a different job and employer.** Apply the user's "
                "request above exactly as asked, and also update the employer, the role and the addressee (and any "
                "statement about the candidate's current situation that is about that other job) to fit this job. "
                "Take the new employer, role and addressee only from the job description and company information.")

    def _fresh_draft_instructions(self, has_document: bool) -> str:
        # The first letter for a new job used to be an edit of the old letter for another employer,
        # which kept its stale framing (old company, old role, outdated "currently" sentences).
        source = ("The 'Existing Cover Letter' was written for a different job and employer: use it only as a source of "
                  "facts about the candidate and of their voice. Do NOT keep its employer, role, addressee, structure, or "
                  "any statement about the candidate's current situation that the candidate's other material does not "
                  "confirm. " if has_document else "")
        return (
            "**CREATIVE LIBERTY GRANTED (A NEW LETTER FOR THIS JOB):**\n"
            "Write a new cover letter tailored to this job ad and employer. "
            + source +
            "Choose the structure and wording that best present the candidate's real fit. "
            "However, do NOT hallucinate facts - only use the factual information in the candidate's CV, earlier cover letter, "
            "own notes, portfolio and what the user said in this chat, and about the employer only what the job description "
            "and company information state."
        )

    def _extra_prompt_inputs(self, state) -> dict:
        return {"tailored_cv": state.get("generated_cv") or "Not written yet."}

    def save_cover_letter(self, cover_letter_text: str, filename: str = "generated_cover_letter.txt") -> str:
        """Save cover letter text to the output folder."""
        return self.save_document(cover_letter_text, filename)

    def save_notes(self, notes: str, filename: str = "generated_cover_letter_notes.txt") -> str:
        """Save the writer's notes to the output folder."""
        return self.save_document(notes, filename)
