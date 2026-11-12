from typing import Optional
import logging

from pydantic import BaseModel, Field

from agents.document_writer import TRUTHFULNESS_PROTOCOL, USER_STATEMENTS_LABEL, DocumentWriterAgent

logger = logging.getLogger(__name__)


CV_GENERATION_SYSTEM_PROMPT = (
    "You are an expert CV writer specializing in creating tailored, professional CVs that match job descriptions. "
    "Your PRIMARY GOAL is the most convincing TRUTHFUL match to this job ad for a human recruiter. "
    "Use the ad's own terms wherever the candidate's material supports them, so automatic (ATS) screening finds them too, "
    "and keep the CV ATS-friendly (standard headings, plain structure); ATS screening is a means, not the goal. "
    "Your CVs are well-structured and highlight the candidate's relevant skills and experiences. "
    "You create compelling CVs that stand out while remaining truthful and accurate. "
    "CRITICAL: You MUST ONLY generate a CV document. You MUST NOT generate a cover letter or any other document, even if instructions mention it. "
    "If instructions mention moving content to a cover letter, ignore that part and focus only on improving the CV. "
    "IMPORTANT: You must provide two separate outputs: 1) A clean CV document with NO explanations or notes, "
    "and 2) Separate notes and explanations for the user about what you did and why. "
    "CRITICAL PRESERVATION RULE: When modification instructions are provided and an existing CV is provided, "
    "you MUST preserve ALL existing content, structure, and formatting that is NOT explicitly mentioned in the modification instructions. "
    "Only make the specific changes requested - do NOT rewrite, restructure, or add content unless explicitly requested "
    "(or unless creative liberty is explicitly granted in the instructions)."
    + TRUTHFULNESS_PROTOCOL
)

CV_GENERATION_HUMAN_PROMPT = (
    "Create a professional, tailored CV based on the following information:\n\n"
    "**Existing CV (Base Document):**\n{base_document}\n{base_document_note}\n\n"
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
    "1. **PRESERVE the existing CV structure and content** - Keep all sections, formatting, and information that is NOT mentioned in the modification instructions\n"
    "2. **ONLY make the specific changes requested** - Do NOT add, remove, or modify anything that is not explicitly mentioned in the modification instructions\n"
    "3. **Do NOT rewrite or restructure** - Only apply the exact changes requested, preserving everything else as-is (UNLESS creative liberty is explicitly granted in the instructions)\n"
    "4. **Do NOT hallucinate or add new content** unless explicitly requested in the modification instructions\n"
    "5. **Maintain consistency** - Keep the same tone, style, and format as the existing CV (UNLESS creative liberty is explicitly granted)\n\n"
    "If NO modification instructions are provided (empty), then you can create or enhance the CV based on the job description and company information.\n\n"
    "General guidelines (apply only when no specific modification instructions are provided, or creative liberty is granted):\n"
    "- Tailor the CV to the job requirements: show first the requirements the candidate really meets\n"
    "- Use the ad's own wording for those requirements, so ATS screening finds them; never add an ad term the candidate's material does not support\n"
    "- Highlight relevant skills, experiences, and achievements\n"
    "- Use clear, professional language\n"
    "- Structure the CV in a standard format (Contact Info, Professional Summary, Experience, Education, Skills)\n"
    "- Follow the usual CV conventions of the job's country (see the job's location) for length and personal details, "
    "unless the candidate's own CV clearly follows another convention\n"
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

NO_CV_TEXT = (
    "No existing CV provided. Build the CV only from the facts in the candidate's notes, the portfolio and what the user "
    "said in this chat; leave out anything they do not state, and use placeholders such as [EMAIL] for missing contact details."
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


class CVWriterAgent(DocumentWriterAgent):
    """Agent for generating tailored CVs based on job descriptions and company information."""

    kind = "cv"
    label = "CV"
    file_stem = "generated_CV"
    content_field = "cv_content"
    response_model = CVGenerationResponse
    system_prompt = CV_GENERATION_SYSTEM_PROMPT
    human_prompt = CV_GENERATION_HUMAN_PROMPT

    def _original_document(self, candidate_text: dict) -> Optional[str]:
        return (candidate_text or {}).get("cv")

    def _no_document_text(self) -> str:
        return NO_CV_TEXT

    def _fresh_draft_instructions(self, has_document: bool) -> str:
        return (
            "**CREATIVE LIBERTY GRANTED:**\n"
            "Use the provided 'Existing CV' primarily as a source of information about the candidate's history, skills, and experience. "
            "You are NOT bound to preserve its structure, formatting, or exact phrasing. "
            "Your goal is to create the BEST possible CV tailored to the Job Description. "
            "Feel free to reorder sections, rewrite bullet points for impact, and choose the most professional format. "
            "However, do NOT hallucinate facts - only use the factual information provided in the Existing CV, "
            "the candidate's earlier cover letter, own notes, portfolio and what the user said in this chat."
        )
