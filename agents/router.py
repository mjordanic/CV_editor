import re
from typing import Literal, Optional
import logging
from langchain.chat_models import init_chat_model
from config.config_loader import local_model_kwargs
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from debug_utils import log_messages, get_logged_message_count

from agents.notices import NOTICES_KEY, pending_notices, with_notices
from agents.work_context import describe as describe_work_context, with_preference

logger = logging.getLogger(__name__)


ROUTER_SYSTEM_PROMPT = (
    "You are a helpful assistant that routes user requests for CV and cover letter generation. "
    "Your role is to interact with other agents for CV and cover letter generation, and with the user, understand what the user wants to generate, summarize "
    "their request to a short actionable message and collect feedback from the user "
    "after documents are created. You should be friendly and professional.\n\n"
    "Based on the conversation history, job description, company information, and the status of generated documents, "
    "you need to:\n"
    "1. Determine what message to show the user (if any)\n"
    "2. Decide what the next action should be (from the available actions listed below)\n"
    "3. Extract and summarize feedback for modifications, SEPARATING content feedback from style feedback:\n"
    "   - CONTENT MODIFICATIONS (CV / Cover Letter text):\n"
    "       * Extract instructions about WHAT to include/exclude/change in the text content.\n"
    "       * Examples: 'add research stay at Berkeley in 2021', 'add MATLAB to skills', 'remove section X', 'emphasize Y'.\n"
    "       * Store in 'cv_content_feedback' or 'cover_letter_content_feedback' fields (separate fields for CV and cover letter).\n"
    "   - PDF APPEARANCE MODIFICATIONS (style/layout/colors/fonts ONLY):\n"
    "       * Extract instructions about HOW the PDF should LOOK (visual appearance only).\n"
    "       * Examples: 'titles in dark blue color and bolded', 'make it more compact', 'use minimal spacing', 'change font to Arial'.\n"
    "       * Store in 'cv_pdf_style' or 'cover_letter_pdf_style' fields: only the field of the document the user names "
    "(both fields only if the user asks for both documents).\n"
    "   CRITICAL: A single user message can contain feedback for BOTH CV and cover letter (content + style for each). Extract them separately.\n"
    "   - If content feedback exists for a document, route to the appropriate 'draft_*' action for that document.\n"
    "   - If content feedback exists for BOTH documents, check the conversation history to determine which one the user wants to generate first, or route to the one that makes most sense based on context.\n"
    "   - IMPORTANT: Check the conversation history to see if the user requested both documents. If one document has been generated but the other hasn't, and the user previously requested both, route to generate the missing document.\n"
    "   - If ONLY style feedback exists (and document already exists), route to 'update_*_pdf_style'.\n"
    "   - If ONLY style feedback exists (and document does NOT exist yet), route to 'draft_cv' or 'draft_cover_letter' first.\n"
    "   - Style feedback will be automatically applied when the PDF is generated (after content is finalized).\n"
    "The feedback should be short, clear, and actionable. "
    "Do NOT include your own questions, clarifications, or suggestions. Leave empty if no actionable feedback provided.\n\n"
    "Available actions (you can ONLY choose from actions listed in the context below):\n"
    "{available_actions}\n\n"
    "If documents have been generated, the user might want to:\n"
    "- Request content modifications (provide feedback about the text)\n"
    "- Request PDF appearance changes (style/layout/colors/fonts only)\n"
    "- Both content and style changes in a single message\n"
    "- Generate the other document\n"
    "- Exit if satisfied\n\n"
    "If no documents have been generated yet, ask the user what they'd like to generate first.\n\n"
    "WHAT THIS PROGRAM CAN AND CANNOT DO:\n"
    "- It can ONLY write and revise the tailored CV and cover letter for this job ad, and change how their PDFs look.\n"
    "- It cannot send emails or applications, contact employers, book interviews, search the web, look up facts "
    "(weather, news, salaries), or write any other kind of document (emails, messages, reference letters).\n"
    "- If the user asks for something it cannot do, or writes something unrelated, say plainly in one short "
    "sentence that this program cannot do that, then ask what they would like to change in the CV or cover letter. "
    "Do not give a reason that suggests it could do it otherwise, and never offer anything that is not one of the "
    "available actions.\n"
    "- If the user changes their mind within a message, follow their LAST stated wish.\n\n"
    "WHAT THE WRITERS DID WITH THE USER'S REQUEST:\n"
    "- The context may include the writers' notes on the user's latest request. If they say that any part of the "
    "request was not applied, or was applied differently from what the user asked, say so plainly in your next "
    "message to the user: which part, what was done instead, and why, in one or two sentences. Never say a change "
    "was made if the notes say it was not.\n\n"
    "WORK ARRANGEMENT:\n"
    "- If the user says in their latest message whether they want remote, hybrid or on-site work (or that they have "
    "no preference), put it in 'work_preference'. Never guess it from anything else.\n"
    "- If the company information shows a 'Preference conflict', tell the user about it plainly in your next "
    "message (once), in the user's words, without changing either side.\n"
    "- If the user has not stated a preference, treat them as open to the job's arrangement: never raise the "
    "arrangement, location or relocation as a problem or a reason not to apply."
)

ROUTER_HUMAN_PROMPT = (
    "You are routing a conversation about CV and cover letter generation. Here's the context:\n\n"
    "**Conversation History:**\n{messages_history}\n\n"
    "**Job Description Information:**\n{job_description_info}\n\n"
    "**Company Information:**\n{company_info}\n\n"
    "**Original Documents Uploaded by User:**\n{candidate_text}\n\n"
    "**Document Generation Status:**\n"
    "- CV Generated: {cv_generated}\n"
    "- Cover Letter Generated: {cover_letter_generated}\n\n"
    "IMPORTANT: Check the conversation history to see if the user requested both documents. If one document has been generated but the other hasn't, and the user previously requested both, route to generate the missing document.\n\n"
    "**Available Feedback for Modifications (if any):**\n{feedback_context}\n\n"
    "**Writers' Notes on the User's Latest Request (not yet shown to the user):**\n{writer_notes}\n\n"
    "Based on this context, determine:\n"
    "1. What message should be shown to the user (if the conversation needs user input)\n"
    "2. What the next action should be (you can ONLY choose from the available actions listed in the system prompt)\n"
    "3. Extract feedback for modification of documents, SEPARATING content from style AND CV from cover letter:\n"
    "   - CONTENT FEEDBACK (what to include/exclude/change in text):\n"
    "       * Extract instructions about WHAT to add/remove/modify in the CV or cover letter content.\n"
    "       * Store CV content feedback in 'cv_content_feedback' field.\n"
    "       * Store cover letter content feedback in 'cover_letter_content_feedback' field.\n"
    "       * CRITICAL: A single user message can contain feedback for BOTH CV and cover letter. Extract them separately.\n"
    "   - STYLE FEEDBACK (how the PDF should look):\n"
    "       * Extract instructions about visual appearance (colors, fonts, spacing, layout).\n"
    "       * Store CV PDF style in 'cv_pdf_style' field.\n"
    "       * Store cover letter PDF style in 'cover_letter_pdf_style' field.\n"
    "       * CRITICAL: A single user message can contain BOTH content and style feedback for BOTH documents. Extract them separately.\n"
    "   - Extract ONLY the actual feedback from the available feedback context. Do NOT include your own questions, clarifications, or suggestions.\n"
    "   - All feedback should be short, clear, and actionable. Leave empty if no actionable feedback provided.\n\n"
    "If this is the first interaction and no documents have been generated, ask the user what they'd like to generate.\n"
    "If documents have been generated, ask for feedback or if they want to generate the other document.\n"
    "If feedback exists and indicates refinement is needed, route to the appropriate generation action with the summarized feedback (if that action is available)."
)


_LANGCHAIN_TYPE_TO_ROLE = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}


# What the user has just been given when the router tries to repeat an action that is no longer
# offered (it was offered only until it ran), used to open the fallback question.
_FINISHED_ACTION_NOTE = {
    "draft_cv": "The CV draft is ready.",
    "draft_cover_letter": "The cover letter draft is ready.",
    "update_cv_pdf_style": "The CV PDF has been updated.",
    "update_cover_letter_pdf_style": "The cover letter PDF has been updated.",
}


def _message_role(msg) -> str:
    """Role of a message dict ('role' key) or LangChain message (whose role is its .type)."""
    if isinstance(msg, dict):
        return msg.get("role", "unknown")
    msg_type = getattr(msg, "type", None)
    return _LANGCHAIN_TYPE_TO_ROLE.get(msg_type, getattr(msg, "role", msg_type or "unknown"))


def _offered_actions(available_actions: str) -> set[str]:
    """Action names listed in the router prompt's available-actions block (lines "- 'name': ...")."""
    return set(re.findall(r"^- '([a-z_]+)'", available_actions, re.M))


def _latest_user_message(messages: list) -> str:
    """Text of the latest non-empty message if it is the user's, otherwise an empty string."""
    for msg in reversed(messages or []):
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if isinstance(content, str) and content.strip():
            return content.strip() if _message_role(msg) == "user" else ""
    return ""


STYLE_REQUESTS_HEADER = "Apply every style request below; where two conflict, the later one wins:"


def _accumulate_style(previous: str | None, new: str | None) -> str | None:
    """
    The stored PDF style with the new request added after the earlier ones.

    Every PDF is rendered from the stored style (a redraft starts again from the base template), so
    replacing it would drop the user's earlier style requests. A request already stored is not added twice.
    """
    new = (new or "").strip()
    if not new:
        return previous
    items = []
    for line in (previous or "").splitlines():
        line = line.strip()
        if line and line != STYLE_REQUESTS_HEADER:
            items.append(line.removeprefix("- "))
    if new not in items:
        items.append(new)
    if len(items) == 1:
        return items[0]
    return STYLE_REQUESTS_HEADER + "\n" + "\n".join(f"- {item}" for item in items)


_WRITER_NOTE_LABELS = {"cv": "CV", "cover_letter": "Cover letter"}


def _format_writer_notes(state) -> str:
    """The writers' notes on the user's latest request for each document, for the router prompt."""
    parts = [
        f"{label} writer: {notes.strip()}"
        for kind, label in _WRITER_NOTE_LABELS.items()
        if (notes := state.get(f"{kind}_request_notes")) and notes.strip()
    ]
    return "\n\n".join(parts) if parts else "None."


def _writer_notes_for_user(state) -> str:
    """The writers' notes on the user's latest request, worded for the user, or "" when there are none. Used when
    the router replaces the model's message with a fixed one, so the notes are still shown before they are cleared."""
    parts = [
        f"{label}: {notes.strip()}"
        for kind, label in _WRITER_NOTE_LABELS.items()
        if (notes := state.get(f"{kind}_request_notes")) and notes.strip()
    ]
    return ("About your last request:\n" + "\n".join(parts)) if parts else ""


def _new_draft_cycle(kind: str, state) -> dict:
    """
    State update that starts a new draft cycle for 'cv' or 'cover_letter'.

    Every router -> draft_* step is a new request (the critique loop redrafts without the router),
    so the critique-driven redraft budget starts again and finalize picks the best draft of this
    cycle only. Without this, an older draft with a higher score would replace the one that
    carries the user's change. A critique the last cycle left unapplied (its budget ran out) is
    cleared, so the writer applies the user's request instead of that critique.
    """
    return {
        f"{kind}_refinement_count": 0,
        f"{kind}_cycle_start": len(state.get(f"{kind}_history") or []),
        f"{kind}_needs_refinement": False,
        f"{kind}_critique_improvement_instructions": None,
        f"{kind}_claims_to_remove": None,
    }


class RouterResponse(BaseModel):
    """Response model for router decisions."""

    next_action: Literal[
        "draft_cv",
        "draft_cover_letter",
        "collect_user_input",
        "update_cv_pdf_style",
        "update_cover_letter_pdf_style",
        "exit",
    ] = Field(
        ...,
        description=(
            "The next action to take: draft_cv, draft_cover_letter, collect_user_input, "
            "update_cv_pdf_style, update_cover_letter_pdf_style, or exit. "
            "If both content and style feedback exist, route to 'draft_cv' or 'draft_cover_letter' "
            "(style will be applied automatically when PDF is generated)."
        ),
    )
    message_to_user: str = Field(
        ...,
        description=(
            "The message to show to the user. If empty, no message is needed "
            "(e.g., when routing directly to an action)."
        ),
    )
    cv_content_feedback: str = Field(
        default="",
        description=(
            "Content-related feedback for CV text modifications. "
            "Extract instructions about WHAT to include/exclude/change in the CV content. "
            "Examples: 'add research stay at Berkeley in 2021', 'add MATLAB to skills', 'remove section X'. "
            "Do NOT include cover letter content or style/appearance instructions here. "
            "Short, clear, actionable. Empty if no CV content feedback provided."
        ),
    )
    cover_letter_content_feedback: str = Field(
        default="",
        description=(
            "Content-related feedback for cover letter text modifications. "
            "Extract instructions about WHAT to include/exclude/change in the cover letter content. "
            "Examples: 'emphasize my research experience', 'add paragraph about motivation', 'remove mention of X'. "
            "Do NOT include CV content or style/appearance instructions here. "
            "Short, clear, actionable. Empty if no cover letter content feedback provided."
        ),
    )
    cv_pdf_style: str = Field(
        default="",
        description=(
            "PDF appearance/style feedback for CV (visual appearance ONLY, no content changes). "
            "Extract instructions about HOW the CV PDF should LOOK. "
            "Examples: 'titles in dark blue color and bolded', 'make it more compact', 'use minimal spacing'. "
            "Do NOT include content modification instructions here. "
            "Fill it ONLY when the user's message is about the CV PDF (or names both documents); if the user "
            "asks only about the cover letter, leave this empty. "
            "Empty if no CV PDF style feedback provided."
        ),
    )
    cover_letter_pdf_style: str = Field(
        default="",
        description=(
            "PDF appearance/style feedback for cover letter (visual appearance ONLY, no content changes). "
            "Extract instructions about HOW the cover letter PDF should LOOK. "
            "Examples: 'titles in dark blue color and bolded', 'make it more compact', 'use minimal spacing'. "
            "Do NOT include content modification instructions here. "
            "Fill it ONLY when the user's message is about the cover letter PDF (or names both documents); if the user "
            "asks only about the CV, leave this empty. "
            "Empty if no cover letter PDF style feedback provided."
        ),
    )
    needs_user_input: bool = Field(
        ...,
        description=(
            "Whether the router needs to wait for user input before proceeding. "
            "If True, show message_to_user and wait for response."
        ),
    )
    work_preference: Literal["", "remote", "hybrid", "on-site", "no preference"] = Field(
        default="",
        description=(
            "The user's own work-arrangement preference, ONLY if their latest message states it "
            "(e.g. 'I only want remote jobs' -> 'remote'). Empty otherwise; never inferred."
        ),
    )


class RouterAgent:
    """Agent for routing user requests and collecting feedback using LLM."""
    
    def __init__(
        self, 
        model: str = "openai:gpt-5-nano", 
        temperature: float = 0.0, 
        max_history_messages: int = 10
    ):
        """
        Initialize the router agent and its backing LLM client.

        Args:
            model: The LLM model identifier to use
            temperature: Temperature setting for the LLM
            max_history_messages: Number of recent messages to include in the prompt context (default: 10)
        """
        logger.info("Initializing RouterAgent...")
        
        self.llm = init_chat_model(model, temperature=temperature, **local_model_kwargs(model))
        self.max_history_messages = max_history_messages
        logger.debug(f"RouterAgent LLM initialized - model: {model}, temperature: {temperature}, max_history_messages: {max_history_messages}")
    
    def _format_messages_history(self, messages: list) -> str:
        """
        Format the recent conversation history for inclusion in the router prompt.

        Args:
            messages: Sequence of LangChain message objects or message dictionaries.

        Returns:
            str: Human-readable transcript limited to the most recent entries.
        """
        if not messages:
            return "No conversation history yet."
        
        formatted = []
        for msg in messages[-self.max_history_messages:]:  # Last N messages for context
            role = _message_role(msg)
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            
            if content:
                formatted.append(f"{role.title()}: {content}")
        
        return "\n".join(formatted) if formatted else "No conversation history yet."
    
    def _format_job_description_info(self, job_description_info: dict | None) -> str:
        """
        Format extracted job description information for the router prompt.

        Args:
            job_description_info: Dictionary produced by `JobDescriptionAgent`.

        Returns:
            str: Summary of the most relevant job description attributes.
        """
        if not job_description_info:
            return "No job description information available."
        
        parts = []
        if job_description_info.get("company_name"):
            parts.append(f"Company: {job_description_info['company_name']}")
        if job_description_info.get("job_title"):
            parts.append(f"Job Title: {job_description_info['job_title']}")
        if job_description_info.get("location"):
            parts.append(f"Location: {job_description_info['location']}")
        if job_description_info.get("candidate_minimal_requirements"):
            parts.append(f"Requirements: {job_description_info['candidate_minimal_requirements']}")
        if job_description_info.get("job_description"):
            parts.append(f"Full Description: {job_description_info['job_description'][:500]}...")  # Truncate for context
        
        return "\n".join(parts) if parts else "No job description information available."
    
    def _format_company_info(self, company_info: dict | None) -> str:
        """
        Format derived company information for the router prompt.

        Args:
            company_info: Dictionary returned by `SearchAgent` containing description/remote policy.

        Returns:
            str: Condensed company description suitable for the LLM prompt.
        """
        if not company_info:
            return "No company information available."
        
        parts = []
        if company_info.get("company_description"):
            desc = company_info["company_description"]
            if isinstance(desc, str):
                parts.append(f"Description: {desc[:500]}...")  # Truncate for context
            else:
                parts.append(f"Description: {str(desc)[:500]}...")
        parts.append(describe_work_context(company_info, company_info.get("work_preference")))
        
        return "\n".join(parts) if parts else "No company information available."
    
    def _format_candidate_text(self, candidate_text: dict | None) -> str:
        """
        Summarize any user-provided documents for LLM conditioning.

        Args:
            candidate_text: Dictionary of `cv`, `cover_letter` and `notes` strings from `DocumentReaderAgent`.

        Returns:
            str: Short preview of available candidate materials or fallback notice.
        """
        if not candidate_text:
            return "No original documents uploaded by the user."
        
        parts = []
        if candidate_text.get("cv"):
            cv_text = candidate_text["cv"]
            if cv_text:
                parts.append(f"Original CV: {cv_text[:300]}...")  # Truncate for context
            else:
                parts.append("Original CV: Not provided")
        else:
            parts.append("Original CV: Not provided")
        
        if candidate_text.get("cover_letter"):
            cl_text = candidate_text["cover_letter"]
            if cl_text:
                parts.append(f"Original Cover Letter: {cl_text[:300]}...")  # Truncate for context
            else:
                parts.append("Original Cover Letter: Not provided")
        else:
            parts.append("Original Cover Letter: Not provided")

        if candidate_text.get("notes"):
            parts.append(f"Candidate's Own Notes: {candidate_text['notes'][:300]}...")
        else:
            parts.append("Candidate's Own Notes: Not provided")
        
        return "\n".join(parts) if parts else "No original documents uploaded by the user."
    
    def _has_user_feedback(self, messages: list) -> bool:
        """
        Check if the most recent non-empty message is from the user.
        This determines if we should allow draft actions for user-initiated modifications.
        
        Args:
            messages: Conversation messages
            
        Returns:
            bool: True if the most recent non-empty message is from the user, False otherwise
        """
        if not messages:
            return False
        
        # Check messages in reverse order (most recent first)
        # Find the first non-empty message and check its role
        for msg in reversed(messages):
            role = _message_role(msg)
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            
            # Skip empty messages - continue to find the last non-empty message
            if not content or not content.strip():
                continue
            
            # Found the last non-empty message - check if it's from user
            # Return immediately regardless of role (don't continue checking earlier messages)
            return role == "user"
        
        # No non-empty messages found
        return False
    
    def _build_available_actions(
        self,
        cv_generated: bool,
        cover_letter_generated: bool,
        messages: list,
        cv_pdf_available: bool,
        cover_letter_pdf_available: bool,
    ) -> str:
        """
        Build the list of available actions dynamically based on what exists and the latest message.

        Redrafting an existing document and re-rendering an existing PDF are offered only when the
        latest message is the user's, so the router cannot repeat an action on its own: once the
        action has run, the latest message is the assistant's and the action is no longer offered.

        Args:
            cv_generated: Whether a CV has been generated already
            cover_letter_generated: Whether a cover letter has been generated already
            messages: Conversation messages (to check for user feedback)
            cv_pdf_available: Whether a CV PDF can be (re)rendered, i.e. the CV text exists
            cover_letter_pdf_available: Whether a cover letter PDF can be (re)rendered

        Returns:
            str: Formatted list of available actions
        """
        actions = []
        
        # Check if there's user feedback (last user message)
        has_user_feedback = self._has_user_feedback(messages)
        
        # draft_cv is available for the first generation, or to modify the CV after a new user message
        # (critique-driven redrafts never go through the router: the critique loop handles them)
        cv_available = has_user_feedback or not cv_generated

        # draft_cover_letter: same rule as draft_cv
        cover_letter_available = has_user_feedback or not cover_letter_generated
        
        # Build actions list
        if cv_available:
            actions.append("- 'draft_cv': Generate a new CV or modify an existing one (ONLY CV-related feedback should be passed)")
        
        if cover_letter_available:
            actions.append("- 'draft_cover_letter': Generate a new cover letter or modify an existing one (ONLY cover letter-related feedback should be passed)")
        
        # PDF appearance-only style update actions (available once the document's text exists, so a
        # PDF that failed to build can be rendered again)
        if cv_pdf_available and has_user_feedback:
            actions.append(
                "- 'update_cv_pdf_style': Regenerate the CV PDF using the SAME CV text but with updated visual style "
                "(layout/colors/fonts/spacing ONLY, NO content changes)."
            )
        if cover_letter_pdf_available and has_user_feedback:
            actions.append(
                "- 'update_cover_letter_pdf_style': Regenerate the cover letter PDF using the SAME cover letter text but "
                "with updated visual style (layout/colors/fonts/spacing ONLY, NO content changes)."
            )
        
        # Always available actions
        actions.append("- 'collect_user_input': Request user input (use this when you need to ask the user a question or get feedback)")
        actions.append("- 'exit': End the conversation")
        
        return "\n".join(actions)
    
    def _format_feedback_context(self, messages: list) -> str:
        """
        The user's latest message, if the latest non-empty message is the user's.

        Critique instructions are not shown here: the critique -> redraft loop runs without the
        router, which is reached only after it has finished.
        """
        user_feedback = _latest_user_message(messages)
        if not user_feedback:
            return "No feedback available."
        return f"**User Feedback:**\n{user_feedback}\n"
    
    def run(self, state):
        """
        Main method to route user requests and collect feedback using LLM.

        Args:
            state: The state dictionary containing messages, job_description_info, company_info, candidate_text, and generation status
            
        Returns:
            dict: Updated state with next action
        """
        logger.info("RouterAgent.run() called")
        
        # Extract state information
        messages = state.get("messages", [])
        job_description_info = state.get("job_description_info")
        company_info = with_preference(state.get("company_info"), state.get("work_preference"))
        candidate_text = state.get("candidate_text")
        generated_cv = state.get("generated_cv")
        generated_cover_letter = state.get("generated_cover_letter")
        
        logger.debug(f"State extracted - messages: {len(messages)}, cv_generated: {generated_cv is not None}, cover_letter_generated: {generated_cover_letter is not None}")
        
        # Log only new messages to debug file
        if messages:
            logged_count = get_logged_message_count()
            if logged_count < len(messages):
                # There are new messages to log
                new_count = log_messages(messages, start_index=logged_count)
                if new_count > 0:
                    logger.debug(f"Logged {new_count} new message(s) to debug file")
        
        # Format context for LLM
        messages_history = self._format_messages_history(messages)
        job_desc_formatted = self._format_job_description_info(job_description_info)
        company_info_formatted = self._format_company_info(company_info)
        candidate_text_formatted = self._format_candidate_text(candidate_text)
        # Appearance-only style updates need the document's text, not an existing PDF: when the last
        # PDF failed to build, a style request must still be able to render it again. (Requiring the
        # PDF made the router refuse the request and tell the user the PDF "has been updated".)
        cv_pdf_available = bool(state.get("cv_pdf_path") or generated_cv)
        cover_letter_pdf_available = bool(state.get("cover_letter_pdf_path") or generated_cover_letter)

        feedback_context = self._format_feedback_context(messages)
        
        # Build available actions dynamically from what exists and the latest message
        available_actions = self._build_available_actions(
            generated_cv is not None,
            generated_cover_letter is not None,
            messages,
            cv_pdf_available,
            cover_letter_pdf_available,
        )
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", ROUTER_SYSTEM_PROMPT),
            ("human", ROUTER_HUMAN_PROMPT)
        ])
        
        # Use structured output
        structured_llm = self.llm.with_structured_output(RouterResponse)
        chain = prompt | structured_llm
        
        # Prepare LLM input
        llm_input = {
            "available_actions": available_actions,
            "messages_history": messages_history,
            "job_description_info": job_desc_formatted,
            "company_info": company_info_formatted,
            "candidate_text": candidate_text_formatted,
            "cv_generated": "Yes" if generated_cv is not None else "No",
            "cover_letter_generated": "Yes" if generated_cover_letter is not None else "No",
            "feedback_context": feedback_context,
            "writer_notes": _format_writer_notes(state),
        }
        
        logger.debug(f"LLM input prepared - messages_history length: {len(messages_history)}, feedback_context length: {len(feedback_context)}")
        
        # Only bypass LLM if explicit content feedback exists for a document that hasn't been generated yet
        # This handles the case where user provided feedback for both documents in one message
        # Check for cover letter feedback when CV is done, or CV feedback when cover letter is done
        cover_letter_content_feedback = state.get("cover_letter_content_feedback")
        cv_content_feedback = state.get("cv_content_feedback")
        cv_pdf_path = state.get("cv_pdf_path")
        cover_letter_pdf_path = state.get("cover_letter_pdf_path")
        
        # If cover letter feedback exists and CV is finalized (PDF generated), route to cover letter
        # Only route if cover letter hasn't been generated yet (to prevent loops)
        if cover_letter_content_feedback and cv_pdf_path and not generated_cover_letter:
            logger.info("Cover letter content feedback exists and CV is finalized - routing to draft_cover_letter")
            return {
                "next": "draft_cover_letter",
                **_new_draft_cycle("cover_letter", state),
                "messages": messages + [
                    {
                        "role": "assistant",
                        "content": "Proceeding with cover letter generation using your previous feedback.",
                    }
                ],
            }
        
        # If CV feedback exists and cover letter is finalized (PDF generated), route to CV
        # Only route if CV hasn't been generated yet (to prevent loops)
        if cv_content_feedback and cover_letter_pdf_path and not generated_cv:
            logger.info("CV content feedback exists and cover letter is finalized - routing to draft_cv")
            return {
                "next": "draft_cv",
                **_new_draft_cycle("cv", state),
                "messages": messages + [
                    {
                        "role": "assistant",
                        "content": "Proceeding with CV generation using your previous feedback.",
                    }
                ],
            }

        # One user message with changes for both existing documents: the router redrafted one and
        # kept the other's request (feedback for an existing document is stored only for this, and
        # finalize clears it). Once the first is done (the latest message is no longer the user's),
        # redraft the other with its request before the user is asked anything.
        if not self._has_user_feedback(messages):
            for kind, generated in (("cover_letter", generated_cover_letter), ("cv", generated_cv)):
                pending = state.get(f"{kind}_content_feedback")
                if pending and pending.strip() and generated:
                    label = "cover letter" if kind == "cover_letter" else "CV"
                    logger.info(f"Pending {label} content feedback from the user's last message - routing to draft_{kind}")
                    return {
                        "next": f"draft_{kind}",
                        **_new_draft_cycle(kind, state),
                        "messages": messages + [
                            {
                                "role": "assistant",
                                "content": f"Now applying your requested changes to the {label}.",
                            }
                        ],
                    }

        logger.info("Calling LLM for routing decision...")
        
        # Make a single LLM call to determine routing (retried: one parse error or rate limit must
        # not end the session after the user has pasted the ad and CV)
        response = chain.with_retry(stop_after_attempt=3).invoke(llm_input)
        
        logger.info(f"LLM response received - next_action: {response.next_action}, needs_user_input: {response.needs_user_input}")
        logger.debug(f"LLM response - message_to_user: {response.message_to_user[:200] if response.message_to_user else 'None'}...")
        logger.debug(f"LLM response - cv_content_feedback: {response.cv_content_feedback[:200] if response.cv_content_feedback else 'None'}...")
        logger.debug(f"LLM response - cover_letter_content_feedback: {response.cover_letter_content_feedback[:200] if response.cover_letter_content_feedback else 'None'}...")
        logger.debug(f"LLM response - cv_pdf_style: {response.cv_pdf_style[:200] if response.cv_pdf_style else 'None'}...")
        logger.debug(f"LLM response - cover_letter_pdf_style: {response.cover_letter_pdf_style[:200] if response.cover_letter_pdf_style else 'None'}...")
        logger.debug(f"Available actions were: {available_actions}")
        
        # Use the LLM's routing decision, but only if it is one of the actions offered. The model can
        # return any value of the RouterResponse literal; without this check it could pick 'draft_cv'
        # after the refinement limit was reached and keep redrafting without the user.
        next_action = response.next_action
        offered = _offered_actions(available_actions)
        if next_action not in offered:
            logger.warning(f"Router LLM chose '{next_action}', which was not offered ({sorted(offered)}) - asking the user instead")
            # Actions are withdrawn once they have run (the latest message is then the assistant's);
            # only then can the message say the step is done.
            done_note = (
                _FINISHED_ACTION_NOTE.get(next_action, "Done.")
                if not self._has_user_feedback(messages)
                else "I can't do that step yet."
            )
            # The writers' notes are cleared below once a message is shown, so this fixed message
            # must carry them (e.g. a part of the request that was not applied).
            notes = _writer_notes_for_user(state)
            ask = "Tell me what you would like to change in the CV or cover letter, or type 'exit' to finish."
            response.message_to_user = f"{done_note}\n\n{notes}\n\n{ask}" if notes else f"{done_note} {ask}"
            next_action = "collect_user_input"
        
        # Pending fixed notices (e.g. claims still unsupported in a final draft, a PDF that does not
        # match its text) go in front of the next message the user sees, word for word: the model
        # cannot drop them. The user sees the router's message only when it asks them something (or
        # at exit), so on any other action they stay pending.
        message_to_user = response.message_to_user or ""
        shows_message = next_action in ("collect_user_input", "exit")
        if shows_message and pending_notices(state):
            message_to_user = with_notices(state, message_to_user)
            if next_action == "exit":
                logger.info(f"\n\n{message_to_user}")  # the console is where the user reads the app's messages
        return_state = {
            "next": next_action,
            "messages": messages + [
                {
                    "role": "assistant",
                    "content": message_to_user or f"Proceeding with: {next_action}",
                }
            ],
        }
        
        if shows_message and pending_notices(state):
            return_state[NOTICES_KEY] = None
        if shows_message:
            # Shown to the user with this message (the prompt asks the model to state what was not
            # applied as asked); cleared so they are not repeated after a later request.
            for kind in _WRITER_NOTE_LABELS:
                if state.get(f"{kind}_request_notes"):
                    return_state[f"{kind}_request_notes"] = None

        if response.work_preference and response.work_preference != state.get("work_preference"):
            # Only what the user said; the writers state it honestly and the critique counts it as the user's own fact.
            logger.info(f"User stated a work-arrangement preference: {response.work_preference}")
            return_state["work_preference"] = response.work_preference

        # Store content feedback for the writers. Feedback for a document that does not exist yet
        # is kept for later (the user may want the other document first). Feedback for an existing
        # document is stored only on the turn that redrafts it, or that redrafts the other document
        # (it is then redrafted right after that one); finalize clears it afterwards, so a request
        # is applied once and cannot loop.
        cv_feedback = (response.cv_content_feedback or "").strip()
        cover_letter_feedback = (response.cover_letter_content_feedback or "").strip()
        latest_user_message = _latest_user_message(messages)

        if next_action == "draft_cv":
            return_state.update(_new_draft_cycle("cv", state))
            if not cv_feedback and generated_cv and latest_user_message:
                # A cheap model sometimes routes to draft_cv but leaves the feedback field empty;
                # without this the writer would get no instruction and the request would be lost.
                cv_feedback = latest_user_message
                logger.info("Router left cv_content_feedback empty - using the user's latest message as the CV request")
            if cv_feedback:
                return_state["cv_content_feedback"] = cv_feedback
                logger.info(f"Storing CV content feedback: {cv_feedback[:100]}")
        elif cv_feedback and not generated_cv:
            return_state["cv_content_feedback"] = cv_feedback
            logger.info(f"Storing CV content feedback (will be used when the CV is drafted): {cv_feedback[:100]}")
        elif cv_feedback and next_action == "draft_cover_letter":
            # One message with changes for both existing documents: the cover letter goes first,
            # and the CV is redrafted with this request once the letter is finalized (see above).
            return_state["cv_content_feedback"] = cv_feedback
            logger.info(f"Storing CV content feedback (will be applied after the cover letter): {cv_feedback[:100]}")
        elif cv_feedback:
            logger.info("CV feedback extracted but the router did not choose draft_cv - not storing it")

        if next_action == "draft_cover_letter":
            return_state.update(_new_draft_cycle("cover_letter", state))
            if not cover_letter_feedback and generated_cover_letter and latest_user_message:
                cover_letter_feedback = latest_user_message
                logger.info("Router left cover_letter_content_feedback empty - using the user's latest message as the cover letter request")
            if cover_letter_feedback:
                return_state["cover_letter_content_feedback"] = cover_letter_feedback
                logger.info(f"Storing cover letter content feedback: {cover_letter_feedback[:100]}")
        elif cover_letter_feedback and not generated_cover_letter:
            return_state["cover_letter_content_feedback"] = cover_letter_feedback
            logger.info(f"Storing cover letter content feedback (will be used when the cover letter is drafted): {cover_letter_feedback[:100]}")
        elif cover_letter_feedback and next_action == "draft_cv":
            # One message with changes for both existing documents: the CV goes first, and the
            # cover letter is redrafted with this request once the CV is finalized (see above).
            return_state["cover_letter_content_feedback"] = cover_letter_feedback
            logger.info(f"Storing cover letter content feedback (will be applied after the CV): {cover_letter_feedback[:100]}")
        elif cover_letter_feedback:
            logger.info("Cover letter feedback extracted but the router did not choose draft_cover_letter - not storing it")

        # Store style feedback (for PDF generator) - can be stored regardless of next_action
        # Style will be applied when PDF is generated (after content is finalized). A new request
        # is added to the earlier ones, so the user's earlier style requests are kept.
        if response.cv_pdf_style:
            return_state["cv_pdf_style"] = _accumulate_style(state.get("cv_pdf_style"), response.cv_pdf_style)
            if response.next_action == "update_cv_pdf_style":
                logger.info(f"Routing to update_cv_pdf_style with style: {response.cv_pdf_style[:100]}..." if len(response.cv_pdf_style) > 100 else f"Routing to update_cv_pdf_style with style: {response.cv_pdf_style}")
            else:
                logger.info(f"Storing CV PDF style feedback (will be applied when PDF is generated): {response.cv_pdf_style[:100]}..." if len(response.cv_pdf_style) > 100 else f"Storing CV PDF style feedback (will be applied when PDF is generated): {response.cv_pdf_style}")
        
        if response.cover_letter_pdf_style:
            return_state["cover_letter_pdf_style"] = _accumulate_style(state.get("cover_letter_pdf_style"), response.cover_letter_pdf_style)
            if response.next_action == "update_cover_letter_pdf_style":
                logger.info(f"Routing to update_cover_letter_pdf_style with style: {response.cover_letter_pdf_style[:100]}..." if len(response.cover_letter_pdf_style) > 100 else f"Routing to update_cover_letter_pdf_style with style: {response.cover_letter_pdf_style}")
            else:
                logger.info(f"Storing cover letter PDF style feedback (will be applied when PDF is generated): {response.cover_letter_pdf_style[:100]}..." if len(response.cover_letter_pdf_style) > 100 else f"Storing cover letter PDF style feedback (will be applied when PDF is generated): {response.cover_letter_pdf_style}")
        
        # If routing to collect_user_input, set user_input_message for the UserInputAgent
        if next_action == "collect_user_input" and message_to_user:
            return_state["user_input_message"] = message_to_user
            logger.debug(f"Setting user_input_message for collect_user_input: {message_to_user[:100]}...")
        
        return return_state
