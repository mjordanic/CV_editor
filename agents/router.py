from typing import Literal, Optional
import logging
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from debug_utils import log_messages, get_logged_message_count

logger = logging.getLogger(__name__)


ROUTER_SYSTEM_PROMPT = (
    "You are a helpful assistant that routes user requests for CV and cover letter generation. "
    "Your role is to interact with other agents for CV and cover letter generation, and with the user, understand what the user wants to generate, summarize "
    "their request to a short actionable message and collect feedback from the user"
    "after documents are created. You should be friendly and professional.\n\n"
    "Based on the conversation history, job description, company information, and the status of generated documents, "
    "you need to:\n"
    "1. Determine what message to show the user (if any)\n"
    "2. Decide what the next action should be (from the available actions listed below)\n"
    "3. Extract and summarize any feedback for modifications. CRITICAL: When routing to 'draft_cv', the feedback MUST ONLY contain CV-related instructions. "
    "Filter out any cover letter-related instructions. When routing to 'draft_cover_letter', the feedback MUST ONLY contain cover letter-related instructions. "
    "Filter out any CV-related instructions. The feedback should be short, clear, and actionable. "
    "Do NOT include your own questions, clarifications, or suggestions. Leave empty if no actionable feedback provided.\n\n"
    "Available actions (you can ONLY choose from actions listed in the context below):\n"
    "{available_actions}\n\n"
    "If documents have been generated, the user might want to:\n"
    "- Request modifications (provide feedback)\n"
    "- Generate the other document\n"
    "- Exit if satisfied\n\n"
    "If no documents have been generated yet, ask the user what they'd like to generate first."
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
    "**Available Feedback for Modifications (if any):**\n{feedback_context}\n\n"
    "Based on this context, determine:\n"
    "1. What message should be shown to the user (if the conversation needs user input)\n"
    "2. What the next action should be (you can ONLY choose from the available actions listed in the system prompt)\n"
    "3. Summarize any feedback for modification of documents. CRITICAL RULES:\n"
    "   - If routing to 'draft_cv': Extract ONLY CV-related feedback. Filter out and ignore any mentions of cover letter or other documents. Focus solely on CV improvements.\n"
    "   - If routing to 'draft_cover_letter': Extract ONLY cover letter-related feedback. Filter out and ignore any mentions of CV or other documents. Focus solely on cover letter improvements.\n"
    "   - Extract ONLY the actual feedback from the available feedback context. Do NOT include your own questions, clarifications, or suggestions.\n"
    "   - It should be short, clear, and actionable. Leave empty if no actionable feedback provided.\n\n"
    "If this is the first interaction and no documents have been generated, ask the user what they'd like to generate.\n"
    "If documents have been generated, ask for feedback or if they want to generate the other document.\n"
    "If feedback exists and indicates refinement is needed, route to the appropriate generation action with the summarized feedback (if that action is available)."
)


class RouterResponse(BaseModel):
    """Response model for router decisions."""
    next_action: Literal["draft_cv", "draft_cover_letter", "collect_user_input", "exit"] = Field(
        ...,
        description="The next action to take: draft_cv, draft_cover_letter, collect_user_input, or exit"
    )
    message_to_user: str = Field(
        ...,
        description="The message to show to the user. If empty, no message is needed (e.g., when routing directly to an action)."
    )
    feedback: str = Field(
        default="",
        description="Summarized feedback for modifications, filtered to match the target document. "
        "If routing to 'draft_cv', include ONLY CV-related feedback (filter out cover letter mentions). "
        "If routing to 'draft_cover_letter', include ONLY cover letter-related feedback (filter out CV mentions). "
        "Do NOT include your own questions or clarifications. It should be summarized, clear, and actionable. "
        "Empty if no actionable feedback provided."
    )
    needs_user_input: bool = Field(
        ...,
        description="Whether the router needs to wait for user input before proceeding. If True, show message_to_user and wait for response."
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
        
        self.llm = init_chat_model(model, temperature=temperature)
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
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
            else:
                role = getattr(msg, "role", "unknown")
                content = getattr(msg, "content", "")
            
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
        if company_info.get("remote_work"):
            parts.append(f"Remote Work Policy: {company_info['remote_work']}")
        
        return "\n".join(parts) if parts else "No company information available."
    
    def _format_candidate_text(self, candidate_text: dict | None) -> str:
        """
        Summarize any user-provided documents for LLM conditioning.

        Args:
            candidate_text: Dictionary of `cv` and `cover_letter` strings from `DocumentReaderAgent`.

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
        
        return "\n".join(parts) if parts else "No original documents uploaded by the user."
    
    def _has_user_feedback(self, messages: list) -> bool:
        """
        Check if the most recent interaction is from the user (not from critique/assistant).
        This determines if we should allow draft actions for user-initiated modifications.
        
        Args:
            messages: Conversation messages
            
        Returns:
            bool: True if the most recent non-empty message is from the user, False otherwise
        """
        if not messages:
            return False
        
        # Check messages in reverse order (most recent first)
        # Find the first non-empty message to determine the source of the current interaction
        for msg in reversed(messages):
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
            else:
                role = getattr(msg, "role", "")
                content = getattr(msg, "content", "")
            
            # Skip empty messages
            if not content or not content.strip():
                continue
            
            # If we find a user message, that's the most recent interaction
            if role == "user":
                return True
            # If we find an assistant/system message first, the most recent interaction is not from user
            elif role in ("assistant", "system"):
                return False
        
        return False
    
    def _build_available_actions(
        self,
        cv_refinement_allowed: bool,
        cover_letter_refinement_allowed: bool,
        cv_critique_instructions: Optional[str],
        cover_letter_critique_instructions: Optional[str],
        messages: list
    ) -> str:
        """
        Build the list of available actions dynamically based on refinement status and user feedback.
        
        Args:
            cv_refinement_allowed: Whether CV critique-based refinement is allowed
            cover_letter_refinement_allowed: Whether cover letter critique-based refinement is allowed
            cv_critique_instructions: Critique instructions for CV (if any)
            cover_letter_critique_instructions: Critique instructions for cover letter (if any)
            messages: Conversation messages (to check for user feedback)
            
        Returns:
            str: Formatted list of available actions
        """
        actions = []
        
        # Check if there's user feedback (last user message)
        has_user_feedback = self._has_user_feedback(messages)
        
        # Determine if draft_cv should be available
        # Available if: refinement allowed OR user feedback exists OR no critique instructions (first generation)
        cv_available = (
            cv_refinement_allowed or 
            has_user_feedback or 
            not cv_critique_instructions
        )
        
        # Determine if draft_cover_letter should be available
        # Available if: refinement allowed OR user feedback exists OR no critique instructions (first generation)
        cover_letter_available = (
            cover_letter_refinement_allowed or 
            has_user_feedback or 
            not cover_letter_critique_instructions
        )
        
        # Build actions list
        if cv_available:
            actions.append("- 'draft_cv': Generate a new CV or modify an existing one (ONLY CV-related feedback should be passed)")
        
        if cover_letter_available:
            actions.append("- 'draft_cover_letter': Generate a new cover letter or modify an existing one (ONLY cover letter-related feedback should be passed)")
        
        # Always available actions
        actions.append("- 'collect_user_input': Request user input (use this when you need to ask the user a question or get feedback)")
        actions.append("- 'exit': End the conversation")
        
        return "\n".join(actions)
    
    def _format_feedback_context(
        self,
        messages: list,
        cv_critique_instructions: Optional[str],
        cover_letter_critique_instructions: Optional[str],
        cv_needs_refinement: bool,
        cover_letter_needs_refinement: bool,
        cv_refinement_allowed: bool,
        cover_letter_refinement_allowed: bool
    ) -> str:
        """
        Format all available feedback sources into a single context string.
        Only includes critique instructions if refinement is allowed (counter check done before calling this).
        
        Args:
            messages: Conversation messages (may contain user feedback)
            cv_critique_instructions: Critique instructions for CV
            cover_letter_critique_instructions: Critique instructions for cover letter
            cv_needs_refinement: Whether CV needs refinement (from critique)
            cover_letter_needs_refinement: Whether cover letter needs refinement (from critique)
            cv_refinement_allowed: Whether CV refinement is allowed (counter < max)
            cover_letter_refinement_allowed: Whether cover letter refinement is allowed (counter < max)
            
        Returns:
            str: Combined feedback context (only includes critique if refinement allowed)
        """
        parts = []
        
        # Extract user feedback from messages (last user message)
        user_feedback = None
        if messages:
            for msg in reversed(messages):
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                else:
                    role = getattr(msg, "role", "")
                    content = getattr(msg, "content", "")
                
                if role == "user" and content and content.strip():
                    user_feedback = content.strip()
                    break
        
        if user_feedback:
            parts.append("**User Feedback:**")
            parts.append(user_feedback)
            parts.append("")
        
        # Add critique instructions if they exist and refinement is allowed
        if cv_critique_instructions and cv_refinement_allowed:
            parts.append("**CV Critique Improvement Instructions:**")
            parts.append(cv_critique_instructions)
            parts.append("")
        
        if cover_letter_critique_instructions and cover_letter_refinement_allowed:
            parts.append("**Cover Letter Critique Improvement Instructions:**")
            parts.append(cover_letter_critique_instructions)
            parts.append("")
        
        if not parts:
            return "No feedback available."
        
        return "\n".join(parts)
    
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
        company_info = state.get("company_info")
        candidate_text = state.get("candidate_text")
        generated_cv = state.get("generated_cv")
        generated_cover_letter = state.get("generated_cover_letter")
        cv_critique_instructions = state.get("cv_critique_improvement_instructions")
        cover_letter_critique_instructions = state.get("cover_letter_critique_improvement_instructions")
        cv_needs_refinement = state.get("cv_needs_refinement", False)
        cover_letter_needs_refinement = state.get("cover_letter_needs_refinement", False)
        cv_refinement_count = state.get("cv_refinement_count", 0)
        cover_letter_refinement_count = state.get("cover_letter_refinement_count", 0)
        
        # Since Router is only called when auto-refinement is skipped (limit reached) or not needed,
        # we can assume refinement is NOT allowed in this context.
        # The conditional edge in langgraph_agent.py handles the "allowed" case.
        cv_refinement_allowed = False
        cover_letter_refinement_allowed = False
        
        # If limit reached, explicitly set needs_refinement to False to prevent routing
        if cv_needs_refinement:
            logger.info(f"CV refinement limit reached ({cv_refinement_count}) - router will not route for refinement")
            cv_needs_refinement = False  # Override to prevent routing
        if cover_letter_needs_refinement:
            logger.info(f"Cover letter refinement limit reached ({cover_letter_refinement_count}) - router will not route for refinement")
            cover_letter_needs_refinement = False  # Override to prevent routing
        
        logger.debug(f"State extracted - messages: {len(messages)}, cv_generated: {generated_cv is not None}, cover_letter_generated: {generated_cover_letter is not None}, cv_critique_instructions: {cv_critique_instructions is not None}, cover_letter_critique_instructions: {cover_letter_critique_instructions is not None}, cv_needs_refinement: {cv_needs_refinement} (allowed: {cv_refinement_allowed}), cover_letter_needs_refinement: {cover_letter_needs_refinement} (allowed: {cover_letter_refinement_allowed})")
        
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
        feedback_context = self._format_feedback_context(
            messages,
            cv_critique_instructions,
            cover_letter_critique_instructions,
            cv_needs_refinement,
            cover_letter_needs_refinement,
            cv_refinement_allowed,
            cover_letter_refinement_allowed
        )
        
        # Build available actions dynamically based on refinement status
        available_actions = self._build_available_actions(
            cv_refinement_allowed,
            cover_letter_refinement_allowed,
            cv_critique_instructions,
            cover_letter_critique_instructions,
            messages
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
        # Counter check already done - critique instructions only included if refinement allowed
        # Available actions are dynamically built based on refinement status
        llm_input = {
            "available_actions": available_actions,
            "messages_history": messages_history,
            "job_description_info": job_desc_formatted,
            "company_info": company_info_formatted,
            "candidate_text": candidate_text_formatted,
            "cv_generated": "Yes" if generated_cv is not None else "No",
            "cover_letter_generated": "Yes" if generated_cover_letter is not None else "No",
            "feedback_context": feedback_context
        }
        
        # If critique indicates refinement is needed but we are here, it means limit was reached
        # The router should present options to the user
        if cv_critique_instructions and state.get("cv_needs_refinement"):
            logger.info("CV needs refinement but auto-refinement skipped (limit reached) - router will ask user")
        if cover_letter_critique_instructions and state.get("cover_letter_needs_refinement"):
            logger.info("Cover letter needs refinement but auto-refinement skipped (limit reached) - router will ask user")
        
        logger.debug(f"LLM input prepared - messages_history length: {len(messages_history)}, feedback_context length: {len(feedback_context)}")
        logger.info("Calling LLM for routing decision...")
        
        # Make a single LLM call to determine routing
        response = chain.invoke(llm_input)
        
        logger.info(f"LLM response received - next_action: {response.next_action}, needs_user_input: {response.needs_user_input}")
        logger.debug(f"LLM response - message_to_user: {response.message_to_user[:200] if response.message_to_user else 'None'}...")
        logger.debug(f"LLM response - feedback: {response.feedback[:200] if response.feedback else 'None'}...")
        logger.debug(f"Available actions were: {available_actions}")
        
        # Use the LLM's routing decision directly (available actions were constrained based on refinement status)
        next_action = response.next_action
        
        # Determine feedback source and populate appropriate state field
        # After user input OR critique, we always go to router, so we know the source
        # Counter check already done - if critique instructions are in feedback_context, refinement is allowed
        # If critique instructions were omitted (limit reached), router LLM won't see them and won't route for refinement
        
        # Prepare return state
        return_state = {
            "next": next_action,
            "messages": messages + [{
                "role": "assistant",
                "content": response.message_to_user if response.message_to_user else f"Proceeding with: {next_action}"
            }]
        }
        
        # If limit reached, explicitly set needs_refinement to False in return state
        # Also clear critique instructions to prevent the LLM from seeing them in future iterations
        if state.get("cv_needs_refinement"):
            return_state["cv_needs_refinement"] = False
            return_state["cv_critique_improvement_instructions"] = None  # Clear to prevent loops
            logger.debug("Setting cv_needs_refinement=False and clearing critique instructions because limit reached")
        if state.get("cover_letter_needs_refinement"):
            return_state["cover_letter_needs_refinement"] = False
            return_state["cover_letter_critique_improvement_instructions"] = None  # Clear to prevent loops
            logger.debug("Setting cover_letter_needs_refinement=False and clearing critique instructions because limit reached")
        
        if response.feedback:
            # Check if this is critique-based feedback (refinement needed and allowed) or user feedback
            # If cv_refinement_allowed is True, critique instructions were included in feedback_context
            # Note: next_action may have been overridden, so check the original response.next_action for feedback routing
            if response.next_action == "draft_cv":
                # Always reset refinement count when routing to draft_cv (new generation cycle)
                return_state["cv_refinement_count"] = 0
                # Reset CV history for fresh loop
                return_state["cv_history"] = []
                logger.debug("Resetting CV refinement counter and cv_history for new draft")
                
                # Critique instructions were NOT in feedback_context (limit reached or user feedback)
                # This is user feedback - populate user_feedback field
                return_state["user_feedback"] = response.feedback
                logger.debug("Storing user feedback for CV")
            elif response.next_action == "draft_cover_letter":
                # Always reset refinement count when routing to draft_cover_letter (new generation cycle)
                return_state["cover_letter_refinement_count"] = 0
                # Reset Cover Letter history for fresh loop
                return_state["cover_letter_history"] = []
                logger.debug("Resetting cover letter refinement counter and cover_letter_history for new draft")
                
                # Critique instructions were NOT in feedback_context (limit reached or user feedback)
                # This is user feedback - populate user_feedback field
                return_state["user_feedback"] = response.feedback
                logger.debug("Storing user feedback for cover letter")
            else:
                # For other actions, treat as user feedback and reset both counters
                return_state["user_feedback"] = response.feedback
                return_state["cv_refinement_count"] = 0  # Reset counter for user-initiated changes
                return_state["cover_letter_refinement_count"] = 0  # Reset counter for user-initiated changes
                logger.debug("Storing user feedback and resetting refinement counters")
        
        # If routing to collect_user_input, set user_input_message for the UserInputAgent
        if next_action == "collect_user_input" and response.message_to_user:
            return_state["user_input_message"] = response.message_to_user
            logger.debug(f"Setting user_input_message for collect_user_input: {response.message_to_user[:100]}...")
        
        return return_state
