from typing import Literal
import logging
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from debug_utils import log_messages, get_logged_message_count

logger = logging.getLogger(__name__)


ROUTER_SYSTEM_PROMPT = (
    "You are a helpful assistant that routes user requests for CV and cover letter generation. "
    "Your role is to interact with the user, understand what they want to generate, and collect feedback "
    "after documents are created. You should be friendly and professional.\n\n"
    "Based on the conversation history, job description, company information, and the status of generated documents, "
    "you need to:\n"
    "1. Determine what message to show the user (if any)\n"
    "2. Decide what the next action should be\n"
    "3. Extract any user feedback for modifications\n\n"
    "Available actions:\n"
    "- 'generate_cv': Generate a new CV or modify an existing one\n"
    "- 'generate_cover_letter': Generate a new cover letter or modify an existing one\n"
    "- 'user_input': Request user input (use this when you need to ask the user a question or get feedback)\n"
    "- 'exit': End the conversation\n\n"
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
    "Based on this context, determine:\n"
    "1. What message should be shown to the user (if the conversation needs user input)\n"
    "2. What the next action should be\n"
    "3. Any feedback the user has provided (for modifications)\n\n"
    "If this is the first interaction and no documents have been generated, ask the user what they'd like to generate.\n"
    "If documents have been generated, ask for feedback or if they want to generate the other document.\n"
    "If the user has provided feedback, route to the appropriate generation action with that feedback."
)


class RouterResponse(BaseModel):
    """Response model for router decisions."""
    next_action: Literal["generate_cv", "generate_cover_letter", "user_input", "exit"] = Field(
        ...,
        description="The next action to take: generate_cv, generate_cover_letter, user_input, or exit"
    )
    message_to_user: str = Field(
        ...,
        description="The message to show to the user. If empty, no message is needed (e.g., when routing directly to an action)."
    )
    user_feedback: str = Field(
        default="",
        description="Any feedback or modification requests from the user. Empty if no feedback provided."
    )
    needs_user_input: bool = Field(
        ...,
        description="Whether the router needs to wait for user input before proceeding. If True, show message_to_user and wait for response."
    )


class RouterAgent:
    """Agent for routing user requests and collecting feedback using LLM."""
    
    def __init__(self):
        """
        Initialize the router agent and its backing LLM client.

        Args:
            None

        Returns:
            None
        """
        logger.info("Initializing RouterAgent...")
        self.llm = init_chat_model("openai:gpt-5-nano", temperature=0)
        logger.debug("RouterAgent LLM initialized")
    
    def _format_messages_history(self, messages: list) -> str:
        """
        Format the recent conversation history for inclusion in the router prompt.

        Args:
            messages: Sequence of LangChain message objects or message dictionaries.

        Returns:
            str: Human-readable transcript limited to the most recent ten entries.
        """
        if not messages:
            return "No conversation history yet."
        
        formatted = []
        for msg in messages[-10:]:  # Last 10 messages for context
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
            "messages_history": messages_history,
            "job_description_info": job_desc_formatted,
            "company_info": company_info_formatted,
            "candidate_text": candidate_text_formatted,
            "cv_generated": "Yes" if generated_cv is not None else "No",
            "cover_letter_generated": "Yes" if generated_cover_letter is not None else "No"
        }
        
        logger.debug(f"LLM input prepared - messages_history length: {len(messages_history)}, job_desc length: {len(job_desc_formatted)}")
        logger.info("Calling LLM for routing decision...")
        
        # Make a single LLM call to determine routing
        response = chain.invoke(llm_input)
        
        logger.info(f"LLM response received - next_action: {response.next_action}, needs_user_input: {response.needs_user_input}")
        logger.debug(f"LLM response - message_to_user: {response.message_to_user[:200] if response.message_to_user else 'None'}...")
        
        # If we need user input, route to user_input node
        if response.needs_user_input and response.message_to_user:
            logger.info("Router needs user input - routing to user_input node")
            logger.debug(f"User input message: {response.message_to_user}")
            
            return {
                "next": "user_input",
                "user_input_message": response.message_to_user,
                "messages": messages + [{
                    "role": "assistant",
                    "content": response.message_to_user
                }]
            }
        
        # If we don't need user input, use the response directly and return
        logger.info("No user input needed - proceeding directly with routing decision")
        
        # Extract user feedback if this is a modification request
        user_feedback = response.user_feedback
        
        # If we're generating and documents already exist, check if there's feedback in messages
        if response.next_action in ["generate_cv", "generate_cover_letter"]:
            if (response.next_action == "generate_cv" and generated_cv is not None) or \
               (response.next_action == "generate_cover_letter" and generated_cover_letter is not None):
                # Check last user message for feedback
                if messages:
                    last_user_msg = None
                    for msg in reversed(messages):
                        if isinstance(msg, dict):
                            role = msg.get("role", "")
                        else:
                            role = getattr(msg, "role", "")
                        if role == "user":
                            last_user_msg = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                            break
                    if last_user_msg and not user_feedback:
                        user_feedback = last_user_msg
        
        logger.info(f"Router decision: next_action={response.next_action}, user_feedback_length={len(user_feedback)}")
        
        return {
            "next": response.next_action,
            "user_feedback": user_feedback,
            "messages": messages + [{
                "role": "assistant",
                "content": response.message_to_user if response.message_to_user else f"Proceeding with: {response.next_action}"
            }]
        }
