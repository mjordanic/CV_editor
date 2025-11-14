from typing import Literal
import logging
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import interrupt
from pydantic import BaseModel, Field

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
    next_action: Literal["generate_cv", "generate_cover_letter", "exit"] = Field(
        ...,
        description="The next action to take: generate_cv, generate_cover_letter, or exit"
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
        self.llm = init_chat_model("openai:gpt-5-nano", temperature=0)
    
    def _format_messages_history(self, messages: list) -> str:
        """Format message history for the prompt."""
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
        """Format job description information for the prompt."""
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
        """Format company information for the prompt."""
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
        """Format candidate text (original documents) for the prompt."""
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
        # Extract state information
        messages = state.get("messages", [])
        job_description_info = state.get("job_description_info")
        company_info = state.get("company_info")
        candidate_text = state.get("candidate_text")
        generated_cv = state.get("generated_cv")
        generated_cover_letter = state.get("generated_cover_letter")
        
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
        
        # First, determine if we need user input and what message to show
        initial_response = chain.invoke({
            "messages_history": messages_history,
            "job_description_info": job_desc_formatted,
            "company_info": company_info_formatted,
            "candidate_text": candidate_text_formatted,
            "cv_generated": "Yes" if generated_cv is not None else "No",
            "cover_letter_generated": "Yes" if generated_cover_letter is not None else "No"
        })
        
        # If we need user input, interrupt and get it
        user_input = None
        if initial_response.needs_user_input and initial_response.message_to_user:
            user_input = interrupt({
                "message": initial_response.message_to_user,
                "required": True
            })
            
            # Add user input to messages for context
            messages = messages + [{"role": "user", "content": str(user_input)}]
            messages_history = self._format_messages_history(messages)
        
        # Now get final decision from LLM with updated context (including user input if provided)
        final_response = chain.invoke({
            "messages_history": messages_history,
            "job_description_info": job_desc_formatted,
            "company_info": company_info_formatted,
            "candidate_text": candidate_text_formatted,
            "cv_generated": "Yes" if generated_cv is not None else "No",
            "cover_letter_generated": "Yes" if generated_cover_letter is not None else "No"
        })
        
        # Extract user feedback if this is a modification request
        user_feedback = ""
        if user_input and final_response.next_action in ["generate_cv", "generate_cover_letter"]:
            # If we're generating and documents already exist, treat user input as feedback
            if (final_response.next_action == "generate_cv" and generated_cv is not None) or \
               (final_response.next_action == "generate_cover_letter" and generated_cover_letter is not None):
                user_feedback = str(user_input)
            else:
                user_feedback = final_response.user_feedback
        else:
            user_feedback = final_response.user_feedback
        
        return {
            "next": final_response.next_action,
            "user_feedback": user_feedback,
            "messages": messages + [{
                "role": "assistant",
                "content": final_response.message_to_user if final_response.message_to_user else f"Proceeding with: {final_response.next_action}"
            }]
        }
