from typing import Literal
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import interrupt
from pydantic import BaseModel, Field


ROUTER_SYSTEM_PROMPT = (
    "You are a helpful assistant that routes user requests for CV and cover letter generation. "
    "Your role is to interact with the user, understand what they want to generate, and collect feedback "
    "after documents are created. You should be friendly and professional."
)

ROUTER_INITIAL_PROMPT = (
    "I have gathered all the necessary information about the job and company. "
    "What would you like me to generate?\n"
    "- Type 'cv' to generate a tailored CV\n"
    "- Type 'cover letter' to generate a cover letter\n"
    "- Type 'exit' to finish"
)

ROUTER_FEEDBACK_PROMPT = (
    "I've generated the {document_type}. Please review it and let me know:\n"
    "- If you're satisfied, type 'done' or 'exit'\n"
    "- If you'd like modifications, please describe what changes you'd like\n"
    "- If you'd like to generate the other document, type 'cv' or 'cover letter'"
)


class RouterResponse(BaseModel):
    """Response model for router decisions."""
    next_action: Literal["generate_cv", "generate_cover_letter", "exit", "modify_cv", "modify_cover_letter"] = Field(
        ...,
        description="The next action to take: generate_cv, generate_cover_letter, exit, modify_cv, or modify_cover_letter"
    )
    user_feedback: str = Field(
        default="",
        description="Any feedback or modification requests from the user"
    )


class RouterAgent:
    """Agent for routing user requests and collecting feedback."""
    
    def __init__(self):
        self.llm = init_chat_model("openai:gpt-5-nano", temperature=0)
    
    def _determine_next_action(self, user_input: str, has_cv: bool, has_cover_letter: bool) -> RouterResponse:
        """
        Determine the next action based on user input.
        
        Args:
            user_input: User's input text
            has_cv: Whether a CV has been generated
            has_cover_letter: Whether a cover letter has been generated
            
        Returns:
            RouterResponse with the next action
        """
        user_input_lower = user_input.lower().strip()
        
        # Check for explicit commands
        if user_input_lower in ["exit", "done", "quit", "finish"]:
            return RouterResponse(next_action="exit", user_feedback="")
        
        if "cv" in user_input_lower and "cover letter" not in user_input_lower:
            if has_cv and ("modify" in user_input_lower or "change" in user_input_lower or "edit" in user_input_lower):
                return RouterResponse(next_action="modify_cv", user_feedback=user_input)
            return RouterResponse(next_action="generate_cv", user_feedback="")
        
        if "cover letter" in user_input_lower or ("cover" in user_input_lower and "letter" in user_input_lower):
            if has_cover_letter and ("modify" in user_input_lower or "change" in user_input_lower or "edit" in user_input_lower):
                return RouterResponse(next_action="modify_cover_letter", user_feedback=user_input)
            return RouterResponse(next_action="generate_cover_letter", user_feedback="")
        
        # If user provides feedback for modification
        if has_cv or has_cover_letter:
            if has_cv and not has_cover_letter:
                return RouterResponse(next_action="modify_cv", user_feedback=user_input)
            elif has_cover_letter and not has_cv:
                return RouterResponse(next_action="modify_cover_letter", user_feedback=user_input)
            # If both exist, ask for clarification
            return RouterResponse(next_action="exit", user_feedback=user_input)
        
        # Default: ask again
        return RouterResponse(next_action="exit", user_feedback="")
    
    def run(self, state):
        """
        Main method to route user requests and collect feedback.
        
        Args:
            state: The state dictionary containing messages and generated documents
            
        Returns:
            dict: Updated state with next action
        """
        has_cv = bool(state.get("generated_cv"))
        has_cover_letter = bool(state.get("generated_cover_letter"))
        
        # Determine what message to show
        if not has_cv and not has_cover_letter:
            # Initial routing - ask what to generate
            user_input = interrupt({
                "message": ROUTER_INITIAL_PROMPT,
                "required": True
            })
        else:
            # Collect feedback after generation
            document_type = "CV" if has_cv else "cover letter"
            user_input = interrupt({
                "message": ROUTER_FEEDBACK_PROMPT.format(document_type=document_type),
                "required": True
            })
        
        # Determine next action
        response = self._determine_next_action(str(user_input), has_cv, has_cover_letter)
        
        # Map modify actions to generate actions (the writer agents will handle modification)
        next_action = response.next_action
        if next_action == "modify_cv":
            next_action = "generate_cv"
        elif next_action == "modify_cover_letter":
            next_action = "generate_cover_letter"
        
        return {
            "next": next_action,
            "user_feedback": response.user_feedback,
            "messages": [{
                "role": "assistant",
                "content": f"Understood. Proceeding with: {next_action}" if next_action != "exit" else "Thank you! Good luck with your application."
            }]
        }

