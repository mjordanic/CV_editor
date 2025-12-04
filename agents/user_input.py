import logging
from langgraph.types import interrupt

logger = logging.getLogger(__name__)


class UserInputAgent:
    """Agent for collecting user input. Uses interrupt() to pause and get user input."""
    
    def __init__(self):
        """
        Initialize the UserInputAgent.

        Args:
            None

        Returns:
            None
        """
        logger.info("Initializing UserInputAgent...")
    
    def run(self, state):
        """
        Main method to collect user input from the user.
        
        This node uses interrupt() to pause execution and wait for user input.
        When resumed, the interrupt() call returns the user's input.
        
        Args:
            state: The state dictionary containing messages and user_input_message
            
        Returns:
            dict: Updated state with user input added to messages
        """
        logger.info("UserInputAgent.run() called")
        
        # Extract state information
        messages = state.get("messages", [])
        user_input_message = state.get("user_input_message", "Please provide your input:")
        
        logger.debug(f"UserInputAgent - messages: {len(messages)}, user_input_message: {user_input_message[:100] if user_input_message else 'None'}...")
        
        # Use interrupt() to pause execution and get user input
        # The interrupt() call will pause the graph and return the user's input when resumed
        logger.info("Requesting user input via interrupt()")
        user_input = interrupt({"message": user_input_message, "required": True})
        
        # Ensure we have a string
        user_input = str(user_input).strip()
        
        if not user_input:
            logger.warning("UserInputAgent: No user input received. This might be an error.")
            return {}
        
        logger.info(f"UserInputAgent: Processing user input - length: {len(user_input)} characters")
        logger.debug(f"UserInputAgent: User input preview: {user_input[:100]}..." if len(user_input) > 100 else f"UserInputAgent: User input: {user_input}")
        
        # Add user input to messages
        updated_messages = messages + [{"role": "user", "content": user_input}]
        
        # Clear the user_input_message from state
        return {
            "messages": updated_messages,
            "user_input_message": None,  # Clear it
            "current_node": "user_input"
        }

