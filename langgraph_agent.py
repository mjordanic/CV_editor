from dotenv import load_dotenv

# Load environment variables BEFORE importing agents
load_dotenv()

# Set up logging
import logging
from datetime import datetime
import os
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# Create logs directory if it doesn't exist
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

# Create log filename with timestamp
log_filename = os.path.join(logs_dir, f"cv_editor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Configure logging with both file and console handlers
logging.basicConfig(
    level=logging.DEBUG,  # Change to logging.DEBUG for more verbose output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()  # Console output
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Logging to file: {log_filename}")

from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from typing_extensions import TypedDict

from agents.search import SearchAgent
from agents.document_reader import DocumentReaderAgent
from agents.job_description import JobDescriptionAgent
from agents.router import RouterAgent
from agents.cv_writer import CVWriterAgent
from agents.cover_letter_writer import CoverLetterWriterAgent


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    job_description_info: dict | None  # Extracted job description information
    candidate_text: dict | None  # CV and cover letter text (keys: 'cv', 'cover_letter')
    company_info: dict | None  # Company information from search (keys: 'company_description', 'remote_work', 'search_results')
    next: Literal["generate_cv", "generate_cover_letter", "exit"] | None  # Next action to take
    generated_cv: str | None  # Generated CV text
    generated_cover_letter: str | None  # Generated cover letter text
    user_feedback: str | None  # User feedback for modifications









class MasterAgent:
    """Master agent that orchestrates the CV editing workflow using LangGraph."""
    
    def __init__(self):
        """Initialize the MasterAgent and build the LangGraph workflow."""
        logger.info("Initializing MasterAgent...")
        
        # Build the graph
        logger.debug("Creating StateGraph")
        graph_builder = StateGraph(State)

        # Instantiate agents
        logger.debug("Instantiating agents...")
        search_agent = SearchAgent()
        document_reader_agent = DocumentReaderAgent()
        job_description_agent = JobDescriptionAgent()
        router_agent = RouterAgent()
        cv_writer_agent = CVWriterAgent()
        cover_letter_writer_agent = CoverLetterWriterAgent()
        logger.debug("All agents instantiated successfully")

        # Add nodes to the graph
        logger.debug("Adding nodes to graph...")
        graph_builder.add_node("get_candidate_information", document_reader_agent.run)
        graph_builder.add_node("get_job_description", job_description_agent.run)
        graph_builder.add_node("get_company_information", search_agent.run)
        graph_builder.add_node("router", router_agent.run)
        graph_builder.add_node("generate_cv", cv_writer_agent.run)
        graph_builder.add_node("generate_cover_letter", cover_letter_writer_agent.run)
        logger.debug("All nodes added to graph")

        # Add edges to the graph
        logger.debug("Adding edges to graph...")
        graph_builder.add_edge(START, "get_candidate_information")
        graph_builder.add_edge("get_candidate_information", "get_job_description")
        graph_builder.add_edge("get_job_description", "get_company_information")
        graph_builder.add_edge("get_company_information", "router")

        graph_builder.add_conditional_edges(
            "router",
            lambda state: state.get("next"),
            {"generate_cv": "generate_cv", "generate_cover_letter": "generate_cover_letter", "exit": END}
        )
        graph_builder.add_edge("generate_cv", "router")
        graph_builder.add_edge("generate_cover_letter", "router")
        logger.debug("All edges added to graph")

        # Create a checkpointer for persistence (required for interrupts)
        logger.debug("Creating MemorySaver checkpointer")
        checkpointer = MemorySaver()

        # Compile the graph with checkpointer
        logger.debug("Compiling graph with checkpointer...")
        self.graph = graph_builder.compile(checkpointer=checkpointer)
        logger.info("MasterAgent initialized successfully - graph compiled and ready")

    def run(self):
        """Run the LangGraph agent workflow."""
        logger.info("MasterAgent.run() called - starting workflow execution")
        
        # Use a thread_id to maintain state across invocations
        thread_id = "main-thread"
        config = {"configurable": {"thread_id": thread_id}}
        logger.debug(f"Using thread_id: {thread_id}")
        
        # Initial state
        initial_state = State(
            messages=[],
            job_description_info=None,
            candidate_text=None,
            company_info=None,
            next=None,
            generated_cv=None,
            generated_cover_letter=None,
            user_feedback=None
        )
        logger.debug("Initial state created")

        iteration_count = 0
        while True:
            try:
                iteration_count += 1
                logger.info(f"Graph invocation #{iteration_count} - invoking graph with current state")
                logger.debug(f"State keys: {list(initial_state.keys())}")
                
                # Invoke the graph with the current state
                result = self.graph.invoke(initial_state, config)
                logger.debug(f"Graph invocation completed. Result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
                
                # Check if there's an interrupt
                if hasattr(result, "__interrupt__") and result.__interrupt__:
                    logger.info("Graph execution interrupted - waiting for user input")
                    # Graph is paused waiting for input
                    interrupt_data = result.__interrupt__
                    if isinstance(interrupt_data, list) and len(interrupt_data) > 0:
                        interrupt_value = interrupt_data[0].get("value", {})
                        if isinstance(interrupt_value, dict):
                            message = interrupt_value.get("message", "Please provide input:")
                        else:
                            message = str(interrupt_value)
                    else:
                        message = "Please provide input:"
                    
                    logger.debug(f"Interrupt message: {message}")
                    print(f"Assistant: {message}")
                    
                    # Get user input
                    logger.debug("Waiting for user input...")
                    user_input = input("You: ")
                    logger.info(f"User input received: {user_input[:100]}..." if len(user_input) > 100 else f"User input received: {user_input}")
                    
                    if user_input.lower() == "exit":
                        logger.info("User requested exit - terminating workflow")
                        print("Bye")
                        break
                    
                    # Resume the graph with user input using Command
                    logger.debug("Resuming graph execution with user input")
                    result = self.graph.invoke(Command(resume=user_input), config)
                    logger.debug("Graph resumed successfully")
                
                # Display the response if there are messages
                if result.get("messages") and len(result["messages"]) > 0:
                    last_message = result["messages"][-1]
                    # Handle both dict and message object formats
                    if isinstance(last_message, dict):
                        content = last_message.get("content", "")
                    else:
                        content = getattr(last_message, "content", "")
                    
                    if content:
                        logger.debug(f"Assistant message to display: {content[:200]}..." if len(content) > 200 else f"Assistant message: {content}")
                        print(f"Assistant: {content}")
                
                # Update initial_state for next iteration (though with checkpointer, this may not be needed)
                initial_state = result
                logger.debug("State updated for next iteration")
                
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received - terminating workflow")
                print("\nBye")
                break
            except Exception as e:
                logger.error(f"Error during graph execution: {e}", exc_info=True)
                print(f"Error: {e}")
                break
        
        logger.info("MasterAgent.run() completed - workflow execution ended")


if __name__ == "__main__":
    agent = MasterAgent()
    agent.run()