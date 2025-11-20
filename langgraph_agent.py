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

# Import debug_utils before using it
from debug_utils import initialize_debug_file

# Initialize debug file with same timestamp
debug_file_path = initialize_debug_file()
logger.info(f"Debug file initialized: {debug_file_path}")

from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from typing_extensions import TypedDict
import sys

from agents.search import SearchAgent
from agents.document_reader import DocumentReaderAgent
from agents.job_description import JobDescriptionAgent
from agents.router import RouterAgent
from agents.cv_writer import CVWriterAgent
from agents.cover_letter_writer import CoverLetterWriterAgent
from agents.user_input import UserInputAgent


class State(TypedDict, total=False):
    """
    Shared LangGraph state tracking conversation context and generation artifacts.

    Keys:
        messages: Rolling chat history consumed by LangGraph nodes.
        job_description_info: Structured information extracted from the job description.
        candidate_text: Original CV and cover letter provided by the user.
        company_info: Research output that informs tailoring.
        next: Router-selected next action.
        generated_cv: Most recent CV produced by the pipeline.
        generated_cover_letter: Most recent cover letter produced by the pipeline.
        user_feedback: Free-form feedback provided after reviewing documents.
        user_input_message: Prompt shown when interrupting for user input.
    """
    messages: Annotated[list, add_messages]
    job_description_info: dict | None  # Extracted job description information
    candidate_text: dict | None  # CV and cover letter text (keys: 'cv', 'cover_letter')
    company_info: dict | None  # Company information from search (keys: 'company_description', 'remote_work', 'search_results')
    next: Literal["generate_cv", "generate_cover_letter", "user_input", "exit"] | None  # Next action to take
    generated_cv: str | None  # Generated CV text
    generated_cover_letter: str | None  # Generated cover letter text
    user_feedback: str | None  # User feedback for modifications
    user_input_message: str | None  # Message to display when requesting user input


def read_multiline_input(prompt: str = "You: ") -> str:
    """
    Read multi-line input from the user, handling long pasted text.
    
    This function allows users to paste long, multi-line text. It reads until:
    - EOF (Ctrl+D on Unix/Mac, Ctrl+Z on Windows) - works great for pasted text
    - Or two consecutive empty lines (press Enter twice)
    - Or type "END" on its own line
    
    Args:
        prompt: The prompt to display to the user
        
    Returns:
        str: The complete input text
    """
    print(prompt, end="", flush=True)
    lines = []
    empty_line_count = 0
    
    try:
        # Read line by line until EOF, two empty lines, or "END" marker
        while True:
            try:
                line = input()
                # Check if this is the "END" marker (must be on its own line, case-insensitive)
                if line.strip().upper() == "END" and len(lines) > 0:
                    break
                # Track consecutive empty lines
                if line.strip() == "":
                    empty_line_count += 1
                    if empty_line_count >= 2 and len(lines) > 0:
                        # Two empty lines = end of input
                        break
                else:
                    empty_line_count = 0
                lines.append(line)
            except EOFError:
                # User pressed Ctrl+D (Unix/Mac) or Ctrl+Z (Windows)
                # This is the normal way to finish pasting text
                break
    except KeyboardInterrupt:
        # User pressed Ctrl+C
        print("\nInput cancelled.")
        return ""
    
    result = '\n'.join(lines)
    return result









class MasterAgent:
    """Master agent that orchestrates the CV editing workflow using LangGraph."""
    
    def __init__(self):
        """
        Initialize the MasterAgent and compile the LangGraph workflow.

        Args:
            None

        Returns:
            None
        """
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
        user_input_agent = UserInputAgent()
        logger.debug("All agents instantiated successfully")

        # Add nodes to the graph
        logger.debug("Adding nodes to graph...")
        graph_builder.add_node("get_candidate_information", document_reader_agent.run)
        graph_builder.add_node("get_job_description", job_description_agent.run)
        graph_builder.add_node("get_company_information", search_agent.run)
        graph_builder.add_node("router", router_agent.run)
        graph_builder.add_node("generate_cv", cv_writer_agent.run)
        graph_builder.add_node("generate_cover_letter", cover_letter_writer_agent.run)
        graph_builder.add_node("user_input", user_input_agent.run)
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
            {
                "generate_cv": "generate_cv",
                "generate_cover_letter": "generate_cover_letter",
                "user_input": "user_input",
                "exit": END
            }
        )
        graph_builder.add_edge("generate_cv", "router")
        graph_builder.add_edge("generate_cover_letter", "router")
        graph_builder.add_edge("user_input", "router")
        logger.debug("All edges added to graph")

        # Create a checkpointer for persistence (required for interrupts)
        logger.debug("Creating MemorySaver checkpointer")
        checkpointer = MemorySaver()

        # Compile the graph with checkpointer
        logger.debug("Compiling graph with checkpointer...")
        self.graph = graph_builder.compile(checkpointer=checkpointer)
        logger.info("MasterAgent initialized successfully - graph compiled and ready")

        # Plot and save the graph to PNG
        try:
            logger.debug("Generating graph visualization...")
            image_dir = "images"
            os.makedirs(image_dir, exist_ok=True)
            self.graph.get_graph().draw_png(os.path.join(image_dir, "graph_visualization.png"))
            logger.info("Graph visualization saved to graph_visualization.png")
        except ImportError as e:
            logger.warning(f"Could not generate graph visualization: {e}")
            logger.warning("Install Graphviz system library first: brew install graphviz")
            logger.warning("Then install pygraphviz with:")
            logger.warning("  export CFLAGS=\"-I$(brew --prefix graphviz)/include\"")
            logger.warning("  export LDFLAGS=\"-L$(brew --prefix graphviz)/lib\"")
            logger.warning("  uv add pygraphviz")
        except Exception as e:
            logger.warning(f"Could not generate graph visualization: {e}")




    def run(self):
        """
        Stream the LangGraph workflow until completion or user exit.

        Args:
            None

        Returns:
            None
        """
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
            user_feedback=None,
            user_input_message=None,
            pending_user_input=None
        )
        logger.debug("Initial state created")

        iteration_count = 0
        should_exit = False
        just_resumed = False  # Track if we just resumed from an interrupt
        while True:
            try:
                if should_exit:
                    break
                    
                iteration_count += 1
                logger.info(f"Graph invocation #{iteration_count} - streaming graph")
                
                # Use stream() with stream_mode="values" to properly handle interrupts
                # After a resume, don't pass initial_state - let LangGraph use the checkpoint from config
                # This ensures we continue from where we left off, not restart from the beginning
                if just_resumed:
                    stream_input = None  # Use checkpoint, don't pass state
                    logger.debug("Streaming without initial state - using checkpoint from config (just resumed)")
                    just_resumed = False  # Reset flag
                else:
                    stream_input = initial_state
                    logger.debug(f"Streaming with initial state. Keys: {list(stream_input.keys())}")
                
                interrupt_occurred = False
                
                # Handle interrupt() calls that add __interrupt__ to state during streaming
                for state in self.graph.stream(stream_input, config, stream_mode="values"):
                    logger.debug(f"Stream state received. Keys: {list(state.keys()) if isinstance(state, dict) else 'N/A'}")
                    
                    # Check for interrupt() calls during streaming (adds __interrupt__ to state)
                    if isinstance(state, dict) and "__interrupt__" in state:
                        interrupt_occurred = True
                        interrupt_data = state["__interrupt__"]
                        logger.info("Graph execution interrupted via interrupt() - waiting for user input")
                        
                        # Extract interrupt message - interrupt() stores the message in __interrupt__
                        # The message is typically the first element if it's a tuple/list, or the value itself
                        interrupt_message = "Please provide your input:"
                        try:
                            if isinstance(interrupt_data, (tuple, list)) and len(interrupt_data) > 0:
                                interrupt_message = str(interrupt_data[0])
                            elif interrupt_data:
                                interrupt_message = str(interrupt_data)
                        except Exception as e:
                            logger.warning(f"Could not extract interrupt message: {e}. Using default message.")
                        
                        logger.debug(f"Interrupt message: {interrupt_message}")
                        print(f"\n\nAssistant: {interrupt_message}")
                        print("(Paste your text, then press Ctrl+D to finish, or press Enter twice, or type 'END' on a new line)")
                        
                        # Get user input
                        logger.debug("Waiting for user input...")
                        user_input = read_multiline_input("You: ")
                        logger.info(f"User input received: {len(user_input)} characters" + (f" (preview: {user_input[:100]}...)" if len(user_input) > 100 else f": {user_input}"))
                        
                        if user_input.lower() == "exit":
                            logger.info("User requested exit - terminating workflow")
                            print("Bye")
                            should_exit = True
                            break
                        
                        # Resume the graph with user input using Command
                        # Command(resume=value) passes the value to the interrupt() call, which returns it
                        logger.debug("Resuming graph execution with user input")
                        for resumed_state in self.graph.stream(Command(resume=user_input), config, stream_mode="values"):
                            initial_state = resumed_state
                            logger.debug(f"Resume stream state received. Keys: {list(resumed_state.keys()) if isinstance(resumed_state, dict) else 'N/A'}")
                            
                            # Check if there's another interrupt in the resumed stream
                            if isinstance(resumed_state, dict) and "__interrupt__" in resumed_state:
                                logger.debug("Another interrupt detected in resume stream")
                                break
                        
                        logger.debug("Graph resumed successfully")
                        just_resumed = True
                        break
                    
                    # Update state as we stream
                    initial_state = state
                
                if should_exit:
                    break
                
                # Check if the graph has reached the END node (exit condition)
                # When router sets next="exit", the graph routes to END and the stream completes
                graph_state = self.graph.get_state(config)
                logger.debug(f"Graph state after stream: next={graph_state.next if graph_state else 'None'}, has_values={bool(graph_state.values if graph_state else False)}")
                
                # Check if graph completed (reached END node)
                # In our graph, the only way to reach END is through exit, so if graph_state.next is None,
                # the graph has completed and we should exit
                if graph_state and graph_state.next is None:
                    # Graph has completed (reached END node)
                    logger.debug("Graph state indicates completion (next is None)")
                    # Check the state values to confirm it was an exit
                    if graph_state.values:
                        last_next = graph_state.values.get("next")
                        logger.debug(f"Last next value in state: {last_next}")
                        if last_next == "exit":
                            logger.info("Graph reached END node (exit) - terminating workflow")
                            should_exit = True
                            break
                    else:
                        # Graph completed but no state values - still exit since END was reached
                        # (In our graph design, END is only reached via exit)
                        logger.info("Graph reached END node - terminating workflow")
                        should_exit = True
                        break
                
                # Also check if the final state indicates exit (before graph reaches END)
                # This catches the case where router sets next="exit" but graph hasn't reached END yet
                if initial_state.get("next") == "exit":
                    logger.info("Exit detected in final state - terminating workflow")
                    should_exit = True
                    break
                
                if should_exit:
                    break
                
                # If no interrupt occurred, continue with normal flow
                if not interrupt_occurred:
                    # Display the response if there are messages
                    if initial_state.get("messages") and len(initial_state["messages"]) > 0:
                        last_message = initial_state["messages"][-1]
                        # Handle both dict and message object formats
                        if isinstance(last_message, dict):
                            content = last_message.get("content", "")
                        else:
                            content = getattr(last_message, "content", "")
                        
                        if content:
                            logger.debug(f"Assistant message to display: {content[:200]}..." if len(content) > 200 else f"Assistant message: {content}")
                            print(f"Assistant: {content}")
                    
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