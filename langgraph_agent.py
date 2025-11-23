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
from langgraph.types import interrupt

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
        next: Router-selected next action (draft_cv, draft_cover_letter, collect_user_input, exit).
        generated_cv: Most recent CV produced by the pipeline.
        generated_cover_letter: Most recent cover letter produced by the pipeline.
        user_feedback: Free-form feedback provided after reviewing documents.
        user_input_message: Prompt shown when interrupting for user input.
    """
    messages: Annotated[list, add_messages]
    job_description_info: dict | None  # Extracted job description information
    candidate_text: dict | None  # CV and cover letter text (keys: 'cv', 'cover_letter')
    company_info: dict | None  # Company information from search (keys: 'company_description', 'remote_work', 'search_results')
    current_node: str | None  # Current node in the graph
    next: Literal["draft_cv", "draft_cover_letter", "collect_user_input", "exit"] | None  # Next action to take
    generated_cv: str | None  # Generated CV text
    generated_cover_letter: str | None  # Generated cover letter text
    user_feedback: str | None  # User feedback for modifications
    user_input_message: str | None  # Message to display when requesting user input


def operations_on_state(state):
    logger.info(f"Current node: {state.get('current_node')}")
    if 'messages' in state:
        logger.info(state.get('messages'))

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
        graph_builder.add_node("load_candidate_documents", document_reader_agent.run)
        graph_builder.add_node("analyze_job_description", job_description_agent.run)
        graph_builder.add_node("research_company_context", search_agent.run)
        graph_builder.add_node("router", router_agent.run)
        graph_builder.add_node("draft_cv", cv_writer_agent.run)
        graph_builder.add_node("draft_cover_letter", cover_letter_writer_agent.run)
        graph_builder.add_node("collect_user_input", user_input_agent.run)
        logger.debug("All nodes added to graph")

        # Add edges to the graph
        logger.debug("Adding edges to graph...")
        graph_builder.add_edge(START, "load_candidate_documents")
        graph_builder.add_edge("load_candidate_documents", "analyze_job_description")
        graph_builder.add_edge("analyze_job_description", "research_company_context")
        graph_builder.add_edge("research_company_context", "router")

        graph_builder.add_conditional_edges(
            "router",
            lambda state: state.get("next"),
            {
                "draft_cv": "draft_cv",
                "draft_cover_letter": "draft_cover_letter",
                "collect_user_input": "collect_user_input",
                "exit": END
            }
        )
        graph_builder.add_edge("draft_cv", "router")
        graph_builder.add_edge("draft_cover_letter", "router")
        graph_builder.add_edge("collect_user_input", "router")
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
        )
        logger.debug("Initial state created")

        iteration_count = 0
        should_exit = False
        just_resumed = False  # Track if we just resumed from an interrupt
        
        stream_input = initial_state
        logger.debug(f"Streaming with initial state. Keys: {list(stream_input.keys())}")
                
               
                
                # Handle interrupt() calls that add __interrupt__ to state during streaming
        # result = self.graph.invoke(stream_input, config)
        # while result.get('__interrupt__')[0].value:
        #     human_input = read_multiline_input()
        #     result = self.graph.invoke(Command(resume=human_input), config=config)
        #     print(result.get('__interrupt__')[0].value)

        
        for state in self.graph.stream(stream_input, config, stream_mode="values"):
            operations_on_state(state)
            
            while '__interrupt__' in state:
                interrupt_data = state["__interrupt__"]
                print(interrupt_data)
                human_input = read_multiline_input()
                for state in self.graph.stream(Command(resume=human_input), config, stream_mode="values"):
                    operations_on_state(state)

            




if __name__ == "__main__":
    agent = MasterAgent()
    agent.run()