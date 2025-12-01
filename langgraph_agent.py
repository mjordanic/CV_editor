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

# Silence noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

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
from agents.critique import CritiqueAgent
from agents.rag_agent import ExperienceRetrievalAgent




class State(TypedDict, total=False):
    """
    Shared LangGraph state tracking conversation context and generation artifacts.

    Keys:
        messages: Rolling chat history consumed by LangGraph nodes.
        job_description_info: Structured information extracted from the job description.
        candidate_text: Original CV and cover letter provided by the user.
        company_info: Research output that informs tailoring.
        next: Router-selected next action (draft_cv, draft_cover_letter, collect_user_input, exit).
        generated_cv: Most recent CV produced by the pipeline (clean CV without explanations).
        generated_cover_letter: Most recent cover letter produced by the pipeline.
        user_feedback: Free-form feedback provided after reviewing documents.
        user_input_message: Prompt shown when interrupting for user input.
        cv_critique: Critique results for the CV (quality score, feedback, improvement instructions).
        cover_letter_critique: Critique results for the cover letter (quality score, feedback, improvement instructions).
        cv_needs_refinement: Whether the CV needs to be refined based on critique feedback.
        cover_letter_needs_refinement: Whether the cover letter needs to be refined based on critique feedback.
        cv_critique_improvement_instructions: Improvement instructions from critique for CV refinement.
        cover_letter_critique_improvement_instructions: Improvement instructions from critique for cover letter refinement.
        cv_needs_critique: Flag to indicate if CV needs critiquing (after generation or refinement).
        cover_letter_needs_critique: Flag to indicate if cover letter needs critiquing (after generation or refinement).
        cv_refinement_count: Number of times CV has been refined based on critique (max 1).
        cover_letter_refinement_count: Number of times cover letter has been refined based on critique (max 1).
    """
    messages: Annotated[list, add_messages]
    job_description_info: dict | None  # Extracted job description information
    candidate_text: dict | None  # CV and cover letter text (keys: 'cv', 'cover_letter')
    company_info: dict | None  # Company information from search (keys: 'company_description', 'remote_work', 'search_results')
    current_node: str | None  # Current node in the graph
    next: Literal["draft_cv", "draft_cover_letter", "collect_user_input", "exit"] | None  # Next action to take
    generated_cv: str | None  # Generated CV text (clean, without explanations)
    generated_cover_letter: str | None  # Generated cover letter text
    user_feedback: str | None  # User feedback for modifications
    user_input_message: str | None  # Message to display when requesting user input
    cv_critique: dict | None  # Critique results for CV
    cover_letter_critique: dict | None  # Critique results for cover letter
    cv_needs_refinement: bool | None  # Whether CV needs refinement based on critique
    cover_letter_needs_refinement: bool | None  # Whether cover letter needs refinement based on critique
    cv_critique_improvement_instructions: str | None  # Improvement instructions for CV from critique
    cover_letter_critique_improvement_instructions: str | None  # Improvement instructions for cover letter from critique
    cv_needs_critique: bool | None  # Flag to indicate if CV needs critiquing
    cover_letter_needs_critique: bool | None  # Flag to indicate if cover letter needs critiquing
    cv_refinement_count: int | None  # Number of times CV has been refined based on critique (max 2 if quality improves)
    cover_letter_refinement_count: int | None  # Number of times cover letter has been refined based on critique (max 2 if quality improves)
    cv_previous_quality_score: int | None  # Previous quality score from critique (for comparison)
    cover_letter_previous_quality_score: int | None  # Previous quality score from critique (for comparison)
    cv_history: list[dict] | None  # History of generated CVs with scores and iteration numbers
    cover_letter_history: list[dict] | None  # History of generated cover letters with scores and iteration numbers
    relevant_experience: str | None  # Retrieved relevant experience from portfolio (RAG)


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
        
        # Load configuration once at startup
        logger.debug("Loading agent configuration...")
        from config.config_loader import load_config, get_agent_config
        
        # Load full config to get workflow settings
        full_config = load_config()
        workflow_config = full_config.get('workflow', {})
        self.max_refinements = workflow_config.get('max_refinements', 2)
        self.quality_improvement_threshold = workflow_config.get('quality_improvement_threshold', 5)
        logger.debug(f"Workflow settings - max_refinements: {self.max_refinements}, quality_improvement_threshold: {self.quality_improvement_threshold}")
        
        # Load agent configurations
        router_config = get_agent_config('router')
        cv_writer_config = get_agent_config('cv_writer')
        cover_letter_config = get_agent_config('cover_letter_writer')
        critique_config = get_agent_config('critique')
        job_desc_config = get_agent_config('job_description')
        search_config = get_agent_config('search')
        logger.debug("Configuration loaded for all agents")
        
        # Build the graph
        logger.debug("Creating StateGraph")
        graph_builder = StateGraph(State)

        # Instantiate agents with configuration from YAML
        logger.debug("Instantiating agents...")
        search_agent = SearchAgent(
            model=search_config['model'],
            temperature=search_config['temperature']
        )
        document_reader_agent = DocumentReaderAgent()
        job_description_agent = JobDescriptionAgent(
            model=job_desc_config['model'],
            temperature=job_desc_config['temperature']
        )
        router_agent = RouterAgent(
            model=router_config['model'],
            temperature=router_config['temperature']
        )
        cv_writer_agent = CVWriterAgent(
            model=cv_writer_config['model'],
            temperature=cv_writer_config['temperature'],
            filter_model=cv_writer_config['filter_model'],
            filter_temperature=cv_writer_config['filter_temperature']
        )
        cover_letter_writer_agent = CoverLetterWriterAgent(
            model=cover_letter_config['model'],
            temperature=cover_letter_config['temperature'],
            filter_model=cover_letter_config['filter_model'],
            filter_temperature=cover_letter_config['filter_temperature']
        )
        user_input_agent = UserInputAgent()
        critique_agent = CritiqueAgent(
            model=critique_config['model'],
            temperature=critique_config['temperature'],
            quality_threshold=critique_config['quality_threshold']
        )
        experience_retrieval_agent = ExperienceRetrievalAgent()
        logger.debug("All agents instantiated successfully")


        # Add nodes to the graph
        logger.debug("Adding nodes to graph...")
        graph_builder.add_node("load_candidate_documents", document_reader_agent.run)
        graph_builder.add_node("analyze_job_description", job_description_agent.run)
        graph_builder.add_node("research_company_context", search_agent.run)
        graph_builder.add_node("retrieve_experience", experience_retrieval_agent.run)
        graph_builder.add_node("router", router_agent.run)
        graph_builder.add_node("draft_cv", cv_writer_agent.run)
        graph_builder.add_node("draft_cover_letter", cover_letter_writer_agent.run)
        graph_builder.add_node("critique_cv", critique_agent.run_cv)
        graph_builder.add_node("critique_cover_letter", critique_agent.run_cover_letter)
        graph_builder.add_node("collect_user_input", user_input_agent.run)
        graph_builder.add_node("finalize_cv", cv_writer_agent.finalize_best_version)
        graph_builder.add_node("finalize_cover_letter", cover_letter_writer_agent.finalize_best_version)
        logger.debug("All nodes added to graph")

        # Add edges to the graph
        logger.debug("Adding edges to graph...")
        graph_builder.add_edge(START, "load_candidate_documents")
        graph_builder.add_edge("load_candidate_documents", "analyze_job_description")
        graph_builder.add_edge("analyze_job_description", "research_company_context")
        graph_builder.add_edge("research_company_context", "retrieve_experience")
        graph_builder.add_edge("retrieve_experience", "router")
        
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
        graph_builder.add_edge("draft_cv", "critique_cv")
        graph_builder.add_edge("draft_cover_letter", "critique_cover_letter")
        
        # Conditional routing for CV critique
        def route_cv_critique(state):
            # Check if refinement is needed AND allowed (count < max or quality improved)
            needs_refinement = state.get("cv_needs_refinement", False)
            refinement_count = state.get("cv_refinement_count", 0)
            
            # Check for quality improvement
            cv_critique = state.get("cv_critique")
            current_score = cv_critique.get("quality_score") if cv_critique else None
            previous_score = state.get("cv_previous_quality_score")
            
            quality_improved = False
            if current_score is not None and previous_score is not None:
                improvement = current_score - previous_score
                if improvement >= self.quality_improvement_threshold:
                    quality_improved = True
                    logger.info(f"CV quality improved by {improvement} points - allowing refinement despite count")

            # Allow refinement if needed AND (count < max OR quality improved significantly)
            if needs_refinement and (refinement_count < self.max_refinements or quality_improved):
                logger.info(f"Routing critique_cv -> draft_cv (Refinement count: {refinement_count}, Quality improved: {quality_improved})")
                return "draft_cv"
            else:
                logger.info(f"Routing critique_cv -> finalize_cv (Refinement count: {refinement_count}, Needs refinement: {needs_refinement}, Quality improved: {quality_improved})")
                return "finalize_cv"

        graph_builder.add_conditional_edges(
            "critique_cv",
            route_cv_critique,
            {
                "draft_cv": "draft_cv",
                "finalize_cv": "finalize_cv"
            }
        )

        # Conditional routing for Cover Letter critique
        def route_cover_letter_critique(state):
            needs_refinement = state.get("cover_letter_needs_refinement", False)
            refinement_count = state.get("cover_letter_refinement_count", 0)
            
            # Check for quality improvement
            cl_critique = state.get("cover_letter_critique")
            current_score = cl_critique.get("quality_score") if cl_critique else None
            previous_score = state.get("cover_letter_previous_quality_score")
            
            quality_improved = False
            if current_score is not None and previous_score is not None:
                improvement = current_score - previous_score
                if improvement >= self.quality_improvement_threshold:
                    quality_improved = True
                    logger.info(f"Cover letter quality improved by {improvement} points - allowing refinement despite count")
            
            # Allow refinement if needed AND (count < max OR quality improved significantly)
            if needs_refinement and (refinement_count < self.max_refinements or quality_improved):
                logger.info(f"Routing critique_cover_letter -> draft_cover_letter (Refinement count: {refinement_count}, Quality improved: {quality_improved})")
                return "draft_cover_letter"
            else:
                logger.info(f"Routing critique_cover_letter -> finalize_cover_letter (Refinement count: {refinement_count}, Needs refinement: {needs_refinement}, Quality improved: {quality_improved})")
                return "finalize_cover_letter"

        graph_builder.add_conditional_edges(
            "critique_cover_letter",
            route_cover_letter_critique,
            {
                "draft_cover_letter": "draft_cover_letter",
                "finalize_cover_letter": "finalize_cover_letter"
            }
        )
        
        graph_builder.add_edge("collect_user_input", "router")
        graph_builder.add_edge("finalize_cv", "router")
        graph_builder.add_edge("finalize_cover_letter", "router")
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
            cv_critique=None,
            cover_letter_critique=None,
            cv_needs_refinement=None,
            cover_letter_needs_refinement=None,
            cv_critique_improvement_instructions=None,
            cover_letter_critique_improvement_instructions=None,
            cv_needs_critique=None,
            cover_letter_needs_critique=None,
            cv_refinement_count=0,
            cover_letter_refinement_count=0,
            cv_history=[],
            cover_letter_history=[],
            relevant_experience=None,
        )
        logger.debug("Initial state created")

        
        stream_input = initial_state
        logger.debug(f"Streaming with initial state. Keys: {list(stream_input.keys())}")
                
               
        # This is a simplified version of the streaming loop that handles interrupt() calls
        # It uses the invoke() method to run the graph and handle interrupt() calls
        result = self.graph.invoke(stream_input, config)
        while result.get('__interrupt__') and result.get('__interrupt__')[0].value:
            interrupt_data = result["__interrupt__"]    
            interrupt_message = interrupt_data[0].value["message"] if interrupt_data and len(interrupt_data) > 0 else "Please provide your input:"
            logger.info(f"\n\n{interrupt_message}")
            logger.info("(When done type 'END' on a new line or hit ENTER twice)")
            
            human_input = read_multiline_input()
            
            result = self.graph.invoke(Command(resume=human_input), config=config)

        ##########################################################################################################
        # This is another way to run the graph and handle interrupt() calls. It uses the stream() 
        # method to stream the graph. It is more verbose as it streams the state object at each step. State
        # can be accessed and inspected at each step. Currently there is a dummy function operations_on_state() 
        # that prints the messages.
        
        # for state in self.graph.stream(stream_input, config, stream_mode="values"):
        #     operations_on_state(state)
            
        #     while '__interrupt__' in state:
        #         interrupt_data = state["__interrupt__"]
        #         # __interrupt__ is a list, and each item has a .value attribute containing the interrupt data
        #         interrupt_message = interrupt_data[0].value["message"] if interrupt_data and len(interrupt_data) > 0 else "Please provide your input:"
        #         logger.info(f"\n\n{interrupt_message}")
        #         logger.info("(When done type 'END' on a new line or hit ENTER twice)")
        #         human_input = read_multiline_input()
        #         for state in self.graph.stream(Command(resume=human_input), config, stream_mode="values"):
        #             operations_on_state(state)

            




if __name__ == "__main__":
    agent = MasterAgent()
    agent.run()