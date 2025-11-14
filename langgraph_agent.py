from dotenv import load_dotenv

# Load environment variables BEFORE importing agents
load_dotenv()

# Set up logging
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

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







graph_builder = StateGraph(State)

search_agent = SearchAgent()
document_reader_agent = DocumentReaderAgent()
job_description_agent = JobDescriptionAgent()
router_agent = RouterAgent()
cv_writer_agent = CVWriterAgent()
cover_letter_writer_agent = CoverLetterWriterAgent()


graph_builder.add_node("get_candidate_information", document_reader_agent.run)
graph_builder.add_node("get_job_description", job_description_agent.run)
graph_builder.add_node("get_company_information", search_agent.run)
graph_builder.add_node("router", router_agent.run)
graph_builder.add_node("generate_cv", cv_writer_agent.run)
graph_builder.add_node("generate_cover_letter", cover_letter_writer_agent.run)


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

# Create a checkpointer for persistence (required for interrupts)
checkpointer = MemorySaver()

# Compile the graph with checkpointer
graph = graph_builder.compile(checkpointer=checkpointer)


def run_langgraph_agent():
    # Use a thread_id to maintain state across invocations
    thread_id = "main-thread"
    config = {"configurable": {"thread_id": thread_id}}
    
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

    while True:
        try:
            # Invoke the graph with the current state
            result = graph.invoke(initial_state, config)
            
            # Check if there's an interrupt
            if hasattr(result, "__interrupt__") and result.__interrupt__:
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
                
                print(f"Assistant: {message}")
                
                # Get user input
                user_input = input("You: ")
                if user_input.lower() == "exit":
                    print("Bye")
                    break
                
                # Resume the graph with user input using Command
                result = graph.invoke(Command(resume=user_input), config)
            
            # Display the response if there are messages
            if result.get("messages") and len(result["messages"]) > 0:
                last_message = result["messages"][-1]
                # Handle both dict and message object formats
                if isinstance(last_message, dict):
                    content = last_message.get("content", "")
                else:
                    content = getattr(last_message, "content", "")
                
                if content:
                    print(f"Assistant: {content}")
            
            # Update initial_state for next iteration (though with checkpointer, this may not be needed)
            initial_state = result
            
        except KeyboardInterrupt:
            print("\nBye")
            break
        except Exception as e:
            print(f"Error: {e}")
            break


if __name__ == "__main__":
    run_langgraph_agent()