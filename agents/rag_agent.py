import logging
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from debug_utils import write_to_debug

logger = logging.getLogger(__name__)

class ExperienceRetrievalAgent:
    """Agent for retrieving relevant experience from a portfolio using RAG."""
    
    def __init__(self, portfolio_path: str = "data/portfolio.txt"):
        """
        Initialize the ExperienceRetrievalAgent.
        
        Args:
            portfolio_path: Path to the portfolio text file
        """
        logger.info("Initializing ExperienceRetrievalAgent...")
        self.portfolio_path = portfolio_path
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = None
        
        # Load and index the portfolio on initialization
        self._index_portfolio()
        logger.debug("ExperienceRetrievalAgent initialized")
    
    def _index_portfolio(self):
        """Load portfolio data and create vector index."""
        if not os.path.exists(self.portfolio_path):
            logger.warning(f"Portfolio file not found at {self.portfolio_path}. RAG will not work.")
            return
            
        try:
            with open(self.portfolio_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Split content into chunks (assuming chunks are separated by double newlines)
            # In a real app, we'd use a smarter splitter, but this works for our format
            raw_chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
            
            documents = [Document(page_content=chunk) for chunk in raw_chunks]
            
            if documents:
                self.vectorstore = FAISS.from_documents(documents, self.embeddings)
                logger.info(f"Indexed {len(documents)} portfolio items.")
            else:
                logger.warning("No documents found in portfolio file.")
                
        except Exception as e:
            logger.error(f"Error indexing portfolio: {e}")

    def retrieve_relevant_experience(self, job_description: str, k: int = 3) -> str:
        """
        Retrieve the top-k most relevant experience chunks for a given job description.
        
        Args:
            job_description: The job description text
            k: Number of chunks to retrieve
            
        Returns:
            str: Concatenated relevant experience text
        """
        if not self.vectorstore:
            logger.warning("Vectorstore not initialized. Cannot retrieve experience.")
            return ""
            
        logger.info(f"Retrieving top {k} relevant experiences...")
        try:
            # Search for relevant documents
            docs = self.vectorstore.similarity_search(job_description, k=k)
            
            # Format the results
            retrieved_text = "\n\n".join([f"RELEVANT EXPERIENCE {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
            
            logger.info(f"Retrieved {len(docs)} relevant items.")
            return retrieved_text
            
        except Exception as e:
            logger.error(f"Error retrieving experience: {e}")
            return ""

    def run(self, state):
        """
        Main method to retrieve relevant experience based on the job description.
        
        Args:
            state: The state dictionary containing job_description_info
            
        Returns:
            dict: Updated state with relevant_experience
        """
        logger.info("ExperienceRetrievalAgent.run() called")
        
        job_description_info = state.get("job_description_info")
        if not job_description_info:
            logger.warning("No job description info found in state. Skipping retrieval.")
            return {"relevant_experience": ""}
            
        # Use the full job description text for retrieval
        job_description_text = job_description_info.get("job_description", "")
        if not job_description_text:
             # Fallback to candidate requirements if full text is missing (unlikely)
             job_description_text = job_description_info.get("candidate_minimal_requirements", "")
        
        relevant_experience = self.retrieve_relevant_experience(job_description_text)
        
        # Write debug info
        debug_content = "RETRIEVED RELEVANT EXPERIENCE:\n"
        debug_content += "-" * 80 + "\n"
        debug_content += relevant_experience
        debug_content += "\n\n"
        write_to_debug(debug_content, "RAG RETRIEVAL DEBUG INFO")
        
        return {
            "relevant_experience": relevant_experience,
            "current_node": "retrieve_experience"
        }
