import logging
import os
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from debug_utils import write_to_debug

logger = logging.getLogger(__name__)

RELEVANCE_VERIFICATION_PROMPT = (
    "You are an expert recruiter verifying the relevance of a candidate's experience to a specific job description.\n\n"
    "**Job Description:**\n"
    "{job_description}\n\n"
    "**Candidate Experience Chunk:**\n"
    "{chunk_content}\n\n"
    "**Task:**\n"
    "Determine if this specific experience chunk is RELEVANT to the job description.\n"
    "Relevant means it demonstrates skills, experience, or qualifications that would help the candidate get this job.\n"
    "Irrelevant means it is unrelated (e.g., a hobby, a different field, or generic fluff).\n\n"
    "Answer ONLY with 'YES' or 'NO'."
)

class ExperienceRetrievalAgent:
    """Agent for retrieving relevant experience from a portfolio using RAG."""
    
    def __init__(
        self, 
        portfolio_path: str = "data/portfolio/portfolio.txt",
        model: str = "openai:gpt-5-nano",
        temperature: float = 0.0,
        similarity_threshold: float = 0.5
    ):
        """
        Initialize the ExperienceRetrievalAgent.
        
        Args:
            portfolio_path: Path to the portfolio text file
            model: LLM model for relevance verification
            temperature: Temperature for verification LLM
            similarity_threshold: L2 distance threshold (lower is better)
        """
        logger.info(f"Initializing ExperienceRetrievalAgent - model: {model}, threshold: {similarity_threshold}")
        self.portfolio_path = portfolio_path
        self.similarity_threshold = similarity_threshold
        
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = init_chat_model(model, temperature=temperature)
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

    def _verify_relevance(self, chunk_content: str, job_description: str) -> bool:
        """
        Verify if the retrieved chunk is actually relevant to the job description using an LLM.
        
        Args:
            chunk_content: The content of the retrieved chunk
            job_description: The job description text
            
        Returns:
            bool: True if relevant, False otherwise
        """
        prompt = RELEVANCE_VERIFICATION_PROMPT.format(
            job_description=job_description[:2000] + "... (truncated)",
            chunk_content=chunk_content
        )
        try:
            response = self.llm.invoke(prompt)
            answer = response.content.strip().upper() if hasattr(response, 'content') else str(response).strip().upper()
            
            is_relevant = "YES" in answer
            logger.debug(f"Relevance verification: {answer} (Relevant: {is_relevant})")
            return is_relevant
        except Exception as e:
            logger.warning(f"Error verifying relevance: {e}. Assuming relevant.")
            return True

    def retrieve_relevant_experience(self, job_description: str, k: int = 5) -> str:
        """
        Retrieve the top-k most relevant experience chunks for a given job description.
        Applies vector similarity threshold and LLM verification.
        
        Args:
            job_description: The job description text
            k: Number of chunks to retrieve (initial pool)
            
        Returns:
            str: Concatenated relevant experience text
        """
        if not self.vectorstore:
            logger.warning("Vectorstore not initialized. Cannot retrieve experience.")
            return ""
            
        logger.info(f"Retrieving top {k} candidates for relevance filtering...")
        try:
            # Search for relevant documents with scores (L2 distance: lower is better)
            docs_and_scores = self.vectorstore.similarity_search_with_score(job_description, k=k)
            
            valid_docs = []
            for doc, score in docs_and_scores:
                logger.debug(f"Candidate chunk score: {score} (Threshold: {self.similarity_threshold})")
                
                # Filter by similarity threshold
                if score > self.similarity_threshold:
                    logger.debug(f"Chunk rejected by threshold ({score} > {self.similarity_threshold})")
                    continue
                
                # Verify with LLM
                if self._verify_relevance(doc.page_content, job_description):
                    valid_docs.append(doc)
                else:
                    logger.debug("Chunk rejected by LLM verification")
            
            if not valid_docs:
                logger.info("No relevant experience found after filtering.")
                return "No highly relevant experience found in portfolio."

            # Format the results
            retrieved_text = ""
            for i, doc in enumerate(valid_docs):
                source = os.path.basename(doc.metadata.get("source", "unknown"))
                retrieved_text += f"RELEVANT EXPERIENCE {i+1} (Source: {source}):\n{doc.page_content}\n\n"
            
            logger.info(f"Retrieved {len(valid_docs)} verified relevant items.")
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
