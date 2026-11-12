import logging
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from config.config_loader import local_model_kwargs
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

def _embeddings_for(model: str):
    """
    Embeddings from the same provider as the chat model: a local Ollama embedding model when running
    against Ollama (no OpenAI key needed; CV_EDITOR_EMBEDDING_MODEL, default nomic-embed-text),
    OpenAI's text-embedding-3-small otherwise. Chat models such as gemma4 cannot embed in Ollama.
    If the embedding model is not pulled, indexing fails with a warning and RAG is skipped.
    """
    if model and model.startswith("ollama:"):
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=os.environ.get("CV_EDITOR_EMBEDDING_MODEL", "nomic-embed-text"))
    return OpenAIEmbeddings(model="text-embedding-3-small")


# Where the user's own portfolio goes. It is optional, unique to each person and never shipped with
# the repository (data/ is git-ignored): without it the session simply has no portfolio.
DEFAULT_PORTFOLIO_PATH = os.path.join("data", "portfolio", "portfolio.txt")

# A portfolio up to this size goes to the writers whole: portfolios are usually a few KB, a top-k cut
# can silently drop the item that matters, and the writer can judge relevance itself. Only a larger
# one is indexed (one paid embedding call) and searched.
FULL_PORTFOLIO_CHARS = 12000

NO_PORTFOLIO = "No portfolio provided."


class ExperienceRetrievalAgent:
    """Agent that gives the writers the candidate's portfolio: whole, or the relevant items of a large one (RAG)."""
    
    def __init__(
        self, 
        portfolio_path: str = DEFAULT_PORTFOLIO_PATH,
        model: str = "openai:gpt-5-nano",
        temperature: float = 0.0,
        similarity_threshold: float = 0.5,
        full_text_limit: int = FULL_PORTFOLIO_CHARS,
    ):
        """
        Initialize the ExperienceRetrievalAgent.
        
        Args:
            portfolio_path: Path to the user's portfolio text file (read only if the user put one there)
            model: LLM model for relevance verification (large portfolios only)
            temperature: Temperature for verification LLM
            similarity_threshold: L2 distance threshold (lower is better)
            full_text_limit: Portfolios up to this many characters go to the writers whole, without retrieval
        """
        logger.info(f"Initializing ExperienceRetrievalAgent - model: {model}, threshold: {similarity_threshold}")
        self.portfolio_path = portfolio_path
        self.model = model
        self.similarity_threshold = similarity_threshold
        self.full_text_limit = full_text_limit
        self.llm = init_chat_model(model, temperature=temperature, **local_model_kwargs(model))
        self.embeddings = None  # created only when a large portfolio is indexed
        self.vectorstore = None
        # full portfolio: the critique checks drafts against it for unsupported claims
        self.portfolio_text = self._read_portfolio()
        logger.debug("ExperienceRetrievalAgent initialized")

    def _read_portfolio(self) -> str:
        """The user's portfolio text, or "" (logged as 'no portfolio') when they did not supply one."""
        if not self.portfolio_path or not os.path.isfile(self.portfolio_path):
            logger.info(f"No portfolio: {self.portfolio_path} does not exist (a portfolio is optional)")
            return ""
        try:
            with open(self.portfolio_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except Exception as e:
            logger.warning(f"No portfolio: could not read {self.portfolio_path} ({type(e).__name__}: {e})")
            return ""
        if not text:
            logger.info(f"No portfolio: {self.portfolio_path} is empty")
            return ""
        logger.info(f"Portfolio read from {self.portfolio_path} ({len(text)} characters)")
        return text

    def _index_portfolio(self):
        """Create the vector index of the portfolio (one paid embedding call)."""
        if not self.portfolio_text:
            return
        try:
            # Split content into chunks (assuming chunks are separated by double newlines)
            raw_chunks = [c.strip() for c in self.portfolio_text.split("\n\n") if c.strip()]
            
            source = os.path.basename(self.portfolio_path)
            documents = [Document(page_content=chunk, metadata={"source": source}) for chunk in raw_chunks]
            
            if documents:
                if self.embeddings is None:
                    self.embeddings = _embeddings_for(self.model)
                self.vectorstore = FAISS.from_documents(documents, self.embeddings)
                logger.info(f"Indexed {len(documents)} portfolio items.")
            else:
                logger.warning("No documents found in portfolio file.")
                
        except Exception as e:
            logger.warning(
                f"Could not build the portfolio vector index from {self.portfolio_path} "
                f"({type(e).__name__}: {e}). RAG is disabled for this run: documents will be "
                f"written without retrieved portfolio experience."
            )

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
            job_description=job_description if len(job_description) <= 2000 else job_description[:2000] + "... (truncated)",
            chunk_content=chunk_content
        )
        try:
            response = self.llm.invoke(prompt)
            answer = response.content.strip().upper() if hasattr(response, 'content') else str(response).strip().upper()
            
            # startswith: "NO ... YES ..." contains YES too
            is_relevant = answer.lstrip("*'\" ").startswith("YES")
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
        Give the writers the portfolio: the whole text when it is small, otherwise the items
        retrieved for this job description.
        
        Args:
            state: The state dictionary containing job_description_info
            
        Returns:
            dict: Updated state with relevant_experience (what the writers see) and portfolio_text
        """
        logger.info("ExperienceRetrievalAgent.run() called")

        if not self.portfolio_text:
            logger.info("No portfolio - the writers get none")
            return {"relevant_experience": "", "portfolio_text": ""}

        if len(self.portfolio_text) <= self.full_text_limit:
            logger.info(f"Portfolio is small ({len(self.portfolio_text)} characters) - giving it to the writers whole")
            return {"relevant_experience": self.portfolio_text, "portfolio_text": self.portfolio_text}

        job_description_info = state.get("job_description_info")
        if not job_description_info:
            logger.warning("No job description info found in state. Skipping retrieval.")
            return {"relevant_experience": "", "portfolio_text": self.portfolio_text}

        if self.vectorstore is None:
            self._index_portfolio()  # only now: no paid call before the user has typed anything
            
        # Use the full job description text for retrieval
        job_description_text = job_description_info.get("job_description", "")
        if not job_description_text:
             # Fallback to candidate requirements if full text is missing (unlikely)
             job_description_text = job_description_info.get("candidate_minimal_requirements", "")
        
        relevant_experience = self.retrieve_relevant_experience(job_description_text)
        if relevant_experience and self.vectorstore is not None:
            relevant_experience = (
                "(The candidate's portfolio is long; these are the items most relevant to this job.)\n\n"
                + relevant_experience
            )
        
        # Write debug info
        debug_content = "RETRIEVED RELEVANT EXPERIENCE:\n"
        debug_content += "-" * 80 + "\n"
        debug_content += relevant_experience
        debug_content += "\n\n"
        write_to_debug(debug_content, "RAG RETRIEVAL DEBUG INFO")
        
        return {
            "relevant_experience": relevant_experience,
            "portfolio_text": self.portfolio_text,
        }
