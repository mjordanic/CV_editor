from pypdf import PdfReader
import os
import logging
from typing import Optional, Dict
from langgraph.types import interrupt
from debug_utils import write_to_debug

logger = logging.getLogger(__name__)

# File types a pasted path is expected to have; anything else that is one line and starts like a
# path (/, ~, ./, C:\) is also treated as a path.
_DOCUMENT_SUFFIXES = (".txt", ".pdf", ".md", ".text")


def read_document_file(path: str) -> str:
    """Text of a .pdf (via pypdf) or a text file; undecodable bytes are replaced, not fatal."""
    if path.lower().endswith(".pdf"):
        return "\n".join((page.extract_text() or "") for page in PdfReader(path).pages)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def looks_like_path(text: str) -> bool:
    """Whether a one-line reply is meant as a file path rather than pasted document text."""
    if "\n" in text or len(text) > 400:
        return False
    if text.startswith(("/", "~", "./", "../", ".\\")) or (len(text) > 2 and text[1:3] in (":\\", ":/")):
        return True
    return text.lower().endswith(_DOCUMENT_SUFFIXES) and len(text.split()) <= 3


def ask_for_document(message: str, what: str) -> str:
    """
    Ask the user (via interrupt) to paste a document or give a path to it, until there is text.

    Only deterministic checks run between the interrupts (no model calls), so when LangGraph
    replays the node on resume, every interrupt gets the same answer it got the first time.
    An empty reply, a missing or unreadable file, or a file without text is not accepted: the
    user is told what was wrong and asked again, instead of the session going on with nothing.

    Args:
        message: The first question to show.
        what: What is being asked for, for the follow-up messages (e.g. "job ad", "CV").

    Returns:
        str: The document text (stripped).
    """
    how = f"Please paste the text of the {what}, or the path to a .txt or .pdf file that holds it."
    while True:
        reply = str(interrupt({"message": message, "required": True}) or "").strip()
        if not reply:
            message = f"I did not receive anything. {how}"
            continue
        path = os.path.expanduser(reply.strip("'\""))
        if os.path.isfile(path):
            try:
                text = read_document_file(path).strip()
            except Exception as e:  # unreadable or damaged file: ask again instead of ending the session
                logger.warning(f"Could not read {what} file {path}: {type(e).__name__}: {e}")
                message = f"I could not read the file '{reply}' ({type(e).__name__}). {how}"
                continue
            if not text:
                message = (f"The file '{reply}' has no readable text (for a PDF, it may be a scanned image). {how}")
                continue
            logger.info(f"Read {what} from file {path} - {len(text)} characters")
            return text
        if looks_like_path(reply):
            message = f"I could not find the file '{reply}'. {how}"
            continue
        return reply


class DocumentReaderAgent:
    """Agent for reading CV and cover letter documents from the CV folder."""
    
    def __init__(self, cv_folder: str = "data/CV"):
        """
        Initialize the DocumentReaderAgent.
        
        Args:
            cv_folder: Path to the folder containing CV and cover letter documents
        """
        logger.info(f"Initializing DocumentReaderAgent with cv_folder: {cv_folder}")
        self.cv_folder = cv_folder
        logger.debug("DocumentReaderAgent initialized")
    
    def read_pdf(self, file_path: str) -> str:
        """
        Read text content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            str: Extracted text from all pages of the PDF
        """
        logger.debug(f"Reading PDF file: {file_path}")
        reader = PdfReader(file_path)
        text_parts = []
        for page in reader.pages:
            text_parts.append(page.extract_text())
        text = "\n".join(text_parts)
        logger.info(f"PDF read successfully - extracted {len(text)} characters from {len(text_parts)} pages")
        return text
    
    def fetch_document_text(self, document_name: str) -> Optional[str]:
        """
        Fetches document text from the CV folder.
        
        Args:
            document_name: Name of the document (e.g., "CV", "cover_letter")
        
        Returns:
            str: The content of the document file (txt or pdf), or None if not found
        """
        logger.debug(f"Fetching document: {document_name}")
        doc_txt_path = os.path.join(self.cv_folder, f"{document_name}.txt")
        doc_pdf_path = os.path.join(self.cv_folder, f"{document_name}.pdf")
        
        # Check if .txt file exists
        if os.path.exists(doc_txt_path):
            logger.debug(f"Found .txt file: {doc_txt_path}")
            with open(doc_txt_path, "r", encoding="utf-8") as f:
                content = f.read()
                logger.info(f"Document {document_name} read from .txt - {len(content)} characters")
                return content
        
        # Check if .pdf file exists
        if os.path.exists(doc_pdf_path):
            logger.debug(f"Found .pdf file: {doc_pdf_path}")
            return self.read_pdf(doc_pdf_path)
        
        # If neither exists, return None
        logger.warning(f"Document {document_name} not found in {self.cv_folder} (checked .txt and .pdf)")
        return None
    
    def read_cv(self) -> Optional[str]:
        """
        Read the CV document if available.
        
        Returns:
            str: CV content if found, None otherwise
        """
        return self.fetch_document_text("CV")
    
    def read_cover_letter(self) -> Optional[str]:
        """
        Read the cover letter document if available.
        
        Returns:
            str: Cover letter content if found, None otherwise
        """
        return self.fetch_document_text("cover_letter")
    
    def read_all_documents(self) -> Dict[str, Optional[str]]:
        """
        Read both CV and cover letter documents if available.
        
        Returns:
            Dict with keys 'cv' and 'cover_letter', containing the document
            content if found, or None if not found
        """
        return {
            "cv": self.read_cv(),
            "cover_letter": self.read_cover_letter()
        }
    
    def run(self, state=None) -> Dict[str, Dict[str, Optional[str]]]:
        """
        Main method to read all available documents.
        
        Args:
            state: Optional state dictionary (for LangGraph compatibility)
        
        Returns:
            Dict with 'candidate_text' key containing a dict with keys 'cv' and 'cover_letter',
            containing the document content if found, or None if not found
        """
        logger.info("DocumentReaderAgent.run() called - reading all documents")
        documents = self.read_all_documents()
        if not (documents.get("cv") or "").strip():
            # Without a CV the writer builds one from the job ad and the portfolio alone, with
            # placeholders such as [YOUR NAME]; ask for it instead.
            logger.warning(f"No CV with text in {self.cv_folder} - asking the user for it")
            documents["cv"] = ask_for_document(
                f"I could not find your CV in {self.cv_folder} (CV.txt or CV.pdf), or it has no text. "
                "Please paste your CV, or the path to a .txt or .pdf file that holds it.",
                "CV",
            )
        cv_found = documents.get("cv") is not None
        cover_letter_found = documents.get("cover_letter") is not None
        logger.info(f"Documents read - CV found: {cv_found}, Cover letter found: {cover_letter_found}")
        if cv_found:
            logger.debug(f"CV length: {len(documents['cv'])} characters")
        if cover_letter_found:
            logger.debug(f"Cover letter length: {len(documents['cover_letter'])} characters")
        
        # Write to debug file
        debug_content = ""
        if cv_found:
            debug_content += "CV CONTENT:\n"
            debug_content += "-" * 80 + "\n"
            debug_content += documents['cv']
            debug_content += "\n\n"
        else:
            debug_content += "CV CONTENT: Not found\n\n"
        
        if cover_letter_found:
            debug_content += "COVER LETTER CONTENT:\n"
            debug_content += "-" * 80 + "\n"
            debug_content += documents['cover_letter']
            debug_content += "\n\n"
        else:
            debug_content += "COVER LETTER CONTENT: Not found\n\n"
        
        write_to_debug(debug_content, "DOCUMENT READER DEBUG INFO")
        
        
        state_update = {
            "messages": state.get('messages', []) + [{
                "role": "assistant",
                "content": "Documents read successfully."
            }],
            "candidate_text": documents,
            "current_node": "document_reader"
        }
        
        return state_update

