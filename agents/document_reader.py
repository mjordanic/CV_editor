from pypdf import PdfReader
import os
from typing import Optional, Dict


class DocumentReaderAgent:
    """Agent for reading CV and cover letter documents from the CV folder."""
    
    def __init__(self, cv_folder: str = "CV"):
        """
        Initialize the DocumentReaderAgent.
        
        Args:
            cv_folder: Path to the folder containing CV and cover letter documents
        """
        self.cv_folder = cv_folder
    
    def read_pdf(self, file_path: str) -> str:
        """
        Read text content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            str: Extracted text from all pages of the PDF
        """
        reader = PdfReader(file_path)
        text_parts = []
        for page in reader.pages:
            text_parts.append(page.extract_text())
        return "\n".join(text_parts)
    
    def fetch_document_text(self, document_name: str) -> Optional[str]:
        """
        Fetches document text from the CV folder.
        
        Args:
            document_name: Name of the document (e.g., "CV", "cover_letter")
        
        Returns:
            str: The content of the document file (txt or pdf), or None if not found
        """
        doc_txt_path = os.path.join(self.cv_folder, f"{document_name}.txt")
        doc_pdf_path = os.path.join(self.cv_folder, f"{document_name}.pdf")
        
        # Check if .txt file exists
        if os.path.exists(doc_txt_path):
            with open(doc_txt_path, "r", encoding="utf-8") as f:
                return f.read()
        
        # Check if .pdf file exists
        if os.path.exists(doc_pdf_path):
            return self.read_pdf(doc_pdf_path)
        
        # If neither exists, return None
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
        documents = self.read_all_documents()
        return {"candidate_text": documents}

