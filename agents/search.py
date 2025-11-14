from tavily import TavilyClient
from typing import Optional, Literal
import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field



tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


QUERY_TEMPLATE = (
    "Find the most relevant information about the following company, including company culture, values, and working environment: "
    "\n {company_information}\n"
)

COMPANY_DESCRIPTION_SYSTEM_PROMPT = (
    "You are a helpful assistant that creates concise company descriptions based on search results."
)

COMPANY_DESCRIPTION_HUMAN_PROMPT = (
    "Based on the following search results, create a short (max 2000 characters) description "
    "about the company, focusing on what the company does, its industry, its values, and any notable "
    "characteristics or values. This description will be used to tailor a CV and cover letter for a job "
    "application so make sure to include all the relevant information about core values and company culture and working environment."
    "Here are the search results: \n\n{search_content}"
)

REMOTE_WORK_SYSTEM_PROMPT = (
    "You are a helpful assistant that analyzes company information to determine if they support remote work."
)

REMOTE_WORK_HUMAN_PROMPT = (
    "Based on the following search results, determine if the company supports remote work. "
    "Look for information about remote work policies, work-from-home options, hybrid work arrangements, "
    "or any mentions of remote positions. Return 'supports_remote' as 'yes' if remote work is clearly and fully supported, "
    "'no' if it's clearly not supported (e.g., only on-site or hybrid positions), or 'NA' if the information is insufficient to conclude.\n\n"
    "Search results:\n\n{search_content}"
)


class RemoteWorkResponse(BaseModel):
    """Response model for remote work detection."""
    supports_remote: Literal["yes", "no", "NA"] = Field(
        ...,
        description="'yes' if company supports remote work, 'no' if not, 'NA' if cannot be determined"
    )


class SearchAgent:
    def __init__(self):
        self.llm = init_chat_model("openai:gpt-5-nano", temperature=0)

    def search_tavily(self, query: str):
        """Search using Tavily API with the given query string."""
        results = tavily_client.search(
            query=query, 
            search_depth="advanced", 
            max_results=5, 
            include_images=False,
            include_raw_content="text"
        )
        return results

    def _extract_search_content(self, results: dict) -> str:
        """Extract and combine content from search results."""
        search_contents = []
        if results.get('results'):
            for result in results['results']:
                content = result.get('content', '')
                if content:
                    search_contents.append(content)
        return "\n\n".join(search_contents)

    def process_search_results(self, results: dict):
        """Process the search results from Tavily API and generate a short company description."""
        combined_content = self._extract_search_content(results)
        
        if not combined_content:
            return "No information found about the company."
        
        # Create prompt for generating company description
        prompt = ChatPromptTemplate.from_messages([
            ("system", COMPANY_DESCRIPTION_SYSTEM_PROMPT),
            ("human", COMPANY_DESCRIPTION_HUMAN_PROMPT)
        ])
        
        # Generate description using OpenAI
        chain = prompt | self.llm
        response = chain.invoke({"search_content": combined_content[:8000]})  # Limit content to avoid token limits
        
        return response

    def is_remote(self, results: dict) -> Literal["yes", "no", "NA"]:
        """
        Determine if the company supports remote work based on search results.
        
        Args:
            results: Search results dictionary from Tavily API
            
        Returns:
            'yes' if company supports remote work, 'no' if not, 'NA' if cannot be determined
        """
        combined_content = self._extract_search_content(results)
        
        if not combined_content:
            return "NA"
        
        # Create prompt for remote work detection
        prompt = ChatPromptTemplate.from_messages([
            ("system", REMOTE_WORK_SYSTEM_PROMPT),
            ("human", REMOTE_WORK_HUMAN_PROMPT)
        ])
        
        # Use structured output to ensure proper return type
        structured_llm = self.llm.with_structured_output(RemoteWorkResponse)
        chain = prompt | structured_llm
        response = chain.invoke({"search_content": combined_content[:8000]})
        
        return response.supports_remote

    def run(self, state):
        """
        Run search with company information from state and update state with company_info.
        
        Args:
            state: State dictionary containing job_description_info
        
        Returns:
            dict: Updated state with company_info
        """
        job_description_info = state.get("job_description_info")
        if not job_description_info:
            return {
                "messages": [{
                    "role": "assistant",
                    "content": "Please provide job description information first before searching for company details."
                }]
            }
        
        # Extract company name and location from job description info
        company_info_dict = {
            "name": job_description_info.get("company_name"),
            "location": job_description_info.get("location")
        }
        
        # Convert dict to string for query
        info_parts = []
        for key in ["name", "location"]:
            if company_info_dict.get(key):
                info_parts.append(f"{key.title()}: {company_info_dict[key]}")
        company_info_str = "; ".join(info_parts) if info_parts else "No additional information provided"
        
        # Format the query template
        formatted_query = QUERY_TEMPLATE.format(
            company_information=company_info_str[:150]  # limit the length of the company information because Tavily API has a limit of 400 characters
        )
        print("formatted_query: ", formatted_query)
        
        # Execute search
        results = self.search_tavily(formatted_query)
        
        # Process results
        company_description = self.process_search_results(results)
        remote_work = self.is_remote(results)
        
        # Format company info for state
        company_info = {
            "company_description": company_description.content if hasattr(company_description, 'content') else str(company_description),
            "remote_work": remote_work,
            "search_results": results  # Store raw search results if needed
        }
        
        return {
            "company_info": company_info,
            "messages": [{
                "role": "assistant",
                "content": f"Company information retrieved. Remote work: {remote_work}"
            }]
        }
