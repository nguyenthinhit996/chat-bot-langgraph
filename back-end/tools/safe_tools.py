from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from dotenv import load_dotenv  # Import load_dotenv

# Load environment variables from .env file
load_dotenv()

@tool
def lookup_policy(query: str) -> str:
    """Consult the company policies to check whether certain options are permitted.
    Use this before making any flight changes performing other 'write' events."""
    return "baggage is restrict"

class SafeTool:
    tools = [
        TavilySearchResults(max_results=1),
        lookup_policy,
    ]