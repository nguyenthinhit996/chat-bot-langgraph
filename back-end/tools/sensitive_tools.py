from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

@tool
def update_ticket_to_new_flight(
    ticket_no: str, new_flight_id: int, *, config: RunnableConfig
) -> str:
    """Update the user's ticket to a new valid flight."""
    return "Ticket successfully updated to new flight." + ticket_no

class SensitiveTools:
    tools = [
        update_ticket_to_new_flight,
    ]