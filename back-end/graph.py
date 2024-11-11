
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from tools.safe_tools import SafeTool
from tools.sensitive_tools import SensitiveTools
from datetime import date, datetime

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition
from state import State
from langgraph.graph import END, StateGraph, START
from utilities import create_tool_node_with_fallback


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            passenger_id = configuration.get("passenger_id", None)
            state = {**state, "user_info": passenger_id}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

def getBuilder():
    llm = ChatOpenAI(model="gpt-4o-mini")

    assistant_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful customer support assistant for Swiss Airlines. "
                " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
                " When searching, be persistent. Expand your query bounds if the first search returns no results. "
                " If a search comes up empty, expand your search before giving up."
                "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
                "\nCurrent time: {time}.",
            ),
            ("placeholder", "{messages}"),
        ]
    ).partial(time=datetime.now)
    # "Read"-only tools (such as retrievers) don't need a user confirmation to use
    safe_tools = SafeTool.tools

    sensitive_tools =  SensitiveTools.tools
    sensitive_tool_names = {t.name for t in sensitive_tools}
    # Our LLM doesn't have to know which nodes it has to route to. In its 'mind', it's just invoking functions.
    assistant_runnable = assistant_prompt | llm.bind_tools(
        safe_tools + sensitive_tools
    )

    builder = StateGraph(State)


    # NEW: The fetch_user_info node runs first, meaning our assistant can see the user's flight information without
    # having to take an action
 
    builder.add_node("assistant", Assistant(assistant_runnable))
    builder.add_node("safe_tools", create_tool_node_with_fallback(safe_tools))
    builder.add_node(
        "sensitive_tools", create_tool_node_with_fallback(sensitive_tools)
    )
    # Define logic
    builder.add_edge(START, "assistant")

    def route_tools(state: State):
        next_node = tools_condition(state)
        # If no tools are invoked, return to the user
        if next_node == END:
            return END
        ai_message = state["messages"][-1]
        # This assumes single tool calls. To handle parallel tool calling, you'd want to
        # use an ANY condition
        first_tool_call = ai_message.tool_calls[0]
        if first_tool_call["name"] in sensitive_tool_names:
            return "sensitive_tools"
        return "safe_tools"


    builder.add_conditional_edges(
        "assistant", route_tools, ["safe_tools", "sensitive_tools", END]
    )
    builder.add_edge("safe_tools", "assistant")
    builder.add_edge("sensitive_tools", "assistant")

    memory = MemorySaver()
    part_3_graph = builder.compile(
        checkpointer=memory,
        # NEW: The graph will always halt before executing the "tools" node.
        # The user can approve or reject (or even alter the request) before
        # the assistant continues
        interrupt_before=["sensitive_tools"],
    )
    return part_3_graph