import json
import operator
import os
from datetime import date
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from flight_tool import search_flights
from policy_tool import check_flight_policy, check_hotel_policy, load_policy

load_dotenv()

search_tool = DuckDuckGoSearchRun()
tools = [search_tool, search_flights, check_flight_policy, check_hotel_policy]
tools_by_name = {tool.name: tool for tool in tools}

# gpt-4o-mini while building (cheap); set OPENAI_MODEL=gpt-4o in .env for the demo.
model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
model_with_tools = model.bind_tools(tools)


def system_prompt() -> str:
    company = load_policy()["company"]
    return f"""You are the {company} business travel assistant. Today is {date.today().isoformat()}.
You help employees find flights, hotels and things to do, and you only recommend options
that comply with {company}'s travel policy.

How to work:
- Flights: convert cities to IATA airport codes, call search_flights, then ALWAYS call
  check_flight_policy on the results before recommending anything. Recommend only
  compliant flights; lead with the policy's recommended_flight.
- Hotels: use web search to find hotels or neighborhoods, then call check_hotel_policy to
  get the nightly cap and verify any rate you mention.
- Attractions and general info: use web search.
- If something is out of policy, say which rule it breaks and offer a compliant alternative.
- If a tool returns an error, explain it and ask for what you need. Never invent flights,
  prices or policy rules.
- If key details are missing (dates, origin), ask a short clarifying question.
- Keep answers concise: short bullet lists with price, duration and stops for flights."""


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


def llm_call(state: MessagesState) -> dict:
    """Call the LLM with the current messages and available tools."""
    return {
        "messages": [
            model_with_tools.invoke([SystemMessage(content=system_prompt())] + state["messages"])
        ]
    }


def tool_node(state: MessagesState) -> dict:
    """Execute tool calls from the last message."""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        try:
            observation = tool.invoke(tool_call["args"])
        except Exception as e:  # surface tool failures to the LLM (and to traces) instead of crashing
            observation = {"status": "error", "message": f"{type(e).__name__}: {e}"}
        content = observation if isinstance(observation, str) else json.dumps(observation)
        result.append(ToolMessage(content=content, tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    """Determine whether to continue to tool execution or end."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END


def build_agent():
    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node("llm_call", llm_call)
    graph_builder.add_node("tool_node", tool_node)

    graph_builder.add_edge(START, "llm_call")
    graph_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    graph_builder.add_edge("tool_node", "llm_call")

    # MemorySaver keeps each conversation's history by thread_id (in memory, resets on restart).
    return graph_builder.compile(checkpointer=MemorySaver())
