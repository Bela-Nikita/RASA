# main.py
import logging
from typing import TypedDict, Annotated, List, Literal
from operator import add 
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import InMemorySaver 
from langgraph.prebuilt import ToolNode
from langchain_community.tools import GoogleSearchRun
from langchain_community.utilities import GoogleSearchAPIWrapper
from dotenv import load_dotenv

# Load environment variables (Day 5B: Deployment Prep)
load_dotenv() 

# --- Import Custom Tools and Built-in Google Search Tool ---
try:
    from custom_tools import RESEARCH_BANK, save_finding, summarize_topic_knowledge
except ImportError:
    print("Error: custom_tools.py not found. Please ensure it is in the same directory.")
    exit()

# Built-in Tool: Google Search
search = GoogleSearchAPIWrapper()
google_search_tool = GoogleSearchRun(api_wrapper=search, name="google_search")

# Define all available tools
all_tools = [save_finding, summarize_topic_knowledge, google_search_tool]

# Check API Key
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it.")

# Configure the LLM (Agent powered by an LLM)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
llm_with_tools = llm.bind_tools(all_tools)

# --- 1. Define the Graph State (Day 3B: Sessions & Memory) ---

class RASAState(TypedDict):
    """
    Shared state object for the multi-agent system.
    Annotated[List[BaseMessage], add] implements the short-term memory.
    """
    messages: Annotated[List[BaseMessage], add] 
    user_id: int
    research_topic: str
    intent: Literal["RESEARCH", "STUDY", "GENERAL"]

# --- 2. Agent Nodes (Functions) ---

# A. Router Agent (Day 1B/5A: Sequential Routing)
def route_request(state: RASAState) -> RASAState:
    """Routes the request based on intent (RESEARCH, STUDY, or GENERAL)."""
    last_message = state["messages"][-1].content
    
    if "research" in last_message.lower() or "find facts" in last_message.lower():
        intent = "RESEARCH"
        topic = last_message.split(" on ") [-1].strip() # Simple topic extraction
    elif "test me" in last_message.lower() or "explain" in last_message.lower():
        intent = "STUDY"
        topic = last_message.split(" on ") [-1].strip()
    else:
        intent = "GENERAL"
        topic = "General Inquiry"

    logging.info(f"[Router] Detected {intent} for topic '{topic}'.")
    
    # Return the keys that need updating in the state
    return {"intent": intent, "research_topic": topic}


# B. Research Agent (Deep Agent/Loop Core)
# This agent handles complex, multi-step research using tools.
def research_agent(state: RASAState) -> RASAState:
    """
    The Research Agent: Uses the ReAct loop to gather information, check LTM, 
    search the web, and save findings.
    """
    # System prompt forces complex decision making (Day 1A/2A)
    system_prompt = f"""
    You are a Deep Research Assistant. Your goal is to provide comprehensive, sourced answers.
    Your topic is {state['research_topic']}.
    
    1. **Plan:** Use `summarize_topic_knowledge` to check what the user already knows.
    2. **Execute:** Use `Google Search` to find new information.
    3. **Synthesize:** If a new fact is found, use `save_finding` to update the Research Bank.
    4. **Output:** Provide a final, summarized answer with citations.
    """
    
    prompt_messages = [HumanMessage(content=system_prompt)] + state["messages"]
    response = llm_with_tools.invoke(prompt_messages)
    
    # Return the state update, containing the message/tool call
    return {"messages": [response]} 


# C. Study Agent (The Knowledge Application Agent)
def study_agent(state: RASAState) -> RASAState:
    """Handles knowledge application, testing, and explanation."""
    last_message = state["messages"][-1].content
    topic = state["research_topic"]

    # Uses the LTM tool to retrieve knowledge before answering
    knowledge_summary = summarize_topic_knowledge.invoke({"user_id": state['user_id'], "topic": topic})

    prompt = f"""
    You are a dedicated Tutor. The user wants to study '{topic}'. 
    Their existing knowledge is: "{knowledge_summary}".
    Answer the user's question, applying their existing knowledge while introducing new concepts.
    User's request: {last_message}
    """
    response = llm.invoke(prompt) # Don't bind tools, just answer based on context
    
    return {"messages": [response]}

# --- 3. Conditional Edges (Routing Logic / Day 5A: A2A Protocol) ---

def check_agent_status(state: RASAState) -> Literal["call_tool", "respond_final"]:
    """Conditional logic to continue the ReAct loop or finish."""
    last_message = state['messages'][-1] 
    
    # If the LLM decided to call a tool, execute it
    if last_message.tool_calls: 
        return "call_tool"
    
    # If no tool call, the agent provided the final answer/report.
    return "respond_final" 

# --- 4. Build the Graph (MAS Orchestration) ---

tool_node = ToolNode(all_tools)

workflow = StateGraph(RASAState)

# Nodes: The Agents and Tool Executors
workflow.add_node("router", route_request)
workflow.add_node("research_agent", research_agent)
workflow.add_node("study_agent", study_agent)
workflow.add_node("tool_executor", tool_node) 

# Edges: Sequential and Conditional Flows
workflow.add_edge(START, "router")

# Router determines the specialist (Multi-Agent System)
workflow.add_conditional_edges(
    "router",
    lambda state: state['intent'],
    {
        "RESEARCH": "research_agent",
        "STUDY": "study_agent",
        "GENERAL": END # Handle simple general inquiries directly (or use a dedicated general agent)
    }
)

# Research Agent Loop (Deep Agent Logic)
workflow.add_conditional_edges(
    "research_agent",
    check_agent_status,
    {
        "call_tool": "tool_executor",
        "respond_final": END,
    }
)

# Study Agent is usually sequential (answers based on LTM and exits)
workflow.add_edge("study_agent", END)

# Tool Result Edge: After the tool runs, its output goes back to the originating agent (ReAct loop)
workflow.add_edge("tool_executor", "research_agent")

# Compile the Graph and enable Checkpointing (Day 3B: Memory)
checkpointer = InMemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# --- 5. Invocation and Example Execution ---

if __name__ == "__main__":
    
    # Example 1: RESEARCH Task (Triggers ReAct Loop to query LTM, search, and save)
    user_id = 101
    research_query = "Please research the current state of Quantum Computing and save the main finding."
    
    inputs_1 = {
        "messages": [HumanMessage(content=research_query)],
        "user_id": user_id,
        "research_topic": "Quantum Computing",
        "intent": ""
    }
    
    print("--- 1. RESEARCH Task (Loop to Find and Save) ---")
    final_state_1 = app.invoke(inputs_1, {"configurable": {"thread_id": f"research_session_{user_id}"}})
    
    print("\n[RASA Research Response]:", final_state_1['messages'][-1].content)
    
    # Example 2: STUDY Task (Triggers retrieval from LTM and tutoring)
    study_query = "Test me on Quantum Computing. What is a qubit?"
    
    inputs_2 = {
        "messages": [HumanMessage(content=study_query)],
        "user_id": user_id,
        "research_topic": "Quantum Computing",
        "intent": ""
    }
    
    print("\n--- 2. STUDY Task (Retrieval and Tutoring) ---")
    final_state_2 = app.invoke(inputs_2, {"configurable": {"thread_id": f"study_session_{user_id}"}})
            
    print("\n[RASA Study Response]:", final_state_2['messages'][-1].content)
