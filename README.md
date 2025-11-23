# RASA: Research and Study Agent (RASA)



### Project Overview
RASA is a sophisticated **Multi-Agent System** built on LangGraph and the Gemini API to solve the problem of **knowledge fragmentation** in academic research and study.

### Problem Statement
Researchers spend excessive time managing fragmented notes and manually searching for information. Standard study tools do not personalize tutoring based on the student's *curated* knowledge base.

### Solution & Innovation
RASA provides a unified workspace with two specialized agents:
1.  **Research Agent:** Executes a ReAct loop to search the web, synthesize findings, and **write** new knowledge into the **Research Knowledge Bank (LTM)** using the `save_finding` tool.
2.  **Study Agent:** Automatically retrieves and **summarizes** the user's LTM before answering a question, ensuring personalized tutoring based on their specific, saved notes. This two-way LTM integration is the core innovation.

---

### Architecture Diagram
The system uses a Hierarchical, Orchestrator-Specialist model.



1.  **Router Agent:** Directs traffic (`RESEARCH` or `STUDY`) (Sequential).
2.  **Research Agent:** Deep Agent Loop (ReAct) for search and LTM writing.
3.  **Study Agent:** Retrieves LTM and generates tutoring response (Sequential).

### Key Implementation Features (Code in `main.py` and `custom_tools.py`)

| Feature Implemented | Code Fulfillment | Criteria |
| :--- | :--- | :--- |
| **1. Multi-Agent System** | `router`, `research_agent`, `study_agent` nodes connected by conditional edges. | **✓** |
| **2. Loop Agents** | `research_agent` logic uses `check_agent_status` to determine if another search or LTM write is required (ReAct loop). | **✓** |
| **3. Custom Tools** | `save_finding` and `summarize_topic_knowledge` (in `custom_tools.py`) interface with LTM. | **✓** |
| **4. Long Term Memory (LTM)** | Implemented via the `ResearchKnowledgeBank` and accessed/updated by the custom tools. | **✓** |
| **5. A2A Protocol** | Achieved by passing the `RASAState` (containing `research_topic`) between the Router, Research Agent, and Study Agent. | **✓** |
| **6. Observability** | Python's `logging` is implemented in `custom_tools.py` to trace LTM reads and writes. | **✓** |

---

## 4. 💻 Code Files

### A. `requirements.txt`

```txt
langgraph
langchain-core
langchain-google-genai
langchain-community # For GoogleSearchRun
python-dotenv
pydantic
