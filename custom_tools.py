# custom_tools.py

from typing import Dict
from langchain_core.tools import tool
import logging

LOGGER = logging.getLogger("RASA.Tools")
LOGGER.setLevel(logging.INFO) 

# --- 1. Research Knowledge Base (Long-Term Memory / Day 3B) ---
class ResearchKnowledgeBank:
    """Simulates a persistent database for saving research findings."""
    def __init__(self):
        # Key: user_id -> Value: Dict[topic, List[summary_of_findings]]
        self.db: Dict[int, Dict[str, list]] = {
            101: {"Quantum Computing": ["Initial notes on Qubit entanglement."]},
        }
        LOGGER.info("ResearchKnowledgeBank (LTM) initialized.")

# Initialize the bank instance (GLOBAL STATE for LTM)
RESEARCH_BANK = ResearchKnowledgeBank()

# --- 2. Custom Tools for Research (@tool decorator) ---

@tool
def save_finding(user_id: int, topic: str, finding: str) -> str:
    """
    Saves a synthesized finding or note to the user's permanent research memory.
    Required for long-running research tasks (Deep Agents).
    """
    if user_id not in RESEARCH_BANK.db:
        RESEARCH_BANK.db[user_id] = {}
    if topic not in RESEARCH_BANK.db[user_id]:
        RESEARCH_BANK.db[user_id][topic] = []
        
    RESEARCH_BANK.db[user_id][topic].append(finding)
    
    # Observability: Log the state change
    LOGGER.warning(f"Finding saved: User {user_id} added finding to topic '{topic}'.")
    
    return f"Finding successfully saved to Research Bank under topic '{topic}'."

@tool
def summarize_topic_knowledge(user_id: int, topic: str) -> str:
    """
    Retrieves and summarizes all existing findings on a topic from LTM.
    """
    user_data = RESEARCH_BANK.db.get(user_id, {})
    findings = user_data.get(topic, ["No current findings on this topic."])
    
    summary = "\n- ".join(findings)
    LOGGER.info(f"Retrieved knowledge summary for user {user_id} on '{topic}'.")
    return f"Existing knowledge on '{topic}':\n- {summary}"
