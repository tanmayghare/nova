from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json

# Base system message
SYSTEM_MESSAGE = """You are an autonomous agent that can interact with web browsers and use tools to accomplish tasks.
You have access to a set of tools and can use them to navigate, interact with, and extract information from web pages.
You should think carefully about each action and explain your reasoning before taking it."""

# --- Tool Formatting ---

def format_tools(tools: list) -> str:
    """Format a list of tools for inclusion in a prompt."""
    # Basic formatting, can be enhanced (e.g., group by category)
    formatted = []
    for tool in tools:
        schema_str = f"Input Schema: {tool.get('input_schema', '{}')}"
        formatted.append(f"- {tool['name']}: {tool['description']} ({schema_str})")
    return "\n".join(formatted)

def format_action_history(action_history: list) -> str:
    """Format the action history for inclusion in a prompt."""
    if not action_history:
        return "No actions taken yet."
    
    formatted = []
    for i, entry in enumerate(action_history):
        action = entry.get('action', {})
        result = entry.get('result', {})
        status = result.get('status', 'unknown')
        message = result.get('message') or result.get('error') or result.get('dom') # Show DOM if it was the result
        
        # Truncate long results like DOM for history
        if isinstance(message, str) and len(message) > 300:
             message = message[:297] + "..."
        
        formatted.append(
            f"Step {i+1}:\n"
            f"  Action: {action.get('tool', 'N/A')}({action.get('input', {})})\n"
            f"  Outcome: {status} - {message}"
        )
    return "\n".join(formatted)

# --- Core Prompts ---

# Updated to include detailed DOM instructions
ACTION_GENERATION_PROMPT_TEMPLATE = """
You are Nova, an intelligent browser automation agent. Your goal is to achieve the user's task by interacting with a web browser.

**Current Task:** {task}

**Available Tools:**
You have access to the following tools to interact with the browser:
{tools}
- finish: Use this tool ONLY when the task goal is fully achieved. (Input Schema: {{'reason': 'Completion reason (optional)'}})

**Conversation History & Context:**
{context}

**Recent Action History:**
{action_history}

**Current Web Page Structure (HTML DOM):**
```html
{dom_context}
```

**Instructions:** 
1.  **Analyze the Goal:** Understand the **Current Task**.
2.  **Review History:** Examine the **Recent Action History** to see what has been done and the outcome.
3.  **Analyze the DOM:** Carefully study the **Current Web Page Structure (HTML DOM)** provided above. This is CRUCIAL for finding elements.
4.  **Select the BEST Tool:** Based on the task, history, and the current DOM, choose the single most appropriate tool from **Available Tools** for the *next* step.
5.  **Determine Tool Input:** 
    *   If using tools like `click` or `type`, find the most reliable CSS selector or XPath for the target element within the **Current Web Page Structure (HTML DOM)**.
    *   For `navigate`, provide the correct URL.
    *   For `type`, provide the text to be typed.
6.  **Reasoning:** Briefly explain your thought process for choosing the tool and its specific input, referencing the DOM and history.
7.  **Output Format:** Respond ONLY with a valid JSON object containing two keys: "thought" (your reasoning string) and "action" (a dictionary with "tool" and "input" keys).

**Example Action Output:**
```json
{{
  "thought": "The task is to log in. The history shows I navigated to the login page. The DOM contains an input field with id 'username' and a button with text 'Login'. I need to type the username first.",
  "action": {{
    "tool": "type",
    "input": {{
      "selector": "#username",
      "text": "testuser"
    }}
  }}
}}
```

**Example Finish Output:**
```json
{{
  "thought": "The history shows the required information was extracted and the task is complete.",
  "action": {{
    "tool": "finish",
    "input": {{
      "reason": "Successfully extracted user profile information."
    }}
  }}
}}
```

**Error Handling:** If the last step in the history resulted in an error, analyze the error message and the current DOM to determine a corrective action (e.g., different selector, wait longer) or use `finish` if the task is impossible.

Generate the JSON for the next step:
"""

ACTION_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(ACTION_GENERATION_PROMPT_TEMPLATE)
])

# Updated for Recovery
ERROR_RECOVERY_PROMPT_TEMPLATE = """
You are Nova, an intelligent browser automation agent, currently in an error recovery state.

**Original Task:** {task}

**Failed Action:**
Tool: {action_tool}
Input: {action_input}

**Error Encountered:**
{error}

**Available Tools:**
{tools}
- finish: Use this tool if recovery is impossible or the task goal cannot be met.

**Conversation History & Context:**
{context}

**Recent Action History (Leading to Error):**
{action_history}

**Current Web Page Structure (HTML DOM After Error):**
```html
{dom_context}
```

**Instructions for Recovery:**
1.  **Analyze the Error:** Understand why the **Failed Action** resulted in the **Error Encountered**.
2.  **Examine the DOM:** Check the **Current Web Page Structure (HTML DOM After Error)**. Did the element change? Is it missing? Is there an error message on the page?
3.  **Propose a Recovery Action:** Based on the error and the DOM, decide on the best *single* recovery step using one of the **Available Tools**.
    *   Could waiting help (`wait_for_selector`)?
    *   Is there a different selector to try (`click`, `type`)?
    *   Should you navigate somewhere else (`navigate`)?
    *   Is the task impossible now (`finish`)?
4.  **Reasoning:** Explain your recovery strategy based on the error and DOM.
5.  **Output Format:** Respond ONLY with a valid JSON object containing "thought" and "action" keys, similar to the standard action generation.

**Example Recovery Output:**
```json
{{
  "thought": "The previous 'click' on '#submit-button' failed with a timeout. Looking at the current DOM, the button ID is now '#btn-submit-final'. I will try clicking this new selector.",
  "action": {{
    "tool": "click",
    "input": {{
      "selector": "#btn-submit-final"
    }}
  }}
}}
```

Generate the JSON for the recovery step:
"""

ERROR_RECOVERY_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(ERROR_RECOVERY_PROMPT_TEMPLATE)
])

# Placeholder for Plan Interpretation (If needed)
PLAN_INTERPRETATION_PROMPT = ChatPromptTemplate.from_template(
    "Interpret the following task: {task}. Context: {context}. Tools: {tools}. Output JSON plan."
)

# Output parsers
action_parser = JsonOutputParser(pydantic_object=Dict[str, Any])
plan_parser = JsonOutputParser(pydantic_object=Dict[str, List[Dict[str, Any]]])
recovery_parser = JsonOutputParser(pydantic_object=Dict[str, Any])