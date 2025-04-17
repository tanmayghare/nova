"""LangGraph workflow implementation for Nova agent."""

from typing import Dict, Any, List, Optional, Tuple
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

from ..agents.base.base_agent import BaseAgent, AgentState
from ..config.config import AgentConfig
from ..llm.llm import LLM
from ..memory.memory import Memory
from ..tools import Tool, ToolRegistry

class AgentWorkflow:
    """LangGraph workflow for agent execution."""

    def __init__(
        self,
        agent: BaseAgent,
        config: AgentConfig,
        llm: LLM,
        memory: Memory,
        tool_registry: ToolRegistry
    ):
        """Initialize the workflow.
        
        Args:
            agent: Base agent instance
            config: Agent configuration
            llm: Language model for decision making
            memory: Memory system for context management
            tool_registry: Registry of available tools
        """
        self.agent = agent
        self.config = config
        self.llm = llm
        self.memory = memory
        self.tool_registry = tool_registry
        self.tool_executor = ToolExecutor(tool_registry)
        
        # Initialize the workflow graph
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> Graph:
        """Create the LangGraph workflow."""
        # Create the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("prepare_context", self._prepare_context)
        workflow.add_node("generate_action", self._generate_action)
        workflow.add_node("execute_action", self._execute_action)
        workflow.add_node("update_state", self._update_state)
        workflow.add_node("check_termination", self._check_termination)

        # Add edges
        workflow.add_edge("prepare_context", "generate_action")
        workflow.add_edge("generate_action", "execute_action")
        workflow.add_edge("execute_action", "update_state")
        workflow.add_edge("update_state", "check_termination")
        workflow.add_conditional_edges(
            "check_termination",
            self._should_continue,
            {
                True: "prepare_context",
                False: "end"
            }
        )

        # Compile the graph
        return workflow.compile()

    async def _prepare_context(self, state: AgentState) -> AgentState:
        """Prepare context for the next action."""
        # Get relevant past examples using RAG
        examples = await self.memory.get_relevant_examples(
            state.task_description,
            state.current_context
        )

        # Update state with context
        state.current_context = self._format_context(
            task_description=state.task_description,
            examples=examples,
            dom_context=state.dom_context,
            extra_context=state.extra_context
        )

        return state

    async def _generate_action(self, state: AgentState) -> AgentState:
        """Generate the next action based on current state."""
        # Create the prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are an autonomous agent that can execute browser actions."),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content=state.current_context)
        ])

        # Create the chain
        chain = (
            RunnablePassthrough.assign(
                chat_history=lambda x: x.get("chat_history", [])
            )
            | prompt
            | self.llm
            | JsonOutputParser()
        )

        # Generate the action
        action = await chain.ainvoke({
            "task_description": state.task_description,
            "context": state.current_context,
            "tools": self.tool_registry.get_tool_descriptions()
        })

        # Update state with generated action
        state.action_history.append({
            "type": "generated",
            "action": action,
            "timestamp": state.timestamp
        })

        return state

    async def _execute_action(self, state: AgentState) -> AgentState:
        """Execute the current action."""
        if not state.action_history:
            raise ValueError("No action to execute")

        current_action = state.action_history[-1]["action"]
        
        try:
            # Execute the action
            result = await self.tool_executor.ainvoke(current_action)
            
            # Update state with result
            state.action_history[-1]["result"] = result
            state.action_history[-1]["status"] = "success"
            state.consecutive_failures = 0

            # Update DOM context if needed
            if "dom_context" in result:
                state.dom_context = result["dom_context"]

        except Exception as e:
            state.action_history[-1]["result"] = {"error": str(e)}
            state.action_history[-1]["status"] = "failed"
            state.consecutive_failures += 1

        return state

    async def _update_state(self, state: AgentState) -> AgentState:
        """Update the agent state after action execution."""
        # Update memory with successful actions
        if state.action_history and state.action_history[-1]["status"] == "success":
            await self.memory.add_example(
                task_description=state.task_description,
                action=state.action_history[-1]["action"],
                result=state.action_history[-1]["result"]
            )

        return state

    async def _check_termination(self, state: AgentState) -> AgentState:
        """Check if the agent should terminate."""
        if state.consecutive_failures >= self.config.max_consecutive_failures:
            state.final_outcome = "FAILED"
        elif self._is_task_complete(state):
            state.final_outcome = "COMPLETED"

        return state

    def _should_continue(self, state: AgentState) -> bool:
        """Determine if the agent should continue execution."""
        return state.final_outcome == "IN_PROGRESS"

    def _is_task_complete(self, state: AgentState) -> bool:
        """Determine if the task is complete."""
        # This should be implemented based on task-specific criteria
        return False

    def _format_context(
        self,
        task_description: str,
        examples: List[Dict[str, Any]],
        dom_context: str,
        extra_context: Optional[str] = None
    ) -> str:
        """Format context for LLM input."""
        context_parts = [
            f"Task: {task_description}",
            f"Current DOM Context: {dom_context}"
        ]

        if examples:
            context_parts.append("Relevant Past Examples:")
            for example in examples:
                context_parts.append(f"- {example['description']}")

        if extra_context:
            context_parts.append(f"Additional Context: {extra_context}")

        return "\n".join(context_parts) 