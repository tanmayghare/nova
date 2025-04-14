from __future__ import annotations

import logging
import asyncio
import os
from typing import Any, Dict, Optional, Sequence, List
from datetime import datetime
import uuid
import json

# --- Tell ChromaDB to use pysqlite3 binary ---
# Necessary for environments with older system sqlite3
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("Successfully switched to pysqlite3 binary.")
except ImportError:
    print("pysqlite3 not found, using system sqlite3.")
# -------------------------------------------

# --- Add ChromaDB and SentenceTransformer imports ---
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError:
    chromadb = None
    SentenceTransformer = None
    print("Warning: chromadb or sentence-transformers not installed. RAG retrieval will be disabled.")
# ----------------------------------------------------

from .browser import Browser, BrowserPool
from .config import AgentConfig, BrowserConfig
from .llm import LLM
from .memory import Memory
from .tools import Tool, ToolRegistry
from .monitoring import PerformanceMonitor
from ..tools.browser import NavigateTool, ClickTool, TypeTool, WaitTool, ScreenshotTool
# Attempt to import InteractionLogger - will be None if not available at this level
# This allows the core agent to optionally use the logger if the subclass provides it.
try:
    from nova.learning.interaction_logger import InteractionLogger
except ImportError:
    InteractionLogger = None

# --- Add DB/Model Config ---
DEFAULT_DB_PATH = "db/chroma_db"
DEFAULT_COLLECTION_NAME = "interaction_history"
DEFAULT_EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
DEFAULT_RAG_N_RESULTS = 3 # Number of RAG examples to retrieve
DEFAULT_RAG_SIMILARITY_THRESHOLD = 0.6 # Cosine similarity threshold (1 - distance)
# ---------------------------

logger = logging.getLogger(__name__)


class Agent:
    """An autonomous agent that can interact with the web and use tools.
    
    This agent provides a core implementation for web automation and task execution,
    combining browser automation, LLM-based decision making, and tool usage.
    """

    def __init__(
        self,
        llm: LLM,
        tools: Optional[Sequence[Tool]] = None,
        memory: Optional[Memory] = None,
        config: Optional[AgentConfig] = None,
        browser_config: Optional[BrowserConfig] = None,
        browser: Optional[Browser] = None,
        max_parallel_tasks: int = 3,
        browser_pool_size: int = 5,
        interaction_logger: Optional[InteractionLogger] = None,
        # --- Add RAG config params ---
        db_path: str = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
        enable_rag: bool = True, # Flag to enable/disable RAG
        rag_n_results: int = DEFAULT_RAG_N_RESULTS,
        rag_similarity_threshold: float = DEFAULT_RAG_SIMILARITY_THRESHOLD
        # -----------------------------
    ) -> None:
        """Initialize the agent with LLM, tools, memory, and configuration.
        
        Args:
            llm: Language model for decision making
            tools: Collection of tools available to the agent
            memory: Memory system for context management
            config: Agent configuration
            browser_config: Browser configuration
            browser: Optional pre-configured browser instance
            max_parallel_tasks: Maximum number of tasks to execute in parallel
            browser_pool_size: Size of the browser connection pool
            interaction_logger: Optional interaction logger instance
            db_path: Path to the ChromaDB database directory.
            collection_name: Name of the ChromaDB collection for interaction history.
            embedding_model_name: Name of the Sentence Transformer model to use.
            enable_rag: Whether to enable RAG retrieval for prompt augmentation.
            rag_n_results: Number of RAG examples to retrieve
            rag_similarity_threshold: Cosine similarity threshold (1 - distance)
        """
        self.llm = llm
        self.tools = tools or []
        self.memory = memory or Memory()
        self.config = config or AgentConfig()
        self.browser_pool = BrowserPool(
            size=browser_pool_size,
            config=browser_config or BrowserConfig()
        )
        self._browser = None
        self.tool_registry = ToolRegistry()
        self.monitor = PerformanceMonitor()
        self.max_parallel_tasks = max_parallel_tasks
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.interaction_logger = interaction_logger
        self.enable_rag = enable_rag
        self.chroma_collection = None
        self.embedding_model = None
        self.rag_n_results = rag_n_results # Store RAG config
        self.rag_similarity_threshold = rag_similarity_threshold # Store RAG config

        # --- Initialize RAG components (if enabled and libraries available) ---
        if self.enable_rag and chromadb and SentenceTransformer:
            try:
                # Ensure DB directory exists
                os.makedirs(db_path, exist_ok=True)
                
                # Load Embedding Model
                logger.info(f"[RAG] Loading embedding model: {embedding_model_name}...")
                self.embedding_model = SentenceTransformer(embedding_model_name)
                logger.info("[RAG] Embedding model loaded.")

                # Initialize ChromaDB
                logger.info(f"[RAG] Initializing ChromaDB client at path: {db_path}")
                chroma_client = chromadb.PersistentClient(path=db_path)
                self.chroma_collection = chroma_client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"} # Ensure consistency
                )
                logger.info(f"[RAG] ChromaDB collection '{collection_name}' ready.")
                # Log initial count
                count = self.chroma_collection.count()
                logger.info(f"[RAG] Collection '{collection_name}' currently contains {count} entries.")

            except Exception as e:
                logger.error(f"[RAG] Failed to initialize RAG components: {e}. RAG will be disabled.", exc_info=True)
                self.enable_rag = False # Disable RAG if initialization fails
                self.chroma_collection = None
                self.embedding_model = None
        elif self.enable_rag:
             logger.warning("[RAG] RAG enabled but chromadb or sentence-transformers not found. Disabling RAG.")
             self.enable_rag = False
        # --------------------------------------------------------------------

        # Register tools
        for tool in self.tools:
            self.tool_registry.register(tool)
            
        # Set browser if provided
        if browser:
            self.browser = browser

    @property
    def browser(self) -> Optional[Browser]:
        """Get the current browser instance."""
        return self._browser

    @browser.setter
    def browser(self, browser: Optional[Browser]) -> None:
        """Set the browser instance and register browser tools."""
        self._browser = browser
        if browser:
            self._register_browser_tools(browser)

    def _register_browser_tools(self, browser: Browser) -> None:
        """Register browser action tools."""
        browser_tools = [
            NavigateTool(browser),
            ClickTool(browser),
            TypeTool(browser),
            WaitTool(browser),
            ScreenshotTool(browser)
        ]
        for tool in browser_tools:
            self.tool_registry.register(tool)
            logger.info(f"Registered browser tool: {tool.config.name}")

    async def start(self) -> None:
        """Start the agent and its browser pool."""
        if self.browser:
            await self.browser.start()
        else:
            if self.browser_pool.started:
                logger.warning("Browser pool already started")
                return
            await self.browser_pool.start()
        self.monitor.start()

    async def stop(self) -> None:
        """Stop the agent and associated browser resources (direct or pool)."""
        # Stop direct browser if it exists
        if self.browser:
            logger.debug("Stopping direct browser instance.")
            await self.browser.stop()
            # We might not need to stop the pool if direct browser was used exclusively
            # but stopping it safely if it was started doesn't hurt.
            # Alternatively, only stop the pool if self.browser was None initially.
        
        # Stop the pool if it was started (and potentially not stopped by direct browser cleanup)
        if self.browser_pool.started:
            logger.debug("Stopping browser pool.")
            await self.browser_pool.stop()
        else:
            # Log if pool wasn't started (matches warning seen in logs)
            logger.warning("Agent stop: Browser pool was not started or already stopped.")
            
        # Stop monitor regardless
        self.monitor.stop()
        
        # Cancel any active tasks
        for task_id, task in self.active_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self.active_tasks.clear()

    async def run(self, task: str, task_id: Optional[str] = None) -> str:
        """Run the agent on a given task.
        
        Args:
            task: The task description to execute
            task_id: Optional task identifier for tracking
            
        Returns:
            The result of executing the task
            
        Raises:
            Exception: If task execution fails
        """
        task_id = task_id or str(uuid.uuid4())
        
        if task_id in self.active_tasks:
            raise RuntimeError(f"Task {task_id} is already running")
            
        browser_for_task = None # Initialize variable
        try:
            await self.start() # Starts pool if not started

            # --- Acquire browser from pool and register tools --- 
            if not self.browser: # Only acquire if no direct browser was set
                 logger.debug(f"Task {task_id} acquiring browser from pool...")
                 browser_for_task = await self.browser_pool.acquire()
                 if browser_for_task:
                      self.browser = browser_for_task # This setter registers browser tools
                      logger.info(f"Task {task_id} acquired browser from pool and registered tools.")
                 else:
                      logger.error(f"Task {task_id} failed to acquire browser from pool.")
                      raise RuntimeError("Failed to acquire browser for task execution.")
            else:
                 logger.debug(f"Task {task_id} using pre-set browser instance.")
            # ---------------------------------------------------

            # Convert string task to dictionary with description
            task_dict = {
                "description": task,
                "id": task_id,
                "created_at": datetime.now().isoformat()
            }
            
            # Create and track the task
            task_obj = asyncio.create_task(self._execute_task(task_dict, task_id))
            self.active_tasks[task_id] = task_obj
            
            logger.debug(f"Agent.run awaiting task {task_id} completion...")
            result = await task_obj
            logger.debug(f"Agent.run task {task_id} completed. Preparing to return result.")
            
            # --- Add Log Before Return --- 
            logger.info(f"Agent.run returning result for task {task_id}: {str(result)[:200]}...") 
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed in Agent.run: {e}", exc_info=True)
            # Ensure failed tasks also return a dict, or handle appropriately
            # Depending on desired behavior, re-raise or return specific error structure
            raise # Re-raise for now to ensure visibility
        finally:
            # --- Add Log in Finally --- 
            logger.info(f"Agent.run entering finally block for task {task_id}")
            self.active_tasks.pop(task_id, None)
            try:
                # --- Release browser back to pool if acquired --- 
                if browser_for_task:
                     logger.debug(f"Task {task_id} releasing browser back to pool.")
                     await self.browser_pool.release(browser_for_task)
                     # Important: Clear self.browser if it was set by this task
                     # to avoid interfering with other tasks or agent state.
                     # Only clear if the released browser is the one currently set.
                     if self.browser == browser_for_task:
                          self.browser = None
                # ---------------------------------------------------
                await self.cleanup()
            except Exception as e:
                logger.error(f"Cleanup failed in Agent.run finally block: {e}", exc_info=True)

    async def run_batch(self, tasks: List[str]) -> Dict[str, str]:
        """Run multiple tasks in parallel.
        
        Args:
            tasks: List of task descriptions to execute
            
        Returns:
            Dictionary mapping task IDs to results
        """
        if not tasks:
            return {}
            
        # Limit concurrent tasks
        semaphore = asyncio.Semaphore(self.max_parallel_tasks)
        
        async def run_with_semaphore(task: str) -> tuple[str, str]:
            async with semaphore:
                task_id = str(uuid.uuid4())
                result = await self.run(task, task_id)
                return task_id, result
                
        # Run tasks in parallel
        task_results = await asyncio.gather(
            *[run_with_semaphore(task) for task in tasks],
            return_exceptions=True
        )
        
        # Process results
        results = {}
        for task_result in task_results:
            if isinstance(task_result, Exception):
                logger.error(f"Task failed: {task_result}", exc_info=True)
                continue
            task_id, result = task_result
            results[task_id] = result
            
        return results

    async def _execute_task(self, task: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        """Execute a single task using a ReAct loop."""
        # --- Interaction Logging Setup ---
        logger_instance = getattr(self, 'interaction_logger', None)
        log_session_id = None
        log_step_id = 0
        user_goal = task.get("description", "Unknown Goal")
        final_outcome = "FAILURE" # Default outcome

        if logger_instance and InteractionLogger and isinstance(logger_instance, InteractionLogger):
            try:
                log_session_id, log_step_id = logger_instance.start_session(user_goal)
                # Store on self if attributes exist (set by subclass)
                if hasattr(self, '_current_log_session_id'):
                     self._current_log_session_id = log_session_id
                if hasattr(self, '_current_log_step_id'):
                     self._current_log_step_id = log_step_id
                logger.info(f"[Interaction Logging] Started session {log_session_id} for task {task_id}")
            except Exception as log_start_err:
                 logger.error(f"[Interaction Logging] Failed to start session: {log_start_err}", exc_info=True)
                 logger_instance = None # Disable logging if start fails
        # ---------------------------------

        with self.monitor.track_task(task_id):
            max_iterations = self.config.max_iterations
            action_history = []
            cumulative_results = []
            current_context = await self.memory.get_context(task["description"]) # Initial context
            consecutive_tool_failures = 0
            dom_context_str = "DOM not queried yet (no action taken)."
            extra_context_for_next_llm = None

            try:
                for i in range(max_iterations):
                    logger.info(f"ReAct Iteration {i+1}/{max_iterations} for task: {task_id}")
                    
                    # --- RAG Retrieval ---
                    retrieved_examples = []
                    if self.enable_rag and self.embedding_model and self.chroma_collection:
                        try:
                            # Embed the current state (DOM from previous step) for query
                            # Ensure dom_context_str is not empty or placeholder before encoding
                            if dom_context_str and not dom_context_str.startswith("DOM not queried"):
                                logger.debug(f"[RAG] Generating embedding for current state (len={len(dom_context_str)}): {dom_context_str[:200]}...")
                                current_state_embedding = self.embedding_model.encode(dom_context_str).tolist()

                                logger.debug(f"[RAG] Querying ChromaDB collection '{self.chroma_collection.name}' with {self.rag_n_results} results requested.")
                                query_results = self.chroma_collection.query(
                                    query_embeddings=[current_state_embedding],
                                    n_results=self.rag_n_results,
                                    # Optional: Add where clause if filtering is needed later
                                    # where={"goal": user_goal} # Example filter
                                    include=['metadatas', 'documents', 'distances'] # Request needed data
                                )
                                logger.debug(f"[RAG] Query results raw: {query_results}")

                                # Process results (adjust based on actual ChromaDB output structure)
                                if query_results and query_results.get('ids') and query_results['ids'][0]:
                                    num_retrieved = len(query_results['ids'][0])
                                    logger.debug(f"[RAG] Retrieved {num_retrieved} potential examples.")
                                    for idx in range(num_retrieved):
                                        # Chroma returns distance, convert to similarity (Cosine: 1 - distance)
                                        distance = query_results['distances'][0][idx] if query_results.get('distances') else 1.0
                                        similarity = 1.0 - distance
                                        
                                        # Apply similarity threshold
                                        if similarity >= self.rag_similarity_threshold:
                                            metadata = query_results['metadatas'][0][idx] if query_results.get('metadatas') else {}
                                            # Prefer document if available (might be state_text), else use metadata
                                            doc = query_results['documents'][0][idx] if query_results.get('documents') and query_results['documents'][0] else None
                                            if doc and 'state_text' not in metadata: # Store doc as state_text if missing in metadata
                                                metadata['state_text'] = doc

                                            retrieved_examples.append({
                                                "metadata": metadata,
                                                "similarity": similarity # Store similarity score
                                            })
                                        else:
                                            logger.debug(f"[RAG] Skipping example {idx+1} due to low similarity ({similarity:.2f} < {self.rag_similarity_threshold:.2f})")
                                    logger.info(f"[RAG] Retrieved {len(retrieved_examples)} examples above similarity threshold {self.rag_similarity_threshold}.")
                                else:
                                     logger.info("[RAG] No similar examples found in the database.")
                            else:
                                logger.debug("[RAG] Skipping retrieval as current state context is not available yet.")

                        except Exception as rag_err:
                            logger.error(f"[RAG] Retrieval failed: {rag_err}", exc_info=True)
                    # --- End RAG Retrieval ---

                    # --- Prepare Context for THIS iteration's LLM call ---
                    start_llm_time = datetime.now()
                    # --- Build tool descriptions string (Correctly) ---
                    tool_desc_list = []
                    # Iterate over tool *objects* from the registry's values
                    for tool_object in self.tool_registry._tools.values(): 
                        # Ensure the tool object has a config attribute (as expected by Browser tools)
                        if hasattr(tool_object, 'config'):
                             tool_desc_list.append(
                                 f"- {tool_object.config.name}: {tool_object.config.description} (input: {tool_object.config.input_schema})\n"
                             )
                        else:
                             # Fallback for tools that might not use the config structure (like SearchTool/CalculatorTool)
                             tool_desc_list.append(
                                 f"- {tool_object.name}: {tool_object.description} (input: {{}})" # Assuming simple dict input if no schema
                             )
                    available_tools_str = "\n".join(tool_desc_list)

                    # --- Prepare Context for generate_plan ---
                    history_str = json.dumps(action_history, indent=2) if action_history else "No history yet."
                    
                    context_parts = [] # Start fresh each iteration
                    
                    # --- Integrate RAG Examples ---
                    rag_examples_str = ""
                    if retrieved_examples:
                         rag_examples_str = self._format_rag_examples(retrieved_examples)

                    if rag_examples_str:
                         context_parts.insert(0, f"Relevant Past Examples:\n{rag_examples_str}") # Insert at the beginning
                    # --- End RAG Integration ---

                    # Add other context parts AFTER RAG examples if they exist
                    context_parts.extend([
                        f"Initial Context/Memory: {current_context}",
                        f"Recent Execution History:\n{history_str}",
                        f"Current Page Structure (After Previous Action):\n```\n{dom_context_str}\n```"
                    ])
                    if extra_context_for_next_llm:
                        context_parts.append(f"Extended Context (Full HTML from Previous Step):\n```html\n{extra_context_for_next_llm}\n```")

                    context_string = "\n\n---\n\n".join(context_parts) # Use separator for clarity

                    extra_context_for_next_llm = None # Reset extra context after use
                    # -----------------------------------------

                    # --- Get Action from LLM ---
                    # Generate the plan (thought, confidence, steps) by passing task and context separately
                    logger.debug(f"Calling generate_plan with task: {task['description'][:100]}... and context (len={len(context_string)}): {context_string[:500]}...")
                    thought, confidence, plan_steps = await self.llm.generate_plan(
                        task=task['description'],
                        context=context_string
                    )
                    llm_duration = (datetime.now() - start_llm_time).total_seconds()
                    logger.debug(f"LLM action generation took {llm_duration:.2f}s")
                    # self.monitor.log_metric(task_id, "llm_action_generation_time", llm_duration)

                    # --- Process LLM Result ---
                    # Check if plan_steps were generated
                    if not plan_steps:
                        # Handle cases where LLM fails to generate a plan
                        error_msg = "LLM failed to generate a plan/next step."
                        logger.error(error_msg)
                        final_outcome = "ERROR_LLM_NOPLAN"
                        # Decide how to proceed: retry? Abort? For now, abort.
                        raise RuntimeError(error_msg)

                    # Assuming generate_plan returns a list with one action step
                    if len(plan_steps) > 1:
                         logger.warning(f"LLM returned {len(plan_steps)} steps in plan, executing only the first.")
                    
                    action = plan_steps[0] # Get the first (and assumed only) step
                    # thought = action_result.get("thought", "No thought provided.") # Thought now comes from generate_plan
                    # action = action_result.get("action", {}) # Action now comes from plan_steps
                    action_type = action.get("type") or action.get("tool") # Use 'type' if present, fallback to 'tool'
                    action_type = action_type.lower() if action_type else "unknown"
                    action_params = action.get("parameters", {}) or action.get("input", {}) # Use 'parameters' or 'input'
                    # Ensure action_history stores the structure consistently
                    action_history.append({"thought": thought, "confidence": confidence, "action": action})

                    # --- Interaction Logging: Log Step (Before Execution/Finish Check) ---
                    if logger_instance and log_session_id:
                        try:
                            # Determine state representation (e.g., DOM context string)
                            state_repr = dom_context_str
                            # If DOM context is empty, try page URL/title as fallback
                            if not state_repr or state_repr.startswith("DOM not queried"):
                                if self.browser and self.browser._page:
                                    try:
                                         page_title = await self.browser._page.title()
                                         page_url = self.browser._page.url
                                         state_repr = f"State: Title='{page_title}', URL='{page_url}'"
                                    except Exception:
                                         state_repr = "Could not get page title/URL"
                                else:
                                     state_repr = "Browser not available for state"

                            action_repr = json.dumps(action) # Log the full action dict

                            log_step_id = logger_instance.log_step(
                                session_id=log_session_id,
                                step_id=log_step_id,
                                simplified_state=state_repr[:1000], # Limit state length
                                user_goal=user_goal,
                                action_taken=action_repr
                            )
                            # Update self._current_log_step_id if exists
                            if hasattr(self, '_current_log_step_id'):
                                self._current_log_step_id = log_step_id

                        except Exception as log_step_err:
                            logger.error(f"[Interaction Logging] Failed to log step: {log_step_err}", exc_info=True)
                    # -----------------------------------------------------------

                    # --- Check for Finish Action EARLY ---
                    # If the LLM decides to finish, treat it as success and exit the loop
                    if action_type == "finish":
                        logger.info(f"LLM decided to finish the task. Goal: {user_goal}")
                        final_outcome = "SUCCESS"
                        # Add observation for the finish action to history
                        observation = {"status": "success", "result": action_params.get("result", "Task marked finished.")}
                        action_history[-1]["observation"] = observation
                        cumulative_results.append(observation)
                        break # Exit the ReAct loop
                    # -----------------------------------

                    logger.info(f"Iteration {i+1}: Thought: {thought}")
                    logger.info(f"Iteration {i+1}: Action Type: {action_type}, Params: {action_params}")

                    # Step 2: Execute Action (skip if low confidence)
                    if action_type == "low_confidence_retry":
                        logger.debug(f"Task {task_id} - Skipping action execution due to low confidence in previous step.")
                        continue # Skip to the next iteration (context will be prepared with retry info)
                        
                    tool_name = action.get("tool")
                    tool_input = action.get("input", {})
                    
                    if not tool_name:
                         logger.error(f"Task {task_id} - Invalid step format from LLM (missing 'tool'): {action}")
                         action_history[-1]["observation"] = {"error": "Invalid action format from LLM (missing 'tool')"}
                         consecutive_tool_failures += 1 # Count this as a failure type
                         # Check failure limit immediately after incrementing
                         if consecutive_tool_failures >= self.config.max_failures:
                             logger.error(f"Task {task_id} - Reached max tool failures ({self.config.max_failures}) after invalid action format. Stopping.")
                             return {"status": "failed", "error": f"Reached max tool failures ({self.config.max_failures}) after invalid action format.", "history": action_history}
                         continue 

                    try:
                        logger.info(f"Task {task_id} - Executing Action: {tool_name}({tool_input})")
                        result: ToolResult = await self.tool_registry.execute_tool(tool_name, tool_input)
                        
                        # --- Action SUCCEEDED --- 
                        consecutive_tool_failures = 0 # Reset counter
                        
                        # --- Get DOM AFTER successful action (Reset extra context) ---
                        extra_context_for_next_llm = None # Clear any fallback context on success
                        try:
                            logger.debug("Getting DOM structure after successful action...")
                            dom_structure = await self._get_structured_dom()
                            dom_context_str = json.dumps(dom_structure, indent=2)
                            logger.debug("DOM structure updated for next context.")
                        except Exception as dom_e:
                            logger.error(f"Task {task_id} - Failed to get structured DOM after action: {dom_e}", exc_info=True)
                            dom_context_str = "Error fetching DOM structure after action."

                        # Screenshot Logic (V1 - store path, don't store bytes yet)
                        screenshot_path_for_history = "Not taken or failed"
                        if self.config.use_vision and self.browser:
                            try:
                                # --- Ensure outputs directory exists ---
                                output_dir = "outputs"
                                os.makedirs(output_dir, exist_ok=True)
                                # --- Create filename within outputs dir ---
                                screenshot_filename = os.path.join(output_dir, f"screenshot_{task_id}_iter_{i+1}.png")
                                # Call screenshot, but ignore the returned bytes for now
                                await self.browser.screenshot(path=screenshot_filename)
                                screenshot_path_for_history = screenshot_filename # Store the relative path
                                logger.info(f"Task {task_id} - Screenshot saved to: {screenshot_path_for_history}")
                            except Exception as ss_e:
                                logger.warning(f"Task {task_id} - Failed to take screenshot: {ss_e}")
                        
                        # --- Prepare Observation (Make result data JSON serializable) ---
                        serializable_data = None
                        if isinstance(result.data, (str, int, float, bool, list, dict, type(None))):
                            serializable_data = result.data
                        elif isinstance(result.data, bytes):
                            serializable_data = f"<bytes data len={len(result.data)}>"
                        else:
                            # Convert other types to string representation
                            try:
                                serializable_data = str(result.data)
                            except Exception:
                                serializable_data = f"<unserializable data type: {type(result.data).__name__}>"
                        
                        observation = {
                            "status": "success", 
                            "result": serializable_data, # Use the sanitized data
                            "error": result.error, # Include error field from ToolResult if any
                            "screenshot": screenshot_path_for_history # Path now includes "outputs/"
                        }
                        await self.memory.add(task_id, action, observation) 
                        action_history[-1]["observation"] = observation
                        cumulative_results.append(serializable_data) # Store serializable result
                        logger.info(f"Task {task_id} - Action Result Data: {serializable_data}")
                        
                        # +++ ADD INLINE RAG DB PROCESSING FOR SUCCESSFUL STEP +++
                        if self.enable_rag and self.embedding_model and self.chroma_collection and logger_instance and log_session_id:
                            try:
                                logger.debug(f"[RAG Inline] Processing successful step {log_step_id - 1} for DB.")
                                # State and action representations should match what logger used
                                state_repr_for_db = state_repr # From the Interaction Logging block above
                                action_repr_for_db = action_repr # From the Interaction Logging block above
                                step_entry_id = f"{log_session_id}_{log_step_id - 1}" # Use ID consistent with log_processor

                                if state_repr_for_db and not state_repr_for_db.startswith("DOM not queried"):
                                    # Generate Embedding
                                    embedding = self.embedding_model.encode(state_repr_for_db).tolist()

                                    # Prepare Metadata
                                    metadata = {
                                        "goal": user_goal,
                                        "action": action_repr_for_db, # Stored as JSON string
                                        "outcome": "SUCCESS",
                                        "state_text": state_repr_for_db 
                                    }

                                    # Add to ChromaDB (use upsert to handle potential race conditions or retries)
                                    self.chroma_collection.upsert(
                                        ids=[step_entry_id],
                                        embeddings=[embedding],
                                        metadatas=[metadata]
                                        # documents=[state_repr_for_db] # Optional: Add if storing full text in documents
                                    )
                                    logger.info(f"[RAG Inline] Successfully added/updated step {step_entry_id} in ChromaDB.")
                                else:
                                    logger.debug("[RAG Inline] Skipping DB add due to missing state representation.")

                            except Exception as rag_inline_err:
                                logger.error(f"[RAG Inline] Failed to process successful step for DB: {rag_inline_err}", exc_info=True)
                        # +++ END INLINE RAG DB PROCESSING +++
                        
                    except Exception as e:
                        # --- Action FAILED --- 
                        logger.error(f"Task {task_id} - Step execution failed: {tool_name}({tool_input}) - {e}", exc_info=True)
                        
                        # Prepare error observation
                        observation = {"status": "error", "error": str(e)}
                        await self.memory.add(task_id, action, observation)
                        action_history[-1]["observation"] = observation
                        
                        # Increment failure counter and check limit
                        consecutive_tool_failures += 1
                        logger.warning(f"Task {task_id} - Consecutive tool failures: {consecutive_tool_failures}/{self.config.max_failures}")
                        if consecutive_tool_failures >= self.config.max_failures:
                            logger.error(f"Task {task_id} - Reached max tool failures ({self.config.max_failures}). Stopping task.")
                            return {"status": "failed", "error": f"Reached max tool failures ({self.config.max_failures}) after error: {e}", "history": action_history}
                        
                        # --- Capture DOM state AFTER error --- 
                        logger.info(f"Task {task_id} - Capturing DOM state after tool execution failure.")
                        try:
                            dom_structure_after_error = await self._get_structured_dom()
                            dom_context_str = json.dumps(dom_structure_after_error, indent=2)
                            logger.debug("DOM structure updated after error for next context.")
                        except Exception as dom_err_after_fail:
                            logger.error(f"Task {task_id} - Failed to get structured DOM *after* tool failure: {dom_err_after_fail}", exc_info=True)
                            dom_context_str = "Error fetching DOM structure after action failure."
                        # --- End Capture DOM state --- 
                        
                        # Keep potential extra_context_for_next_llm from previous low-confidence step if applicable
                        logger.warning("Keeping potential extra HTML context after tool failure.")
                        # Loop continues, error info and post-error DOM will be in next context
                        
                    # Update context for the next iteration
                    # Use observation (result of action) as the primary update
                    current_context = await self.memory.get_context(task["description"]) # Re-fetch context after update
                    logger.debug(f"Re-fetched memory context after step.")

                # Loop finished after max iterations - MOVED TO else BLOCK BELOW
                # logger.warning(f"Task {task_id} reached max iterations ({max_iterations})")
                # final_outcome = "MAX_STEPS_REACHED"
                # return {"status": "max_iterations_reached", "result": cumulative_results}
                else: # This else block executes ONLY if the for loop completes without a break
                    logger.warning(f"Task {task_id} reached max iterations ({max_iterations}) without finishing.")
                    final_outcome = "MAX_STEPS_REACHED"
                    # Note: If the loop finishes normally, we don't explicitly return here.
                    # The function will proceed to the finally block and then implicitly return None
                    # or whatever the containing function expects. We might need a result dict here too.
                    # For now, just setting the outcome for the finally block.
                    # Let's return a consistent dictionary structure like other exit points:
                    return {"status": "max_iterations_reached", "result": cumulative_results}

            except Exception as e:
                logger.error(f"Task execution failed for {task_id}: {e}", exc_info=True)
                # Determine specific error outcome if not already set
                if final_outcome not in ["ERROR_LLM", "ERROR_TOOL_FAILURE"]:
                    final_outcome = "ERROR_EXECUTION"
                # Return error status and any partial results
                return {"status": "error", "error": str(e), "result": cumulative_results}
            finally:
                # --- Interaction Logging: Update Outcome ---
                if logger_instance and log_session_id:
                    try:
                        logger.info(f"[Interaction Logging] Updating outcome to '{final_outcome}' for session {log_session_id}")
                        logger_instance.update_outcome(log_session_id, final_outcome)
                    except Exception as log_update_err:
                         logger.error(f"[Interaction Logging] Failed to update outcome: {log_update_err}", exc_info=True)
                # -------------------------------------------
                # Reset session info on self if attributes exist
                if hasattr(self, '_current_log_session_id'):
                    self._current_log_session_id = None
                if hasattr(self, '_current_log_step_id'):
                    self._current_log_step_id = 0
                logger.info(f"Finished executing task {task_id} with final outcome determination: {final_outcome}")

    async def _execute_browser_action(self, browser: Browser, action: Dict[str, Any]) -> Any:
        """Execute a browser action."""
        try:
            action_type = action.get("type")
            if action_type == "navigate":
                return await browser.navigate(action["url"])
            elif action_type == "click":
                return await browser.click(action["selector"])
            elif action_type == "type":
                return await browser.type(action["selector"], action["text"])
            elif action_type == "wait":
                return await browser.wait(action["selector"], action.get("timeout", 10))
            elif action_type == "screenshot":
                return await browser.screenshot(action.get("path"))
            else:
                raise ValueError(f"Unknown browser action type: {action_type}")
        except Exception as e:
            logger.error("Browser action failed", exc_info=True)
            raise

    async def _execute_tool_action(self, action: Dict[str, Any]) -> Any:
        """Execute a tool action.
        
        Args:
            action: Dictionary containing the tool action details
            
        Returns:
            Result of the tool action
            
        Raises:
            ValueError: If the tool or action is not found
            Exception: If the tool action fails
        """
        try:
            tool_name = action.get("tool")
            if not tool_name:
                raise ValueError("Tool name not specified in action")
                
            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                raise ValueError(f"Tool not found: {tool_name}")
                
            # Execute the tool action
            return await tool.execute(action)
            
        except Exception as e:
            logger.error("Tool action failed", exc_info=True)
            raise

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.browser:
            await self.browser.stop()
        else:
            await self.browser_pool.stop()
        self.monitor.stop()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        return self.monitor.get_metrics()

    # --- Add Method to Extract Structured DOM --- 
    async def _get_structured_dom(self, max_elements: int = 100) -> List[Dict[str, Any]]:
        """Extract structured information about interactive elements from the current page."""
        if not self.browser or not hasattr(self.browser, '_page') or not self.browser._page:
            logger.warning("Cannot get structured DOM: Browser or _page not available.")
            return []

        page = self.browser._page
        structured_elements = []
        kept_attributes = ['id', 'name', 'class', 'type', 'placeholder', 'aria-label', 'role', 'href', 'alt', 'title']

        try:
            # Query for potentially interactive elements
            # Expand selectors as needed
            selectors = 'a, button, input, textarea, select, [role="button"], [role="link"], [role="textbox"], [role="menuitem"]'
            element_handles = await page.query_selector_all(selectors)
            logger.info(f"Found {len(element_handles)} potential interactive elements.")

            count = 0
            for handle in element_handles:
                if count >= max_elements:
                    logger.warning(f"Reached max elements ({max_elements}) for structured DOM.")
                    break
                try:
                    is_visible = await handle.is_visible()
                    is_enabled = await handle.is_enabled()
                    
                    # Filter out non-visible or disabled elements (common criteria)
                    if not is_visible or not is_enabled:
                        continue

                    # Get basic info
                    tag_name = (await handle.evaluate('element => element.tagName')).lower()
                    attributes = await handle.evaluate(f'element => {{ const attrs = {{}}; Array.from(element.attributes).forEach(attr => {{ if ({json.dumps(kept_attributes)}.includes(attr.name)) attrs[attr.name] = attr.value; }}); return attrs; }}')
                    text_content = (await handle.text_content() or "").strip()
                    bbox = await handle.bounding_box() # Format: {'x', 'y', 'width', 'height'}
                    
                    # Add placeholder to text content if it's empty and placeholder exists
                    if not text_content and attributes.get('placeholder'):
                        text_content = f"(placeholder: {attributes['placeholder']})"
                        
                    # Basic filtering based on bounding box (ignore tiny elements if needed)
                    if not bbox or bbox['width'] < 2 or bbox['height'] < 2:
                         continue

                    element_data = {
                        "tag": tag_name,
                        "attributes": attributes,
                        "text": text_content,
                        "bbox": bbox
                    }
                    structured_elements.append(element_data)
                    count += 1

                except Exception as e:
                    # Log error for specific element but continue with others
                    logger.warning(f"Error processing element for structured DOM: {e}", exc_info=False) # Avoid excessive logging
                finally:
                    # Ensure handles are disposed if playwright-python requires it (check documentation)
                    # await handle.dispose() # Uncomment if necessary
                    pass 

            logger.info(f"Extracted {len(structured_elements)} elements for structured DOM.")
            return structured_elements

        except Exception as e:
            logger.error(f"Failed to query or process elements for structured DOM: {e}", exc_info=True)
            return []
    # --- End Method --- 

    # +++ ADD RAG Formatting Helper Method +++
    def _format_rag_examples(self, examples: List[Dict]) -> str:
        """Formats retrieved RAG examples into a string for the LLM prompt."""
        if not examples:
            return ""
        formatted_examples = ["Here are some examples of successful actions taken in similar situations:"]
        # Sort by similarity (descending, closer to 1 is better for cosine similarity)
        examples.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        for i, example in enumerate(examples):
            metadata = example.get("metadata", {})
            state_text = metadata.get("state_text", "Unknown state") # Or use example.get("document")
            goal = metadata.get("goal", "Unknown goal")
            action_str = metadata.get("action", "{}") # Action should be stored as JSON string
            similarity = example.get('similarity', 0.0)

            try:
                # Try parsing action string back to dict for cleaner display
                action_dict = json.loads(action_str)
                action_display = json.dumps(action_dict, indent=2)
            except (json.JSONDecodeError, TypeError):
                 action_display = str(action_str) # Keep as string if parsing fails or not string

            formatted_examples.append(
                f"--- Example {i+1} (Similarity: {similarity:.2f}) ---\n"
                f"Past Goal: {goal}\n"
                f"Past State Snapshot (Summary): {state_text[:250]}...\n" # Show snippet of stored state
                f"Successful Action Taken:\n{action_display}"
            )
        return "\n".join(formatted_examples)
    # +++ END RAG Formatting Helper Method +++
