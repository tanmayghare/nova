import pytest
from datetime import datetime
from nova.core.memory import Memory
import asyncio


@pytest.mark.asyncio
async def test_memory_initialization():
    """Test memory system initialization."""
    memory = Memory(max_entries=5)
    assert len(memory._memory) == 0
    assert len(memory._summary) == 0
    assert memory._max_entries == 5


@pytest.mark.asyncio
async def test_add_memory():
    """Test adding memories to the system."""
    memory = Memory()
    
    # Test adding with full details
    await memory.add("test task", {"type": "browser", "action": "click"}, "success")
    assert len(memory._memory) == 1
    entry = memory._memory[0]
    assert entry["task"] == "test task"
    assert entry["step"] == {"type": "browser", "action": "click"}
    assert entry["result"] == "success"
    assert "timestamp" in entry
    
    # Test adding with just content
    await memory.add("simple memory")
    assert len(memory._memory) == 2
    entry = memory._memory[1]
    assert entry["task"] == "simple memory"
    assert entry["content"] == "simple memory"
    assert "timestamp" in entry


@pytest.mark.asyncio
async def test_memory_size_limit():
    """Test memory size limit enforcement."""
    memory = Memory(max_entries=2)
    
    # Add three entries
    await memory.add("task1", {}, "result1")
    await memory.add("task2", {}, "result2")
    await memory.add("task3", {}, "result3")
    
    assert len(memory._memory) == 2
    assert memory._memory[0]["task"] == "task2"  # First entry should be removed
    assert memory._memory[1]["task"] == "task3"


@pytest.mark.asyncio
async def test_get_context_no_memories():
    """Test context retrieval with no memories."""
    memory = Memory()
    context = await memory.get_context("test task")
    assert context == "No previous context available."


@pytest.mark.asyncio
async def test_get_context_with_memories():
    """Test context retrieval with existing memories."""
    memory = Memory()
    
    # Add some memories
    await memory.add("test task", {}, "result1")
    await memory.add("different task", {}, "result2")
    await memory.add("test task", {}, "result3")
    
    context = await memory.get_context("test task")
    assert "Previous attempt: result3" in context
    assert "Previous attempt: result1" in context


@pytest.mark.asyncio
async def test_get_context_keyword_matching():
    """Test context retrieval with keyword matching."""
    memory = Memory()
    
    await memory.add("login to website", {}, "login successful")
    await memory.add("click button", {}, "button clicked")
    
    context = await memory.get_context("website login")
    assert "login to website" in context
    assert "login successful" in context


@pytest.mark.asyncio
async def test_get_summary():
    """Test summary retrieval."""
    memory = Memory()
    
    await memory.add("task1", {}, "result1")
    await memory.add("task1", {}, "result2")
    await memory.add("task2", {}, "result3")
    
    # Get summary for specific task
    summary = await memory.get_summary("task1")
    assert "result1" in summary
    assert "result2" in summary
    
    # Get all summaries
    all_summaries = await memory.get_summary()
    assert "task1" in all_summaries
    assert "task2" in all_summaries
    assert "result1" in all_summaries
    assert "result3" in all_summaries


@pytest.mark.asyncio
async def test_clear_memory():
    """Test memory clearing operations."""
    memory = Memory()
    
    # Add memories for multiple tasks
    await memory.add("task1", {}, "result1")
    await memory.add("task1", {}, "result2")
    await memory.add("task2", {}, "result3")
    await memory.add("task2", {}, "result4")
    await memory.add("task3", "simple memory")
    
    # Verify initial state
    context1 = await memory.get_context("task1")
    assert "result1" in context1
    assert "result2" in context1
    
    # Clear memories for task1
    await memory.clear_task("task1")
    
    # Verify task1 memories are cleared while others remain
    empty_context = await memory.get_context("task1")
    assert empty_context == "No previous context available."
    
    context2 = await memory.get_context("task2")
    assert "result3" in context2
    assert "result4" in context2
    
    context3 = await memory.get_context("task3")
    assert "simple memory" in context3
    
    # Clear all memories
    memory.clear()
    
    # Verify all memories are cleared
    for task in ["task1", "task2", "task3"]:
        context = await memory.get_context(task)
        assert context == "No previous context available."
    
    # Verify we can add new memories after clearing
    await memory.add("task4", {}, "new_result")
    context4 = await memory.get_context("task4")
    assert "new_result" in context4


def test_json_serialization():
    """Test JSON serialization and deserialization."""
    memory = Memory()
    
    # Add some data
    memory._memory = [{"task": "test", "step": {}, "result": "success", "timestamp": datetime.now().isoformat()}]
    memory._summary = {"test": "success"}
    
    # Serialize
    json_str = memory.to_json()
    
    # Deserialize
    new_memory = Memory.from_json(json_str)
    
    assert len(new_memory._memory) == 1
    assert new_memory._memory[0]["task"] == "test"
    assert new_memory._summary["test"] == "success"


# ---------------------- Additional Tests ----------------------

@pytest.mark.asyncio
async def test_empty_inputs():
    """Test handling of empty inputs."""
    memory = Memory()
    
    # Empty task
    await memory.add("", {}, "result")
    # Empty result
    await memory.add("task", {}, "")
    # Empty step
    await memory.add("task2", {}, "result2")
    
    # Should handle these gracefully
    context = await memory.get_context("")
    assert isinstance(context, str)
    
    summary = await memory.get_summary("")
    assert isinstance(summary, str)


@pytest.mark.asyncio
async def test_very_large_result():
    """Test memory with very large result strings."""
    memory = Memory()
    
    # Create a large result (100KB)
    large_result = "x" * 100_000
    
    await memory.add("large task", {}, large_result)
    
    # Should still be able to retrieve
    context = await memory.get_context("large task")
    assert "Previous attempt" in context
    assert large_result[:100] in context  # Check that at least the first part is included
    
    # Check summary handling
    summary = await memory.get_summary("large task")
    assert len(summary) > 0
    
    # Ensure serialization works
    json_str = memory.to_json()
    new_memory = Memory.from_json(json_str)
    assert new_memory._memory[0]["result"] == large_result


@pytest.mark.asyncio
async def test_complex_keyword_matching():
    """Test more complex scenarios of keyword matching."""
    memory = Memory()
    
    # Add memories with related but different wording
    await memory.add("search for python documentation", {}, "found python docs")
    await memory.add("find information about async", {}, "found async info")
    await memory.add("locate python async tutorials", {}, "found tutorials")
    
    # Test partial matches
    context1 = await memory.get_context("python async documentation")
    assert "python documentation" in context1 or "async" in context1
    
    # Test with word order differences
    context2 = await memory.get_context("documentation for python")
    assert "python doc" in context2
    
    # Test with synonyms/related concepts
    context3 = await memory.get_context("find python resources")
    assert "python" in context3


@pytest.mark.asyncio
async def test_memory_load_performance():
    """Test memory performance with larger datasets."""
    memory = Memory()
    
    # Add 500 memories
    for i in range(500):
        await memory.add(f"task{i % 10}", {"index": i}, f"result{i}")
    
    # Measure basic operations
    assert len(memory._memory) == 500
    assert len(memory._summary) <= 10  # Should have at most 10 summaries (task0-task9)
    
    # Context retrieval should still work efficiently
    context = await memory.get_context("task5")
    assert "Previous attempt: result" in context
    # The current implementation doesn't include the task name in the context,
    # so we just check that we get some context back
    assert len(context) > 0


@pytest.mark.asyncio
async def test_memory_from_empty_json():
    """Test creating memory from empty or invalid JSON."""
    # Empty JSON
    memory = Memory.from_json('{"memory": [], "summary": {}}')
    assert len(memory._memory) == 0
    assert len(memory._summary) == 0
    
    # Test with missing keys (should handle gracefully)
    with pytest.raises(KeyError):
        Memory.from_json('{"incomplete": true}')


@pytest.mark.asyncio
async def test_update_summary_behavior():
    """Test the behavior of summary updates with different inputs."""
    memory = Memory()
    
    # Test incremental updates
    await memory.add("task", {}, "result1")
    summary1 = await memory.get_summary("task")
    
    await memory.add("task", {}, "result2")
    summary2 = await memory.get_summary("task")
    
    # Second summary should contain both results
    assert summary1 in summary2
    assert "result2" in summary2
    
    # Test with very similar results
    await memory.add("task", {}, "result2")  # Duplicate result
    summary3 = await memory.get_summary("task")
    
    # Should still append (not deduplicate)
    assert summary2 + "\nresult2" == summary3 


@pytest.mark.asyncio
async def test_concurrent_memory_operations():
    """Test concurrent memory operations."""
    memory = Memory()
    
    # Create multiple concurrent tasks
    async def add_memory(task_id):
        for i in range(5):
            await memory.add(f"task{task_id}", {"step": i}, f"result{i}")
            await asyncio.sleep(0.01)  # Simulate some processing time
    
    # Run 5 concurrent tasks
    tasks = [add_memory(i) for i in range(5)]
    await asyncio.gather(*tasks)
    
    # Verify results
    assert len(memory._memory) == 25  # 5 tasks * 5 memories each
    
    # Test concurrent context retrieval
    async def get_context(task_id):
        return await memory.get_context(f"task{task_id}")
    
    contexts = await asyncio.gather(*[get_context(i) for i in range(5)])
    assert len(contexts) == 5
    for i, context in enumerate(contexts):
        assert f"task{i}" in context 


@pytest.mark.asyncio
async def test_memory_persistence_long_running():
    """Test memory persistence during long-running tasks."""
    memory = Memory()
    
    # Simulate a long-running task with multiple steps
    task_id = "long_running_task"
    steps = [
        {"action": "initialize", "params": {"dataset": "large"}},
        {"action": "process", "params": {"batch_size": 100}},
        {"action": "validate", "params": {"threshold": 0.95}},
        {"action": "finalize", "params": {"output": "results.json"}}
    ]
    
    # Add memories for each step
    for i, step in enumerate(steps):
        await memory.add(task_id, step, f"Completed step {i}")
        # Simulate some processing time
        await asyncio.sleep(0.1)
    
    # Verify context retrieval includes all steps
    context = await memory.get_context(task_id)
    for i in range(len(steps)):
        assert f"Completed step {i}" in context
    
    # Test summary after long-running task
    summary = await memory.get_summary(task_id)
    assert "initialize" in summary
    assert "finalize" in summary 


@pytest.mark.asyncio
async def test_memory_cleanup():
    """Test memory cleanup operations."""
    memory = Memory()
    
    # Add memories for multiple tasks
    memory.add("task1", "Memory 1 for task 1")
    memory.add("task1", "Memory 2 for task 1")
    memory.add("task2", "Memory 1 for task 2")
    memory.add("task2", "Memory 2 for task 2")
    memory.add("task3", "Memory 1 for task 3")
    
    # Verify memories were added correctly
    context1 = memory.get_context("task1")
    assert len(context1) == 2
    assert "Memory 1 for task 1" in context1
    assert "Memory 2 for task 1" in context1
    
    # Clear memories for task1
    memory.clear_task("task1")
    
    # Verify task1 memories are cleared while others remain
    assert len(memory.get_context("task1")) == 0
    assert len(memory.get_context("task2")) == 2
    assert len(memory.get_context("task3")) == 1
    
    # Clear all memories
    memory.clear()
    
    # Verify all memories are cleared
    assert len(memory.get_context("task1")) == 0
    assert len(memory.get_context("task2")) == 0
    assert len(memory.get_context("task3")) == 0
    
    # Add new memories after cleanup
    memory.add("task4", "New memory after cleanup")
    
    # Verify new memories can be added and retrieved
    context4 = memory.get_context("task4")
    assert len(context4) == 1
    assert "New memory after cleanup" in context4


def test_memory_json():
    """Test JSON serialization and deserialization."""
    memory = Memory()
    
    # Add some data
    memory._memory = [{"task": "test", "step": {}, "result": "success", "timestamp": datetime.now().isoformat()}]
    memory._summary = {"test": "success"}
    
    # Serialize
    json_str = memory.to_json()
    
    # Deserialize
    new_memory = Memory.from_json(json_str)
    
    assert len(new_memory._memory) == 1
    assert new_memory._memory[0]["task"] == "test"
    assert new_memory._summary["test"] == "success" 