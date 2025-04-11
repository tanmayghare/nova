# Nova Templates

This directory contains templates for creating custom tools and agents in Nova.

## Tool Template

The `tool_template.py` file provides a template for creating custom tools. To create a new tool:

1. Copy `tool_template.py` to a new file
2. Rename the `CustomTool` class to your tool's name
3. Update the tool's name, description, and parameters
4. Implement the `execute` method with your tool's logic

Example usage:
```python
from nova.tools import Tool
from typing import Dict, Any

class MyCustomTool(Tool):
    def __init__(self):
        super().__init__(
            name="my_tool",
            description="Description of my custom tool",
            parameters={
                "param1": {"type": "string", "description": "First parameter"},
                "param2": {"type": "integer", "description": "Second parameter"}
            }
        )
    
    def execute(self, params: Dict[str, Any]) -> str:
        # Implement your tool's logic here
        return f"Processed {params['param1']} and {params['param2']}"
```

## Agent Configuration Template

The `agent_config_template.yaml` file provides a template for configuring custom agents. To create a new agent configuration:

1. Copy `agent_config_template.yaml` to a new file
2. Update the agent's name, description, and settings
3. Configure the tools you want the agent to use
4. Set up the memory configuration
5. Customize the system prompt and examples

Example configuration:
```yaml
name: "MyCustomAgent"
description: "Description of my custom agent"

settings:
  model: "gpt-4"
  temperature: 0.7
  max_tokens: 2000

tools:
  - name: "my_tool"
    enabled: true
    config:
      param1: "default_value"
      param2: 42

memory:
  type: "conversation"
  max_history: 10

system_prompt: |
  You are a helpful assistant specialized in [your specialty].
  Your capabilities include:
  - [capability 1]
  - [capability 2]
  - [capability 3]
```

## Best Practices

1. **Tool Development**:
   - Keep tools focused on a single responsibility
   - Provide clear parameter descriptions
   - Handle errors gracefully
   - Document your tool's usage

2. **Agent Configuration**:
   - Choose appropriate model settings
   - Configure only the tools you need
   - Write clear and specific system prompts
   - Test your configuration thoroughly

3. **Testing**:
   - Test tools in isolation
   - Verify agent behavior with different configurations
   - Check error handling and edge cases

## Contributing

When contributing new templates:
1. Follow the existing template structure
2. Include clear documentation
3. Add examples of usage
4. Update this README if necessary 