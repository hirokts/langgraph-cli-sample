# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LangGraph CLI application that provides an intelligent command-line assistant powered by Azure OpenAI or OpenAI API. The application uses agent-based reasoning with real-time streaming responses.

## Commands

### Setup and Installation
```bash
# Install dependencies using uv (required)
uv sync

# Create .env file with required environment variables
# For Azure OpenAI:
USE_AZURE=true
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_DEPLOYMENT_ID=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-10-21
LLM_MODEL=gpt-4o

# For OpenAI:
USE_AZURE=false
OPENAI_API_KEY=your_openai_api_key
LLM_MODEL=gpt-4o
```

### Running the Application
```bash
# Basic usage - send a message to the AI assistant
python main.py send "Your message here"

# Examples with tools
python main.py send "今何時ですか？"  # Uses get_current_time tool
python main.py send "10 * 5を計算して"  # Uses calculator tool
```

## Architecture

### Core Components

1. **main.py:17-98** - Entry point and streaming handler
   - Manages environment configuration and provider selection
   - Implements real-time streaming using `astream_events`
   - Handles tool execution events and response formatting

2. **agent.py** - Agent graph implementation
   - **AgentState (lines 15-17)**: State management for conversation flow
   - **create_agent_graph() (lines 117-137)**: Main graph builder using StateGraph
   - **call_model() (lines 20-71)**: LLM invocation with tool binding
   - **call_tools() (lines 73-109)**: Tool execution handler
   - Supports both custom StateGraph and prebuilt ReAct agent patterns

3. **tools.py** - Tool definitions
   - `get_current_time_tool`: Returns current timestamp
   - `calculator_tool`: Evaluates mathematical expressions
   - Tool mapping dictionary for dynamic tool resolution

### Key Design Patterns

1. **Provider Abstraction**: The `USE_AZURE` environment variable switches between Azure OpenAI and OpenAI seamlessly (agent.py:12, main.py:22)

2. **State Machine Flow**: 
   - START → call_model → (call_tools → call_model)* → END
   - Conditional edges based on tool_calls presence in AI response

3. **Streaming Architecture**: 
   - Uses LangGraph's `astream_events` for character-level streaming
   - Events: `on_chat_model_stream`, `on_tool_start`, `on_tool_end`

4. **Error Handling**: Tool errors are captured and returned as ToolMessage (agent.py:96-102)

## Important Notes

- The project uses `uv` for dependency management (not pip)
- Python 3.13+ is required (see pyproject.toml:6)
- The typo in agent.py:12 (`UZUE_AZURE` instead of `USE_AZURE`) should be fixed
- Tool execution uses `eval()` in calculator - consider security implications
- Streaming is enabled by default for both providers