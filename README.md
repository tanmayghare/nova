# Nova - Intelligent Browser Automation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

🚨 **This is a research and experimental project** 🚨

Nova is an experimental intelligent browser automation agent built with Python, powered by LangChain and LangGraph. This project is actively being developed and is intended for research, experimentation, and educational purposes.

## ⚠️ Important Notice

This project is in early development and should be considered experimental. Features may change, break, or be removed without notice. Use at your own risk and not recommended for production environments.

## 🧪 What is Nova?

Nova is an AI-powered browser automation agent that can:
- Navigate websites and perform complex multi-step tasks
- Make intelligent decisions using Large Language Models (LLMs)
- Interact with web pages through natural language instructions
- Learn from interactions and improve over time
- Handle dynamic web content and adapt to changes

## 🚀 Features

- **AI-Powered Decision Making**: Uses LLMs to understand and execute complex web tasks
- **Multi-Provider LLM Support**: Compatible with OpenAI, Anthropic, NVIDIA, and other providers
- **Intelligent Web Automation**: Goes beyond simple scripting with context-aware actions
- **Memory System**: Learns from previous interactions to improve performance
- **Flexible Architecture**: Modular design for easy customization and extension
- **Comprehensive Logging**: Detailed monitoring and debugging capabilities

## 📋 Quick Start

### Prerequisites
- Python 3.9 or higher
- A compatible LLM provider account (OpenAI, Anthropic, NVIDIA, etc.)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/nova.git
   cd nova
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -e .[dev]
   ```

4. **Install Playwright browsers:**
   ```bash
   playwright install
   ```

5. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and preferences
   ```

### Basic Usage

```bash
# Run with default task
python run_nova.py

# Run with custom task
python run_nova.py "Go to duckduckgo.com and search for 'large language models'"
```

## 📚 Documentation

For detailed documentation, examples, and guides, see:

- [Project Overview](docs/README.md) - Comprehensive project documentation
- [User Guides](docs/user-guides/) - Step-by-step tutorials and guides
- [API Documentation](docs/api/) - Complete API reference
- [Architecture](docs/architecture/) - System design and architecture
- [Examples](examples/) - Code examples and use cases

## 🤝 Research & Experimentation

This project welcomes researchers, developers, and enthusiasts interested in:
- AI-powered web automation
- Human-computer interaction
- Language model applications
- Browser automation techniques
- Agent-based systems

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This is experimental software. The authors are not responsible for any consequences of using this software. Always review and understand the code before running it, especially when interacting with external services or websites.

## 🔗 Links

- [Issues](https://github.com/yourusername/nova/issues) - Report bugs or request features
- [Discussions](https://github.com/yourusername/nova/discussions) - Community discussions and questions 