# Nova - Intelligent Browser Automation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

🎓 **Educational and Experimental Project** 🧪

Nova is an intelligent browser automation agent built with Python, powered by LangChain and LangGraph. This project is provided as-is for educational and experimental purposes.

## ⚠️ Important Notice

This project is **not under active development** and is provided for educational and experimental use only. Features may not work as expected, and no support or updates will be provided. 

**If you're interested in developing this further, please fork the repository and create your own version.**

## 🧪 What is Nova?

Nova is an AI-powered browser automation agent that demonstrates how to:
- Navigate websites and perform complex multi-step tasks
- Make intelligent decisions using Large Language Models (LLMs)
- Interact with web pages through natural language instructions
- Implement memory systems for learning from interactions
- Handle dynamic web content and adapt to changes

## 🚀 Features

- **AI-Powered Decision Making**: Uses LLMs to understand and execute complex web tasks
- **Multi-Provider LLM Support**: Compatible with OpenAI, Anthropic, NVIDIA, and other providers
- **Intelligent Web Automation**: Goes beyond simple scripting with context-aware actions
- **Memory System**: Demonstrates learning from previous interactions
- **Flexible Architecture**: Modular design for easy customization and extension
- **Comprehensive Logging**: Detailed monitoring and debugging capabilities

## 📋 Quick Start

### Prerequisites
- Python 3.9 or higher
- A compatible LLM provider account (OpenAI, Anthropic, NVIDIA, etc.)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tanmayghare/nova.git
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

## 🍴 Fork and Develop

This project is provided as-is for educational purposes. If you want to:
- Add new features
- Fix bugs
- Improve functionality
- Adapt it for your needs

**Please fork this repository and create your own version.** The codebase provides a solid foundation for building your own AI-powered browser automation tools.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This is experimental educational software provided as-is. The authors are not responsible for any consequences of using this software. Always review and understand the code before running it, especially when interacting with external services or websites.

**No support, updates, or maintenance will be provided for this project.** 