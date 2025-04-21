# Nova - Intelligent Browser Automation

This is the root directory of the Nova project. For detailed documentation, please refer to:

- [Project Overview](docs/README.md)
- [User Guides](docs/user-guides/)
- [API Documentation](docs/api/)
- [Architecture](docs/architecture/)
- [Deployment Guide](docs/deployment/)

## Quick Start

1.  **Install Dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    pip install -e .[dev]
    ```

2.  **Configure Environment:**
    Copy the example environment file and configure it with your settings (e.g., API keys, model preferences).
    ```bash
    cp .env.example .env
    # Edit .env with your values
    ```
    *Key variables to check:* `LLM_PROVIDER`, `MODEL_NAME`, `NVIDIA_API_KEY` (if using NIM), `BROWSER_HEADLESS`, etc.

3.  **Run the Example:**
    Execute the example script (it uses a default task if none is provided):
    ```bash
    python run_nova.py
    ```
    Or provide your own task:
    ```bash
    python run_nova.py "Go to duckduckgo.com and search for 'large language models'"
    ```

For more detailed setup and development information, see the [Development Guide](docs/user-guides/development.md). 