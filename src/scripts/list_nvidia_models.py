import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def list_nvidia_models():
    """Lists all available NVIDIA models from the NVIDIA AI Endpoints catalog."""
    try:
        # Load API key from .env file
        load_dotenv()
        api_key = os.getenv("NVIDIA_API_KEY", "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")

        if not api_key:
            logging.error("Error: NVIDIA_API_KEY not found in environment variables or .env file.")
            print("\nPlease ensure your NVIDIA_API_KEY is set in a .env file or environment variables.")
            return

        # Initialize ChatNVIDIA to access the model catalog
        chat = ChatNVIDIA(nvidia_api_key=api_key)
        
        print("\nAvailable NVIDIA models:")
        print("-" * 80)

        # Get available models
        models = chat.get_available_models()
        
        if not models:
            print("No models found. Please check your API key and network connection.")
            return

        # Print each model
        for model in models:
            print(f"- {model}")

        print("-" * 80)
        print("\nNote: Some models may require specific API access or permissions.")
        print("For more information, visit: https://docs.nvidia.com/ai-endpoints/")

    except ImportError:
        logging.error("Error: langchain-nvidia-ai-endpoints library not found.")
        print("\nPlease install the required library: pip install langchain-nvidia-ai-endpoints")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"\nAn error occurred. Please check the logs and ensure your API key is valid. Error: {e}")

if __name__ == "__main__":
    list_nvidia_models() 