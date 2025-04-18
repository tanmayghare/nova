import os
import google.generativeai as genai
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def list_chat_models():
    """Lists Gemini models compatible with the generateContent method."""
    try:
        # Load API key from .env file
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            logging.error("Error: GOOGLE_API_KEY not found in environment variables or .env file.")
            print("\nPlease ensure your GOOGLE_API_KEY is set in a .env file or environment variables.")
            return

        # Configure the generative AI client
        genai.configure(api_key=api_key)
        logging.info("Google GenAI client configured.")

        print("\nAvailable Gemini models supporting 'generateContent' (suitable for ChatGoogleGenerativeAI):")
        print("-" * 80)

        compatible_models_found = False
        # List models and filter
        for model in genai.list_models():
            # Check if the model supports the method used by ChatGoogleGenerativeAI
            if 'generateContent' in model.supported_generation_methods:
                print(f"- {model.name}")
                compatible_models_found = True
        
        if not compatible_models_found:
             print("No models specifically listing 'generateContent' found.")
             print("NOTE: Models like 'gemini-pro' might still work even if not explicitly listed here.")
             print("Refer to Google's documentation for the latest compatibility.")

        print("-" * 80)

    except ImportError:
        logging.error("Error: google-generativeai or python-dotenv library not found.")
        print("\nPlease install the required libraries: pip install google-generativeai python-dotenv")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"\nAn error occurred. Please check the logs and ensure your API key is valid. Error: {e}")

if __name__ == "__main__":
    list_chat_models() 