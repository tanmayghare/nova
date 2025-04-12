"""Example script demonstrating vision capabilities."""

import os
import logging
from pathlib import Path
from nova.core.vision import VisionModel
from nova.core.llama import LlamaModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_local_image():
    """Test analyzing a local image."""
    try:
        # Initialize vision model
        vision_model = VisionModel()
        
        # Path to test image (replace with your image path)
        image_path = "~/Downloads/test_image.jpg"
        image_path = os.path.expanduser(image_path)
        
        if not os.path.exists(image_path):
            logger.error(f"Image not found at {image_path}")
            return
            
        # Analyze image
        query = "What do you see in this image? Please describe it in detail."
        response = vision_model.analyze_image(image_path, query)
        
        print("\nLocal Image Analysis:")
        print(f"Image: {image_path}")
        print(f"Query: {query}")
        print(f"Response: {response}")
        
    except Exception as e:
        logger.error(f"Error in local image test: {str(e)}")

def test_url_image():
    """Test analyzing an image from URL."""
    try:
        # Initialize vision model
        vision_model = VisionModel()
        
        # Test image URL (replace with your image URL)
        image_url = "https://example.com/test_image.jpg"
        
        # Analyze image
        query = "What do you see in this image? Please describe it in detail."
        response = vision_model.analyze_image_url(image_url, query)
        
        print("\nURL Image Analysis:")
        print(f"Image URL: {image_url}")
        print(f"Query: {query}")
        print(f"Response: {response}")
        
    except Exception as e:
        logger.error(f"Error in URL image test: {str(e)}")

if __name__ == "__main__":
    print("Testing Vision Model Capabilities")
    print("================================")
    
    # Test local image analysis
    test_local_image()
    
    # Test URL image analysis
    test_url_image()

    # Initialize llama model
    model = LlamaModel(model_name="mistral-small3.1:24b-instruct-2503-q4_K_M") 