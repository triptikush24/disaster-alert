import sys
import numpy as np
from PIL import Image
import logging
import json
from model_loader import load_model


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def process_image(image_path):
    try:
        logger.info(f"Processing image: {image_path}")
        
        
        model = load_model()
        logger.info("Model loaded successfully")

        logger.info("Loading image")
        image = Image.open(image_path)
        
        # Convert image to RGB if it's not
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.info(f"Original image size: {image.size}")
        image = image.resize((224, 224))
        logger.info("Image resized to 224x224")
        
        image_array = np.array(image)
        logger.info(f"Image array shape: {image_array.shape}")
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        logger.info(f"Preprocessed array shape: {image_array.shape}")

        # Get prediction from model
        prediction = model.predict(image_array)
        risk_score = prediction[0][1] 
        
        # Consider it risky if probability > 0.5
        risk_detected = bool(risk_score > 0.5)
        
        return {
            'success': True,
            'prediction': prediction.tolist(),
            'risk_detected': risk_detected,
            'risk_score': float(risk_score)
        }
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = process_image(sys.argv[1])
        print(json.dumps(result)) 