import pickle
import sys
import torch
import numpy as np
from PIL import Image

def inspect_model(model_path):
    try:
        print(f"Attempting to load model from: {model_path}")
        
        # Load the model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print("\nModel Information:")
        print("-----------------")
        print(f"Model Type: {type(model)}")
        print(f"Model Architecture:\n{model}")
        
        if isinstance(model, torch.nn.Module):
            print("\nPyTorch Model Details:")
            print("---------------------")
            print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
            
            for name, param in model.named_parameters():
                if 'weight' in name and len(param.shape) > 1:
                    print(f"First layer shape: {param.shape}")
                    break
        
        elif hasattr(model, 'get_params'):
            print("\nScikit-learn Model Details:")
            print("-------------------------")
            print("Parameters:", model.get_params())
            
            if hasattr(model, 'classes_'):
                print("\nClasses:", model.classes_)
            
            if hasattr(model, 'n_features_in_'):
                print("Input features:", model.n_features_in_)
        
        print("\nTesting model with dummy data:")
        print("-----------------------------")
        # dummy RGB image (224x224 is common for many models)
        dummy_image = np.random.rand(224, 224, 3).astype(np.float32)
        dummy_input = np.expand_dims(dummy_image, axis=0)
        
        try:
            prediction = model.predict(dummy_input)
            print(f"Output shape: {prediction.shape}")
            print(f"Output type: {type(prediction)}")
            print(f"Sample prediction: {prediction}")
        except Exception as e:
            print(f"Could not make test prediction: {str(e)}")
            
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
    except Exception as e:
        print(f"Error inspecting model: {str(e)}")

if __name__ == "__main__":
    model_path = 'model/model3.pkl'
    
    import os
    if not os.path.exists(model_path):
        print(f"Error: {model_path} does not exist!")
        print("Current directory:", os.getcwd())
        print("Files in model directory:")
        if os.path.exists('model'):
            print(os.listdir('model'))
        else:
            print("'model' directory not found!")
        sys.exit(1)
        
    inspect_model(model_path) 