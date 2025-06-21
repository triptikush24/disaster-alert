import torch
import torch.nn as nn
import pickle
import numpy as np

class RiskDetectionModel(nn.Module):
    def __init__(self):
        super(RiskDetectionModel, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=1)

    def predict(self, image_array):
        self.eval()
        with torch.no_grad():
            # Reshape image to get average RGB values
            rgb_means = np.mean(image_array, axis=(1, 2))
            x = torch.FloatTensor(rgb_means)
            output = self.forward(x)
            return output.numpy()

def load_model():
    model = RiskDetectionModel()
    
    # Load the state dict
    with open('model/model3.pkl', 'rb') as f:
        state_dict = pickle.load(f)
    
    model.load_state_dict(state_dict)
    return model

if __name__ == "__main__":
    # Test the model
    model = load_model()
    
    # Create test image (224x224x3)
    test_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
    
    # Make prediction
    prediction = model.predict(test_image)
    print("Test prediction:", prediction)
    print("Shape:", prediction.shape) 