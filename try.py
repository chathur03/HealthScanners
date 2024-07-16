import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import cv2

# Define the CNN class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # It has a sequence of layers
        self.cnn_model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),  # First layer
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=5),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),  # Depth of feature maps increases with out_channels
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=5)
        )

        self.fc_model = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=1)
        )

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)  # Flattens the 2D array
        x = self.fc_model(x)
        x = torch.sigmoid(x)  # Use torch.sigmoid instead of F.sigmoid
        return x

model = CNN()
model = model.load_state_dict(torch.load('static/models/brain.pth', map_location='cpu'))


def predict(image_path, model_path="static/models/brain.pth"):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # Resize to (128, 128)

    # Convert BGR to RGB (if using OpenCV)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize pixel values to [0, 1]
    img = img / 255.0

    # Convert to torch tensor and add a batch dimension
    img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)



    # Set device to CPU
    device = torch.device('cpu')

    # Load the model and move it to the CPU
    model = CNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)  # Ensure model is on the CPU
    model.eval()  # Set the model to evaluation mode


    # Perform the prediction
    with torch.no_grad():  # No need to track gradients for inference
        output = model(img_tensor)
        prediction = output.item()  # Get the scalar value from the tensor

    return prediction

# Example usage
image_path = 'static/brain.jpg'
result = predict(image_path)
print(f"Prediction: {result}")