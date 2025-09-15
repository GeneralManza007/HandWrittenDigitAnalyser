from flask import Flask, render_template, request, jsonify
from matplotlib import image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io
import base64

app = Flask(__name__)

# Define the same network structure
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x # this function returns the raw scores for each class

# Load trained model
model = CNNNet()
model.load_state_dict(torch.load("mnist_model.pth", map_location=torch.device("cpu")))
model.eval()

# Transform for input images
transform = transforms.Compose([
    transforms.Grayscale(),          # make sure it's grayscale
    transforms.Resize((28, 28)),     # resize to 28x28
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']  # this is the base64 image from frontend
    image_data = base64.b64decode(data.split(',')[1])  # also the strip header
    image = Image.open(io.BytesIO(image_data)).convert("L")  # and convert to grayscale
    image = transform(image).unsqueeze(0)  # and add batch dimension

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)

        # Get top 3 predictions
        top3_prob, top3_idx = torch.topk(probabilities, 3)

        predictions = [
            {"digit": idx.item(), "confidence": round(prob.item() * 100, 2)}
            for prob, idx in zip(top3_prob[0], top3_idx[0])
        ]

    return jsonify({"predictions": predictions})


if __name__ == "__main__":
    app.run(debug=True)
