import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Here we load the model and run a simple test prediction
model = Net()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()  # need to set it to evaluation mode, meaning no dropout, batchnorm, etc.

# Here we load the test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# Need to pick one random image from test dataset
import random
index = random.randint(0, len(test_dataset) - 1)
image, label = test_dataset[index]

# Here we'll run the model prediction
with torch.no_grad():
    output = model(image.unsqueeze(0))  # And penultimately add batch dimension
    predicted = torch.argmax(output, dim=1).item() # Here we'll get the index of the max log-probability

print(f"True label: {label}") # And finally print the true and predicted labels
print(f"Predicted: {predicted}")
