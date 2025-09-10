import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# First we need to load the MNIST dataset and prepare it for training
transform = transforms.Compose([
    transforms.ToTensor(),  # then we need to convert the images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # and then normalize the pixels between -1 and 1
])

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True) #here we download the dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True) # and create a data loader for batching the data

# Next we define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # input layer
        self.fc2 = nn.Linear(128, 64)     # hidden layer
        self.fc3 = nn.Linear(64, 10)      # output layer (10 digits)

    def forward(self, x):
        x = x.view(-1, 28*28)  
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()

# 3. Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training loop
epochs = 3
for epoch in range(epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 5. Save the model
torch.save(model.state_dict(), "mnist_model.pth")
print("Model saved as mnist_model.pth")
