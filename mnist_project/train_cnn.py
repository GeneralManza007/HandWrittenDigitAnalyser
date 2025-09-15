import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# 1. First we gotta define the new CNN model
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # meaning 1 input channel (grayscale), 32 output channels, 3x3 kernel  
        self.pool = nn.MaxPool2d(2, 2) # meaning 2x2 pooling                         
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # meaning 32 input channels, 64 output channels, 3x3 kernel
        self.fc1 = nn.Linear(64*7*7, 128) # meaning 64 channels, 7x7 feature maps
        self.fc2 = nn.Linear(128, 10) # meaning 128 input features, 10 output classes (digits 0-9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x # this function returns the raw scores for each class

# 2. The transformations with augmentation
transform = transforms.Compose([
    transforms.RandomRotation(10), 
    transforms.RandomAffine(0, translate=(0.1,0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 3. Next we need to load MNIST dataset
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 4. the training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. training the model
epochs = 5
for epoch in range(epochs):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# 6. Testing accuracy
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
print(f"Test Accuracy: {100. * correct / total:.2f}%")

# 7. Save trained model
torch.save(model.state_dict(), "mnist_model.pth")
print("Model saved as mnist_model.pth")
