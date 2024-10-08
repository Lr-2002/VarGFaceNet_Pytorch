
import torch
import torch.optim as optim
import torch.nn as nn
from VarGFaceNet import VarGFaceNet  # Import the model from the repo
from lfw_dataloader import get_loaders
import torch.nn.functional as F
# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def training(train_loader, val_loader):
    # Load Data
    # Initialize model
    num_classes = 5755
    model = VarGFaceNet(num_classes=num_classes)  # Update num_classes for your specific dataset
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training
    num_epochs = 10
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            if labels.max() >= num_classes or labels.min() < 0:
                print(labels.max(), labels.min())
                raise ValueError("Labels out of bounds!")

            optimizer.zero_grad()
            outputs = model(images)
            # outputs = torch.softmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # if loss.item() < 1:
            # # test
            #     outputs = model(images)
            #     probabilities = F.softmax(outputs, dim=1)
            #     predicted = torch.argmax(probabilities, 1)
            #     total = labels.size(0)
            #     correct = (predicted == labels).sum().item()
            #     print('accuracy ', correct/total)

            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

        # Validation accuracy calculation
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probabilities = F.softmax(outputs, dim=1)
                predicted = torch.argmax(probabilities, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%")

if __name__ =='__main__':
    loaders = get_loaders()
    for i, (train_loader, val_loader) in enumerate(loaders):
        print(i)
        training(train_loader, val_loader)