
import torch
import torch.optim as optim
import torch.nn as nn
from VarGFaceNet import VarGFaceNet  # Import the model from the repo
from lfw_dataloader import get_loaders
from sklearn.metrics import classification_report  # 引入分类报告
import sys
sys.path.append('../')
from util import get_pt_dataloader, save_csv
import torch.nn.functional as F
# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def training(train_loader, val_loader, num_classes=5755, fold=0):
    # Load Data
    # Initialize model
    model = VarGFaceNet(num_classes=num_classes)  # Update num_classes for your specific dataset
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training
    num_epochs = 20
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
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            predicted = torch.argmax(probabilities, 1)
            all_labels.extend(labels.cpu().numpy())  # 收集真实标签
            all_predictions.extend(predicted.cpu().numpy())  # 收集预测标签

        # 计算精确度、召回率和 F1 分数
    report = classification_report(all_labels, all_predictions, output_dict=True)
    save_csv(report, f'vargface_{fold}.csv')
        # correct = 0
        # total = 0
        # with torch.no_grad():
        #     for images, labels in val_loader:
        #         images, labels = images.to(device), labels.to(device)
        #         outputs = model(images)
        #         probabilities = F.softmax(outputs, dim=1)
        #         predicted = torch.argmax(probabilities, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()
        #
        # accuracy = 100 * correct / total
        # print(f"Validation Accuracy: {accuracy:.2f}%")

if __name__ =='__main__':
    # train_loader, val_loader = get_pt_dataloader()
    # training(train_loader, val_loader, num_classes=1800)
    loaders = get_loaders(batch_size=256)
    for i, (train_loader, val_loader) in enumerate(loaders):
        print(i)
        training(train_loader, val_loader, fold=i)