import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from MedicalData import MedicalDataset
from prepare_dataset import get_data_loader
from torch.utils.data import DataLoader


def main():
    model = models.resnet50(pretrained=False)
    num_classes = 15
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    X_train, X_test, y_train, y_test = get_data_loader()
    print(X_train)
    train_dataset = MedicalDataset(X_train, y_train, transform)
    train_loader = DataLoader(train_dataset, shuffle=True)

    test_dataset = MedicalDataset(X_test, y_test, transform)
    test_loader = DataLoader(test_dataset, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    num_epochs = 10
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for item in train_loader:
            inputs, labels = item["data"], item["label"]
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for item in test_loader:
                inputs, labels = item["data"], item["label"]
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                # print(outputs)
                predicted = outputs > 0.5
                # print(predicted)
                correct += (predicted == labels).sum().item()
                total += labels.numel()

        accuracy = 100 * correct / total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')

    print('Training completed')


if __name__ == '__main__':
    main()
