import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
# from utils import preprocess_enhance


def main():
    model = models.resnet18(pretrained=False)
    num_classes = 4
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.Lambda(preprocess_enhance),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root='Lung_Dataset', transform=transform)
    batch_size = 32
    num_data = len(dataset)
    split_ratio = 0.1
    split = int(split_ratio * num_data)
    indices = list(range(num_data))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    num_epochs = 10
    best_accuracy = 0.0
    all_labels = []
    all_preds = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

        model.eval()
        val_labels = []  # 存储验证集的真实标签
        val_preds = []  # 存储验证集的模型预测
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(predicted.cpu().numpy())
        recall = recall_score(val_labels, val_preds, average='weighted')
        precision = precision_score(val_labels, val_preds, average='weighted')
        f1 = f1_score(val_labels, val_preds, average='weighted')
        accuracy = 100 * correct / total
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Recall: {recall}, Precision: {precision}, F1 Score: {f1}, Accuracy: {accuracy}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')
        all_labels.extend(val_labels)
        all_preds.extend(val_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    print(f"AUC: {auc}")
    print("Training finished.")


if __name__ == '__main__':
    main()
