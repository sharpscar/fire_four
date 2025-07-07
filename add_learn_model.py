import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

# FireNetImproved
class FireNetImproved(nn.Module):
    def __init__(self):
        super(FireNetImproved, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor()
])

# 데이터셋 로딩 (Train / Validation Split)
data_path = '/home/yang/PycharmProjects/PythonProject/Additional_learning'
full_dataset = datasets.ImageFolder(data_path, transform=transform)
val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_data, val_data = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# 클래스 가중치 (불 691, 불아님 1899 기준)
class_counts = [3000, 3034]
total_samples = sum(class_counts)
class_weights = [total_samples / c for c in class_counts]
weights_tensor = torch.tensor(class_weights).to('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FireNetImproved().to(device)
model.load_state_dict(torch.load("fire_model_improved_best_3.pth", map_location=device))
model.train()

criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Best 모델 저장용 변수
best_val_acc = 0.0
best_model_path = "fire_model_improved_finetuned_best.pth"

# 학습 루프
epochs = 20
for epoch in range(epochs):
    # === Train ===
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    train_acc = correct / total

    # === Validation ===
    model.eval()
    val_loss = 0.0
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total

    print(f"[Epoch {epoch+1}] Train Loss: {running_loss:.4f}, Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")

    # Best 모델 저장
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print("f Best 모델 갱신됨! 저장됨: {best_model_path}")

# 최종 저장.
torch.save(model.state_dict(), "fire_model_improved_finetuned_final.pth")
print("✅ 전체 Epoch 종료. 최종 모델 저장 완료!")
