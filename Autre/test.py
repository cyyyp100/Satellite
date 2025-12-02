

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -------------------------
# 1. Config de base
# -------------------------

BATCH_SIZE = 512
EPOCHS = 5
LR = 1e-3

torch.manual_seed(42)

# -------------------------
# 2. Dataset & DataLoaders
# -------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # moyenne / std de MNIST
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

def get_loaders(batch_size):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    return train_loader, test_loader

# -------------------------
# 3. Modèle CNN un peu costaud
# -------------------------

class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# -------------------------
# 4. Fonctions train / test
# -------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc

# -------------------------
# 5. Benchmark sur un device
# -------------------------

def benchmark(device_str):
    if device_str == "mps":
        device = torch.device("mps")
    elif device_str == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"\n===== Benchmark sur {device} =====")

    train_loader, test_loader = get_loaders(BATCH_SIZE)

    model = DeepCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    epoch_times = []
    t0_total = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Synchronisation pour mesurer correctement le temps GPU
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        t1 = time.perf_counter()
        epoch_time = t1 - t0
        epoch_times.append(epoch_time)

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Temps: {epoch_time:.2f}s | "
            f"Train loss: {train_loss:.4f} | Train acc: {train_acc*100:.2f}% | "
            f"Test acc: {test_acc*100:.2f}%"
        )

    t1_total = time.perf_counter()
    total_time = t1_total - t0_total
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    print(f"\n>>> Résumé {device}:")
    print(f"Temps total : {total_time:.2f}s")
    print(f"Temps moyen par epoch : {avg_epoch_time:.2f}s")

    return total_time, avg_epoch_time

# -------------------------
# 6. Main : CPU vs GPU
# -------------------------

def main():
    devices_to_test = ["cpu"]

    if torch.backends.mps.is_available():
        devices_to_test.append("mps")
    elif torch.cuda.is_available():
        devices_to_test.append("cuda")

    print("Devices testés :", devices_to_test)

    results = {}
    for d in devices_to_test:
        total_time, avg_epoch = benchmark(d)
        results[d] = (total_time, avg_epoch)

    print("\n===== COMPARAISON FINALE =====")
    for d, (total_time, avg_epoch) in results.items():
        print(f"{d.upper():<4} -> total: {total_time:.2f}s | epoch moyen: {avg_epoch:.2f}s")


if __name__ == "__main__":
    main()