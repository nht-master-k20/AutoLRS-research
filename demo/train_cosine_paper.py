import sys
import os

# --- [FIX PATH] ---
current_dir = os.path.dirname(os.path.abspath(__file__))
autolrs_path = os.path.abspath(os.path.join(current_dir, '../autolrs'))
if autolrs_path not in sys.path:
    sys.path.append(autolrs_path)
# ------------------

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models.vgg import VGG
import argparse
import time
import csv

# ... (Phần còn lại giữ nguyên code cũ) ...
def get_data_loaders(batch_size):
    print("Preparing CIFAR-10 dataset...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    return trainloader, testloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"COSINE Training on: {device}")

    train_loader, test_loader = get_data_loaders(args.batch_size)
    net = VGG('VGG16').to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log_file = open("cosine_vgg_log.csv", "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["Time", "Step", "Epoch", "Train_Loss", "Val_Loss", "Val_Acc", "LR"])
    start_time_global = time.time()
    global_step = 0

    print("Start Cosine Training...")
    for epoch in range(1, args.epochs + 1):
        net.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % 20 == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                writer.writerow([time.time() - start_time_global, global_step, epoch, loss.item(), "", "", cur_lr])

        scheduler.step()

        # Validation
        net.eval()
        total, correct, val_loss = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        avg_loss = val_loss / total
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} | Acc: {acc:.2f}%")
        writer.writerow([time.time() - start_time_global, global_step, epoch, "", avg_loss, acc, cur_lr])
        log_file.flush()

    log_file.close()

if __name__ == '__main__':
    main()