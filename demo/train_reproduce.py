import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import time
import csv

# --- IMPORT TRỰC TIẾP ---
from autolrs_callback import AutoLRS
from models.vgg import VGG


# ------------------------

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
    parser.add_argument('--port', type=int, default=12315)
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"AUTO-LRS Training on: {device}")

    train_loader, test_loader = get_data_loaders(args.batch_size)
    net = VGG('VGG16').to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    log_file = open("reproduce_vgg_log.csv", "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["Time", "Step", "Epoch", "Train_Loss", "Val_Loss", "Val_Acc", "LR"])
    start_time_global = time.time()
    global_step = 0

    # [FIX]: Dùng dictionary để lưu epoch, đảm bảo val_fn luôn thấy giá trị mới nhất
    training_state = {'epoch': 0}

    def val_fn():
        # Lấy epoch hiện tại từ dictionary
        current_epoch = training_state['epoch']

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
        net.train()

        acc = 100. * correct / total
        avg_loss = val_loss / total
        cur_lr = optimizer.param_groups[0]['lr']

        # [ĐÃ ĐỒNG BỘ]
        print(f"Epoch {current_epoch} | Acc: {acc:.2f}% | Loss: {avg_loss:.4f} | LR: {cur_lr:.6f}")

        # Ghi log với current_epoch chuẩn xác
        writer.writerow([time.time() - start_time_global, global_step, current_epoch, "", avg_loss, acc, cur_lr])
        log_file.flush()
        return avg_loss

    autolrs = AutoLRS(net, optimizer, val_fn, listening_port=args.port, warmup_steps=50, warmup_lr=0.01)

    print("Start Training...")
    for epoch in range(1, args.epochs + 1):
        # [FIX]: Cập nhật epoch vào dictionary
        training_state['epoch'] = epoch

        net.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Gửi loss cho AutoLRS Server
            autolrs.on_train_batch_end(loss.item())

            global_step += 1
            if global_step % 20 == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                writer.writerow([time.time() - start_time_global, global_step, epoch, loss.item(), "", "", cur_lr])

        # Validation cuối mỗi epoch
        val_fn()

    log_file.close()


if __name__ == '__main__': main()