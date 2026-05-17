import os, time, json, argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from sklearn.model_selection import train_test_split

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='dataset')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--save', default='best_model.pth')
    p.add_argument('--patience', type=int, default=4)
    return p.parse_args()

def build_model(num_classes, device):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model.to(device)

def make_loaders(dataset_path, batch_size):
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Build index splits BEFORE assigning transforms
    full = datasets.ImageFolder(root=dataset_path)
    targets = [s[1] for s in full.samples]
    idx = list(range(len(full)))
    tr_idx, tmp_idx = train_test_split(idx, test_size=0.3, stratify=targets, random_state=42)
    tmp_targets = [targets[i] for i in tmp_idx]
    val_idx, te_idx = train_test_split(tmp_idx, test_size=0.5, stratify=tmp_targets, random_state=42)

    # Separate dataset objects with correct transforms — NO leakage
    train_ds = datasets.ImageFolder(root=dataset_path, transform=train_tf)
    val_ds   = datasets.ImageFolder(root=dataset_path, transform=val_tf)
    test_ds  = datasets.ImageFolder(root=dataset_path, transform=val_tf)

    train_loader = DataLoader(Subset(train_ds, tr_idx),  batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(Subset(val_ds,   val_idx), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(Subset(test_ds,  te_idx),  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, full.classes

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += out.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, 100. * correct / total

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    train_loader, val_loader, test_loader, classes = make_loaders(args.dataset, args.batch_size)
    print(f'Classes ({len(classes)}): {classes}')

    # Save class names immediately
    with open('class_names.json', 'w') as f:
        json.dump(classes, f)
    print('Saved class_names.json')

    model = build_model(len(classes), device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_val_acc, no_improve = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        scheduler.step(vl_loss)
        print(f'Epoch {epoch:02d}/{args.epochs} | {time.time()-t0:.1f}s | '
              f'train {tr_loss:.4f}/{tr_acc:.2f}% | val {vl_loss:.4f}/{vl_acc:.2f}%')

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), args.save)
            print(f'  ✓ Saved best model (val_acc={best_val_acc:.2f}%)')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f'Early stopping at epoch {epoch}')
                break

        # Unfreeze backbone after epoch 5
        if epoch == 5:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1)
            print('  → Unfroze backbone, lr lowered')

    # Test
    model.load_state_dict(torch.load(args.save, map_location=device))
    _, te_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(f'Test accuracy: {te_acc:.2f}%')

if __name__ == '__main__':
    main()