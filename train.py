import os
import time
import json
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split


def prepare_data(dataset_path='dataset', batch_size=32):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory '{dataset_path}' not found")

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full = datasets.ImageFolder(root=dataset_path, transform=train_transform)
    classes = full.classes

    n = len(full)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size
    train_ds, val_ds, test_ds = random_split(full, [train_size, val_size, test_size])

    # set eval transforms for val/test
    val_ds.dataset.transform = test_transform
    test_ds.dataset.transform = test_transform

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, classes, full


def build_model(num_classes, device):
    model = models.resnet18(pretrained=False)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes))
    model = model.to(device)
    return model


def train(model, device, train_loader, val_loader, epochs=8, lr=1e-4, out_path='resnet18_model.pth'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / val_total if val_total > 0 else 0.0
        val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0.0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{epochs} — {elapsed:.1f}s — Train loss {train_loss:.4f}, Train acc {train_acc:.2f}% — Val loss {val_loss:.4f}, Val acc {val_acc:.2f}%")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_path)
            print(f"Saved best model (val_acc={best_val_acc:.2f}%) to {out_path}")

    return history


def test(model, device, test_loader, classes):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
    # simple accuracy
    correct = sum(int(p == t) for p, t in zip(all_preds, all_labels))
    acc = 100.0 * correct / max(1, len(all_labels))
    print(f"Test Accuracy: {acc:.2f}%")
    return acc, all_preds, all_labels


def plot_history(history, out_img='train_curves.png'):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, history['train_loss'], label='train')
    plt.plot(epochs, history['val_loss'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, history['train_acc'], label='train')
    plt.plot(epochs, history['val_acc'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_img, dpi=150, bbox_inches='tight')
    print(f"Saved training curves to {out_img}")


def main():
    dataset_path = 'dataset'
    batch_size = 32
    epochs = 8
    out_model = 'resnet18_model.pth'

    print('Preparing data...')
    train_loader, val_loader, test_loader, classes, full = prepare_data(dataset_path, batch_size)
    print(f'Found {len(classes)} classes')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = build_model(len(classes), device)

    # if a previous checkpoint exists, load it (fine-tune further)
    if os.path.exists(out_model):
        try:
            model.load_state_dict(torch.load(out_model, map_location=device))
            print('Loaded existing model checkpoint — continuing fine-tuning')
        except Exception as e:
            print('Could not load existing checkpoint:', e)

    history = train(model, device, train_loader, val_loader, epochs=epochs, lr=1e-4, out_path=out_model)

    # save history
    with open('train_history.json', 'w') as f:
        json.dump(history, f)
    print('Saved train_history.json')

    plot_history(history)

    # load best model and test
    best_model = build_model(len(classes), device)
    best_model.load_state_dict(torch.load(out_model, map_location=device))
    test_acc, preds, labels = test(best_model, device, test_loader, classes)


if __name__ == '__main__':
    main()
import os
import time
import argparse
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import numpy as np


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='dataset')
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--save', default='resnet18_model.pth')
    return p.parse_args()


def main():
    args = get_args()
    dataset_path = args.dataset
    if not os.path.exists(dataset_path):
        print('Dataset not found at', dataset_path)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=dataset_path, transform=train_transform)
    classes = full_dataset.classes
    print('Found classes:', len(classes))

    # split
    n = len(full_dataset)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    # set transforms for val/test
    val_dataset.dataset.transform = test_transform
    test_dataset.dataset.transform = test_transform

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # model
    model = models.resnet18(pretrained=False)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, len(classes)))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(args.epochs):
        since = time.time()
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels).item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total

        # validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels).item()
                val_total += labels.size(0)

        val_loss = val_loss / val_total if val_total else 0.0
        val_acc = val_corrects / val_total if val_total else 0.0

        elapsed = time.time() - since
        print(f'Epoch {epoch+1}/{args.epochs} - Train loss: {epoch_loss:.4f} acc: {epoch_acc:.4f} - Val loss: {val_loss:.4f} acc: {val_acc:.4f} - {elapsed:.1f}s')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        torch.save(best_state, args.save)
        print('Saved best model to', args.save, 'with val_acc=', best_val_acc)
    else:
        torch.save(model.state_dict(), args.save)
        print('Saved final model to', args.save)

    # final test evaluation
    model.load_state_dict(torch.load(args.save, map_location=device))
    model.to(device)
    model.eval()
    test_corrects = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels).item()
            test_total += labels.size(0)
    test_acc = test_corrects / test_total if test_total else 0.0
    print('Test accuracy:', test_acc)


if __name__ == '__main__':
    main()
