import torch
import torch.nn as nn

from model.nin import NIN
from dataset.loader import get_loaders
from engine.train import train_one_epoch
from engine.evaluate import evaluate
from Configure import config


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    train_loader, val_loader, test_loader = get_loaders(batch_size=config.BATCH_SIZE)

    model = NIN(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.LR,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )

    images, targets = next(iter(train_loader))

    images = images.to(device)
    targets = targets.to(device)

    model.train()

    for i in range(200):
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            pred = outputs.argmax(dim=1)
            acc = (pred == targets).float().mean()

            print(i, "loss:", loss.item(), "acc:", acc.item())


if __name__ == "__main__":
    main()
