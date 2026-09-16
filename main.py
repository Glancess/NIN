import torch
import torch.nn as nn

from model.nin import NIN
from dataset.loader import get_loaders
from engine.train import train_one_epoch
from engine.evaluate import evaluate
from Configure import config
from utils.results import save_result
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    train_loader, val_loader, test_loader = get_loaders(batch_size=config.BATCH_SIZE)
    set_seed(config.SEED)
    model = NIN(num_classes=10, kernel_size=config.KERNEL_SIZE).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.LR,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )
    best_val_top1 = 0.0
    best_epoch = 0
    for epoch in range(config.EPOCHS):

        train_loss, train_top1, train_top5 = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            topk=config.TOPK,
        )

        val_loss, val_top1, val_top5 = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            topk=config.TOPK,
        )

        print(
            f"Epoch [{epoch + 1}/{config.EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Top1: {train_top1 * 100:.2f}% "
            f"Top5: {train_top5 * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} "
            f"Top1: {val_top1 * 100:.2f}% "
            f"Top5: {val_top5 * 100:.2f}%"
        )
        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            checkpoint_name = (
                f"best_k{config.KERNEL_SIZE}" f"_wd{config.WEIGHT_DECAY}.pth"
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "val_top1": val_top1,
                    "kernel_size": config.KERNEL_SIZE,
                    "weight_decay": config.WEIGHT_DECAY,
                },
                checkpoint_name,
            )
    print("=" * 50)
    print(f"Kernel Size : {config.KERNEL_SIZE}")
    print(f"Weight Decay: {config.WEIGHT_DECAY}")
    print(f"Best Val Top1: {best_val_top1:.4f}")
    print("=" * 50)
    save_result(config.KERNEL_SIZE, config.WEIGHT_DECAY, best_val_top1, best_epoch)
    # test_loss, test_top1, test_top5 = evaluate(
    #     model=model,
    #     data_loader=test_loader,
    #     criterion=criterion,
    #     device=device,
    #     topk=config.TOPK,
    # )

    # print(
    #     f"Test Loss: {test_loss:.4f} "
    #     f"Top1: {test_top1 * 100:.2f}% "
    #     f"Top5: {test_top5 * 100:.2f}%"
    # )


if __name__ == "__main__":
    main()
