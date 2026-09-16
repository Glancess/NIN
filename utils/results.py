import csv
import os


def save_result(kernel_size, weight_decay, best_val_top1, best_epoch):
    file_path = "results.csv"

    file_exists = os.path.exists(file_path)

    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(
                [
                    "kernel_size",
                    "weight_decay",
                    "best_val_top1",
                    "best_epoch",
                ]
            )

        writer.writerow(
            [
                kernel_size,
                weight_decay,
                best_val_top1,
                best_epoch,
            ]
        )
