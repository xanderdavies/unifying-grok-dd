import wandb
import argparse
from tqdm import tqdm
from datetime import datetime
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np

from dataset import ArithmeticDataset, ArithmeticIterator
from adamw import AdamW
from transformer import Transformer
from utils import add_tags

"""
Argument Parsing
"""
parser = argparse.ArgumentParser(
    description="Replication of grokking behavior observed in Power et al.'s 'Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"
)

# model hyperparameters
parser.add_argument(
    "--num-layers", default=2, type=int, help="Number of layers in the transformer"
)
parser.add_argument(
    "--num-heads", default=1, type=int, help="Number of attention heads per layer"
)
parser.add_argument("--d-model", default=128, type=int, help="Dimension of the model")

# training hyperparameters
parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--weight-decay", default=1e-5, type=float, help="Weight decay")
parser.add_argument("--beta1", default=0.9, type=float, help="AdamW beta1")
parser.add_argument("--beta2", default=0.98, type=float, help="AdamW beta2")
parser.add_argument("--use-sgd", default=False, action="store_true")
parser.add_argument("--full-batch", default=False, action="store_true")
parser.add_argument("--momentum", type=float, default=0)
parser.add_argument("--log-normalized-loss", default=True, action="store_true")

# data hyperparameters
parser.add_argument(
    "--vocab-len", default=2000, type=int, help="Transformer vocab length"
)
parser.add_argument("--train-split", default=50, type=int, help="Train split")
parser.add_argument(
    "--embedding-noise",
    default=0,
    type=float,
    help="Add noise to the embedding (value e.g., 0.1)",
)
parser.add_argument("--random-data", default=False, action="store_true")

# run hyperparameters
parser.add_argument(
    "--optimization-budget",
    default=3e5,
    type=int,
    help="Number of training steps to run",
)  # 3e10
parser.add_argument(
    "--wandb-project", default="grokking", type=str, help="Wandb project name"
)
parser.add_argument(
    "--no-logging", action="store_true", help="Disable logging to wandb"
)
parser.add_argument(
    "--device", default=None, type=str, help="Device used for training."
)
parser.add_argument(
    "--resume-run-id", default=None, type=str, help="WandB run to resume."
)
parser.add_argument("--load-path", default=None, type=str, help="Load this model.")
parser.add_argument(
    "--num-jobs",
    default=1,
    type=int,
    help="Number of jobs to run on this gpu (default 1).",
)

arguments = parser.parse_args()
OPTIMIZATION_BUDGET = arguments.optimization_budget
LOG = not arguments.no_logging
DEVICE = arguments.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


def train_step(args, model, optimizer, reg_criterion, criterion, batch):
    X, y = batch["text"], batch["target"]
    X, y = X.to(DEVICE), y.to(DEVICE)
    y_hat = model(X, embedding_noise=args.embedding_noise)
    loss = criterion(y_hat, y)
    train_acc = (y_hat[:, -2, :].argmax(dim=1) == y[:, -2]).float().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        loss_with_reg = reg_criterion(y_hat, y)
        if args.log_normalized_loss:
            loss_normalized = criterion(y_hat, y, normalize=True).item()
        else:
            loss_normalized = None
    return train_acc.item(), loss.item(), loss_with_reg.item(), loss_normalized


def get_train_logdict(
    train_acc, train_loss, loss_with_reg, loss_normalized, epoch, train_step_counter
):
    log_dict = {
        "Loss/train": train_loss,
        "Loss/train_with_reg": loss_with_reg,
        "Loss/train_normalized": loss_normalized,
        "Accuracy/train": train_acc,
        "epoch": epoch,
        "train_step": train_step_counter,
    }
    if not LOG:
        print(f"Epoch {epoch}: train loss {train_loss}, train accuracy {train_acc}")
    return log_dict


def get_interpolation_logdict(
    epoch_train_acc, epoch, train_step_counter, interpolated_99, interpolated_100
):
    log_dict = dict()
    if epoch_train_acc > 0.99 and not interpolated_99:
        print(f"> .99 interpolation achieved at train_step {train_step_counter}")
        interpolated_99 = True
        log_dict.update(
            {
                "Time to .99 Train Accuracy (epoch)": epoch,
                "Time to .99 Train Accuracy (train step)": train_step_counter,
            }
        )
    if epoch_train_acc == 1 and not interpolated_100:
        print(f"> .99 interpolation achieved at train_step {train_step_counter}")
        interpolated_100 = True
        log_dict.update(
            {
                "Time to 1 Train Accuracy (epoch)": epoch,
                "Time to 1 Train Accuracy (train step)": train_step_counter,
            }
        )

    return log_dict


def validate(
    args,
    model,
    train_dataloader,
    val_dataloader,
    zero_enc,
    one_enc,
    criterion,
    epoch,
    train_step_counter,
    generalized_99,
    generalized_100,
    generalized_90,
):
    """"validates model and returns log_dict"""
    model.eval()
    with torch.no_grad():
        loss, accuracy = 0, 0
        div_1_loss, div_1_acc = 0, 0
        zero_start_loss, zero_start_acc = 0, 0
        not_zero_loss, not_zero_acc = 0, 0
        either_loss, either_acc = 0, 0
        a_eq_b_loss, a_eq_b_acc = 0, 0
        for batch in val_dataloader:  # only one batch
            X, y = batch["text"], batch["target"]  # batch['text'], batch['target']
            X, y = X.to(DEVICE), y.to(DEVICE)

            y_hat = model(X)
            loss += criterion(y_hat, y).item()
            accuracy += (
                (y_hat[:, -2, :].argmax(dim=1) == y[:, -2]).float().mean().item()
            )

            zero_start_y = y[y[:, 0] == zero_enc]
            zero_start_X = X[y[:, 0] == zero_enc]
            zero_start_y_hat = model(zero_start_X)

            div_1_y = y[y[:, 2] == one_enc]
            div_1_X = X[y[:, 2] == one_enc]
            div_1_y_hat = model(div_1_X)

            not_zero_y = y[~(y[:, 0] == zero_enc)]
            not_zero_X = X[~(y[:, 0] == zero_enc)]
            not_zero_y_hat = model(not_zero_X)

            a_eq_b_y = y[y[:, 0] == y[:, 2]]
            a_eq_b_X = X[y[:, 0] == y[:, 2]]
            a_eq_b_y_hat = model(a_eq_b_X)

            either_mask = ((y[:, 2] == one_enc) * 1 + (y[:, 0] == zero_enc) * 1) > 0
            either_y = y[either_mask]
            either_X = X[either_mask]
            either_y_hat = model(either_X)

            div_1_loss += criterion(div_1_y_hat, div_1_y).item()
            div_1_acc += (
                (div_1_y_hat[:, -2, :].argmax(dim=1) == div_1_y[:, -2])
                .float()
                .mean()
                .item()
            )

            zero_start_loss += criterion(zero_start_y_hat, zero_start_y).item()
            zero_start_acc += (
                (zero_start_y_hat[:, -2, :].argmax(dim=1) == zero_start_y[:, -2])
                .float()
                .mean()
                .item()
            )

            not_zero_loss += criterion(not_zero_y_hat, not_zero_y).item()
            not_zero_acc += (
                (not_zero_y_hat[:, -2, :].argmax(dim=1) == not_zero_y[:, -2])
                .float()
                .mean()
                .item()
            )

            either_loss += criterion(either_y_hat, either_y).item()
            either_acc += (
                (either_y_hat[:, -2, :].argmax(dim=1) == either_y[:, -2])
                .float()
                .mean()
                .item()
            )

            a_eq_b_loss += criterion(a_eq_b_y_hat, a_eq_b_y).item()
            a_eq_b_acc += (
                (a_eq_b_y_hat[:, -2, :].argmax(dim=1) == a_eq_b_y[:, -2])
                .float()
                .mean()
                .item()
            )

    log_dict = {
        "Loss/val": loss / len(val_dataloader),
        "Loss/val_div_1": div_1_loss / len(val_dataloader),
        "Loss/val_0_div": zero_start_loss / len(val_dataloader),
        "Loss/val_not_0": not_zero_loss / len(val_dataloader),
        "Loss/val_either_0_1": either_loss / len(val_dataloader),
        "Loss/val_a_eq_b": a_eq_b_loss / len(val_dataloader),
        "Accuracy/val": accuracy / len(val_dataloader),
        "Accuracy/val_div_1": div_1_acc / len(val_dataloader),
        "Accuracy/val_0_div": zero_start_acc / len(val_dataloader),
        "Accuracy/val_a_eq_b": a_eq_b_acc / len(val_dataloader),
        "Accuracy/val_not_zero": not_zero_acc / len(val_dataloader),
        "Accuracy/val_either_0_1": either_acc / len(val_dataloader),
        "Percent Zero Prediction": (y_hat[:, -2, :].argmax(dim=1) == zero_enc).sum()
        / len(y_hat[:, -2, :]),
        "epoch": epoch,
        "train_step": epoch * len(train_dataloader),
    }
    if args.log_normalized_loss:
        log_dict["Loss/val_normalized"] = criterion(y_hat, y, normalize=True).item()

    val_acc = log_dict["Accuracy/val"]
    if val_acc > 0.9 and not generalized_90:
        generalized_90 = True
        log_dict.update(
            {
                "Time to .9 Test Accuracy (epoch)": epoch,
                "Time to .9 Test Accuracy (train step)": train_step_counter,
            }
        )
    if val_acc > 0.99 and not generalized_99:
        generalized_99 = True
        log_dict.update(
            {
                "Time to .99 Test Accuracy (epoch)": epoch,
                "Time to .99 Test Accuracy (train step)": train_step_counter,
            }
        )
    if val_acc == 1 and not generalized_100:
        generalized_100 = True
        log_dict.update(
            {
                "Time to 1 Test Accuracy (epoch)": epoch,
                "Time to 1 Test Accuracy (train step)": train_step_counter,
            }
        )
    return log_dict


def run(args: argparse.Namespace):
    """
    Model
    """
    # decoder-only transfrormer with causal attention masking; 2 layers, width 128, 4 attention heads
    model = (
        Transformer(
            n_layers=args.num_layers,
            n_heads=args.num_heads,
            d_model=args.d_model,
            non_linearity="relu",
            vocab_len=args.vocab_len,
        )
        .float()
        .to(DEVICE)
    )
    if args.load_path is not None:
        model.load_state_dict(torch.load(args.load_path))

    """
    Dataset
    """
    train_dataset, val_dataset = ArithmeticDataset.splits(
        train_pct=args.train_split, operator="/", operand_length=None,
    )

    if args.random_data:
        # for a / b = c (mod n), shuffles 'c's for a given 'a'. This preserves the anti-correlation between train-test data.
        for i in range(
            min(train_dataset.data[:, 1]), max(train_dataset.data[:, 1]) + 1
        ):
            mask = train_dataset.data[:, 1] == i
            train_dataset.data[mask, -2] = torch.tensor(
                np.random.permutation(train_dataset.data[mask][:, -2])
            ).type_as(train_dataset.data)

    train_dataloader = ArithmeticIterator(
        train_dataset,
        DEVICE,
        batchsize_hint=(
            0 if not args.full_batch else -1
        ),  # 0 -> default (512), -1 -> full batch
    )
    val_dataloader = ArithmeticIterator(val_dataset, DEVICE, batchsize_hint=-1,)

    sample = train_dataset.tokenizer.encode("0 / 1 = 0 <|eos|>")
    zero_enc, one_enc = sample[0].item(), sample[2].item()

    """
    WandB Logging
    """
    tags = [
        f"d_model={args.d_model}",
        f"num_layers={args.num_layers}",
        f"num_heads={args.num_heads}",
    ]
    date_time = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    name = f"split_{args.train_split}-decay_{args.weight_decay}-dim_{args.d_model}-{date_time}"
    aug_name, aug_tags = add_tags(
        ("full_batch-", args.full_batch),
        (f"noise_{args.embedding_noise}-", args.embedding_noise > 0),
        (f"sgd_lr_{args.lr}_mom_{args.momentum}-", args.use_sgd),
        ("random_data-", args.random_data),
    )
    name = aug_name + name
    tags += aug_tags

    mode = None if LOG else "disabled"
    if args.resume_run_id is None:
        wandb.init(
            entity="initially-overconf-learners",
            project=args.wandb_project,
            id=date_time,
            settings=wandb.Settings(start_method="thread"),
            tags=tags,
            name=name,
            config=args,
            mode=mode,
        )
    else:
        wandb.init(
            id=args.resume_run_id,
            resume="must",
            project=args.wandb_project,
            settings=wandb.Settings(start_method="thread"),
            tags=tags,
            name=name,
            config=args,
            mode=mode,
        )
    wandb.watch(model)

    # log number of parameters
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    wandb.log({"Number of Parameters": num_params}, commit=False)
    print(f"Model has {num_params} trainable parameters.")

    # make weights directory if needed
    try:
        os.makedirs("weights")
    except:
        pass

    """
    Optimizer
    """
    if args.use_sgd:
        optimizer = SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
        )

    """
    Criterion
    """

    class CELReg(nn.CrossEntropyLoss):
        def __init__(self, l2_reg, model):
            super().__init__()
            self.l2_reg = l2_reg
            self.model: nn.Module = model
            self.reg_term = 0

        def forward(self, input, target, **kwargs):
            loss = super().forward(input, target, **kwargs)
            reg_term = 0
            for n, p in model.named_parameters():
                if n.split(".")[-1] == "weight":
                    reg_term += torch.sum(torch.pow(p, 2))
            reg_term /= 2
            self.reg_term = reg_term
            return loss + self.l2_reg * reg_term

    class CE_Normalize(nn.CrossEntropyLoss):
        def forward(self, y_hat, target, normalize=False):
            y_hat = y_hat[:, -2, :]
            if normalize:
                y_hat = F.normalize(y_hat, p=2, dim=1)
            return super().forward(y_hat, target[:, -2])

    criterion = CE_Normalize()
    reg_cel = CELReg(args.weight_decay, model)
    reg_criterion = lambda y_hat, target, *args, **kwargs: reg_cel(
        y_hat[:, -2, :], target[:, -2]
    )

    """
    Train
    """
    steps_per_epoch = len(train_dataloader)
    (
        interpolated_99,
        interpolated_100,
        generalized_99,
        generalized_100,
        generalized_90,
    ) = (False, False, False, False, False)
    train_step_counter = 0

    # initial validation step
    log_dict = validate(
        args,
        model,
        train_dataloader,
        val_dataloader,
        zero_enc,
        one_enc,
        criterion,
        0,
        0,
        generalized_99,
        generalized_100,
        generalized_90,
    )
    wandb.log(log_dict)

    # train
    for epoch in tqdm(range(1, int(OPTIMIZATION_BUDGET / steps_per_epoch) + 1)):
        model.train()
        epoch_data = []
        for i, batch in enumerate(train_dataloader):
            train_step_counter += 1

            step_data = train_step(
                args, model, optimizer, reg_criterion, criterion, batch
            )
            epoch_data.append(step_data)

            if train_step_counter < 1000 and i != len(train_dataloader) - 1:
                log_dict = validate(
                    args,
                    model,
                    train_dataloader,
                    val_dataloader,
                    zero_enc,
                    one_enc,
                    criterion,
                    epoch,
                    train_step_counter,
                    generalized_99,
                    generalized_100,
                    generalized_90,
                )
                log_dict.update(
                    get_train_logdict(*step_data, epoch, train_step_counter)
                )
                wandb.log(log_dict)

        if train_step_counter >= 1000:
            step_data = np.mean(epoch_data, axis=0)

        log_dict = validate(
            args,
            model,
            train_dataloader,
            val_dataloader,
            zero_enc,
            one_enc,
            criterion,
            epoch,
            train_step_counter,
            generalized_99,
            generalized_100,
            generalized_90,
        )
        log_dict.update(get_train_logdict(*step_data, epoch, train_step_counter))
        log_dict.update(
            get_interpolation_logdict(
                step_data[0],
                epoch,
                train_step_counter,
                interpolated_99,
                interpolated_100,
            )
        )
        wandb.log(log_dict)


if __name__ == "__main__":
    run(arguments)
