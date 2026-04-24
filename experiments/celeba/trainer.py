import os
from argparse import ArgumentParser

import numpy as np
import time
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.celeba.data import CelebaDataset
from experiments.celeba.models import Network
from experiments.utils import (
    build_experiment_output_stem,
    common_parser,
    extract_weight_method_parameters_from_args,
    get_device,
    set_logger,
    set_seed,
    str2bool,
)
from methods.weight_methods import WeightMethods


class CelebaMetrics():
    """
    CelebA metric accumulator.
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.tp = 0.0 
        self.fp = 0.0 
        self.fn = 0.0 
        
    def incr(self, y_preds, ys):
        # y_preds: [ y_pred (batch, 1) ] x 40
        # ys     : [ y_pred (batch, 1) ] x 40
        y_preds  = torch.stack(y_preds).detach() # (40, batch, 1)
        ys       = torch.stack(ys).detach()      # (40, batch, 1)
        y_preds  = y_preds.gt(0.5).float()
        self.tp += (y_preds * ys).sum([1,2]) # (40,)
        self.fp += (y_preds * (1 - ys)).sum([1,2])
        self.fn += ((1 - y_preds) * ys).sum([1,2])
                
    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall    = self.tp / (self.tp + self.fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        return f1.cpu().numpy()


def main(path, lr, bs, device):
    # we only train for specific task
    model = Network().to(device)
    
    train_set = CelebaDataset(data_dir=path, split='train')
    val_set   = CelebaDataset(data_dir=path, split='val')
    test_set  = CelebaDataset(data_dir=path, split='test')

    train_loader = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=bs, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(
            dataset=val_set, batch_size=bs, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=bs, shuffle=False, num_workers=2)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs    = args.n_epochs

    metrics   = np.zeros([epochs, 40], dtype=np.float32) # test_f1
    metric    = CelebaMetrics()
    loss_fn   = torch.nn.BCELoss()

    # weight method
    weight_methods_parameters = extract_weight_method_parameters_from_args(args)
    weight_method = WeightMethods(
        args.method, n_tasks=40, device=device, **weight_methods_parameters[args.method]
    )

    best_val_f1 = 0.0
    best_epoch = None

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)
    n_test_batches = len(test_loader)

    for epoch in range(epochs):
        # training
        model.train()
        t0 = time.time()
        print(f"[info] epoch {epoch + 1}/{epochs} | starting training", flush=True)
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device)
            y = [y_.to(device) for y_ in y]
            y_ = model(x)
            losses = torch.stack([loss_fn(y_task_pred, y_task) for (y_task_pred, y_task) in zip(y_, y)])
            optimizer.zero_grad()
            loss, extra_outputs = weight_method.backward(
                losses=losses,
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(model.task_specific_parameters()),
                last_shared_parameters=list(model.last_shared_parameters()),
            )
            optimizer.step()
            if "famo" in args.method:
                with torch.no_grad():
                    y_ = model(x)
                    new_losses = torch.stack([loss_fn(y_task_pred, y_task) for (y_task_pred, y_task) in zip(y_, y)])
                    weight_method.method.update(new_losses.detach())
            print(
                f"[train] epoch {epoch + 1}/{epochs} "
                f"batch {batch_idx}/{n_train_batches} "
                f"mean_loss={losses.mean().item():.6f}",
                flush=True,
            )
        t1 = time.time()

        model.eval()
        # validation
        metric.reset()
        print(f"[info] epoch {epoch + 1}/{epochs} | starting validation", flush=True)
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader, start=1):
                x = x.to(device)
                y = [y_.to(device) for y_ in y]
                y_ = model(x)
                losses = torch.stack([loss_fn(y_task_pred, y_task) for (y_task_pred, y_task) in zip(y_, y)])
                metric.incr(y_, y)
                print(
                    f"[val] epoch {epoch + 1}/{epochs} "
                    f"batch {batch_idx}/{n_val_batches} "
                    f"mean_loss={losses.mean().item():.6f}",
                    flush=True,
                )
        val_f1 = metric.result()
        if val_f1.mean() > best_val_f1:
            best_val_f1 = val_f1.mean()
            best_epoch = epoch

        # testing
        metric.reset()
        print(f"[info] epoch {epoch + 1}/{epochs} | starting test", flush=True)
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_loader, start=1):
                x = x.to(device)
                y = [y_.to(device) for y_ in y]
                y_ = model(x)
                losses = torch.stack([loss_fn(y_task_pred, y_task) for (y_task_pred, y_task) in zip(y_, y)])
                metric.incr(y_, y)
                print(
                    f"[test] epoch {epoch + 1}/{epochs} "
                    f"batch {batch_idx}/{n_test_batches} "
                    f"mean_loss={losses.mean().item():.6f}",
                    flush=True,
                )
        test_f1 = metric.result()
        metrics[epoch] = test_f1

        t2 = time.time()
        print(
            f"[info] epoch {epoch+1} | train takes {(t1-t0)/60:.1f} min | test takes {(t2-t1)/60:.1f} min",
            flush=True,
        )
        
        if "famo" in args.method:
            name = f"{args.method}_gamma{args.gamma}_sd{args.seed}"
        elif "fairgrad" in args.method:
            name = f"{args.method}_alpha{args.alpha}_sd{args.seed}"
        else:
            name = f"{args.method}_sd{args.seed}"
        
        torch.save({"metric": metrics, "best_epoch": best_epoch}, os.path.join(args.save_dir, f"{name}.stats"))


if __name__ == "__main__":
    parser = ArgumentParser("Celeba", parents=[common_parser])
    parser.set_defaults(
        data_path=os.path.join(os.getcwd(), "dataset"),
        lr=3e-4,
        n_epochs=15,
        batch_size=256,
        save_dir="/root/autodl-tmp/exp_logs_save/modular/celeba/save",
    )
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)
    device = get_device(gpus=args.gpu)
    main(path=args.data_path,
         lr=args.lr,
         bs=args.batch_size,
         device=device)
