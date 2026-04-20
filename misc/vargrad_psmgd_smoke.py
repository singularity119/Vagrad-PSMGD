from pathlib import Path
import sys

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods.weight_methods import WeightMethods


class ToyMTL(nn.Module):
    def __init__(self, in_dim=8, hidden_dim=16, n_tasks=3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList(nn.Linear(hidden_dim, 1) for _ in range(n_tasks))

    def forward(self, x):
        features = self.shared(x)
        return [head(features) for head in self.heads]

    def shared_parameters(self):
        return self.shared.parameters()

    def task_specific_parameters(self):
        return self.heads.parameters()

    def last_shared_parameters(self):
        return []


def main():
    torch.manual_seed(0)
    device = torch.device("cpu")
    n_tasks = 3

    model = ToyMTL(n_tasks=n_tasks).to(device)
    method_specs = [
        (
            "modular",
            dict(
                preprocessing="identity",
                solver="uniform",
                scheduler="every_step",
                use_momentum=False,
            ),
        ),
        (
            "modular",
            dict(
                preprocessing="vargrad",
                solver="fairgrad",
                scheduler="psmgd_periodic",
                use_momentum=True,
                beta_v=0.9,
                beta_m=0.9,
                psmgd_R=3,
                psmgd_alpha=0.5,
                alpha=1.0,
            ),
        )
    ]

    for method_name, method_kwargs in method_specs:
        print(f"== {method_name} {method_kwargs} ==")
        method = WeightMethods(
            method_name,
            n_tasks=n_tasks,
            device=device,
            **method_kwargs,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

        for step in range(4):
            x = torch.randn(12, 8, device=device)
            outputs = model(x)
            losses = torch.stack(
                [
                    (outputs[0] - 0.5).pow(2).mean(),
                    (outputs[1] + 0.25).abs().mean(),
                    (outputs[2]).pow(2).mean(),
                ]
            )

            optimizer.zero_grad()
            loss, extra = method.backward(
                losses=losses,
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(model.task_specific_parameters()),
            )
            optimizer.step()

            print(
                f"step={step} loss={loss.item():.4f} "
                f"weights={extra['weights'].tolist()} updated={extra['updated_weights']}"
            )


if __name__ == "__main__":
    main()
