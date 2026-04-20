import torch
import torch.nn as nn

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
    method = WeightMethods(
        "vargrad_psmgd",
        n_tasks=n_tasks,
        device=device,
        beta=0.9,
        update_weights_every=3,
        weight_smoothing=0.5,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    for step in range(6):
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
