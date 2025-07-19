from torch.optim.lr_scheduler import LambdaLR
import torch

def get_cnn_refiner_scheduler(optimizer, gamma=0.999, apply_every=40):
    def make_lambda(gamma_val, apply_every_val):
        def f(iter):
            return float(gamma_val) ** (iter // apply_every_val)
        return f

    num_groups = len(optimizer.param_groups)
    scheduler_lambdas = [make_lambda(gamma, apply_every) for _ in range(num_groups)]

    # Debug print to check if param groups and lambdas match
    print(f"Scheduler: {num_groups} param groups, {len(scheduler_lambdas)} lambdas")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_lambdas)
