from geometry.metric import MetricField
import torch

u_metric = MetricField(
    # use of stack to prevent breaking the autograd history
    lambda x, y: torch.stack(
        [
            torch.stack([x * y + 0.1, torch.sin(x * y)]),
            torch.stack([torch.sin(x * y), 2 * x * y + 0.1]),
        ],
    )
)
x = torch.tensor([1.3, 3.21])

conn_coeffs = u_metric.christoffels()(x)
print(conn_coeffs)
print(conn_coeffs.shape)
print(conn_coeffs[1, :, :])
