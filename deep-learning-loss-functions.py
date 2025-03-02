
# Loss Functions
import numpy as np
import torch
import torch.nn as nn

# 1. Mean Squared Error Loss
# Sum of squared differences between the predicted vector y and the ground truth vector y_hat
y_pred = torch.tensor([0.6, 1.29, 1.99, 2.69, 3.4])
y_actual = torch.tensor([1,1,2,2,4])
mse_loss = torch.sub(y_pred, y_actual)
mse_loss = torch.pow(mse_loss, 2)
mse_loss = torch.mean(mse_loss)
print(f"Manually computed MSE Loss:", mse_loss)
# Compare against PyTorch's implementation
torch_mse_loss = nn.MSELoss()
torch_mse_loss = torch_mse_loss(y_pred, y_actual)
print(f"Pytorch implemented MSE Loss:", torch_mse_loss)

# 2. Cross-Entropy Loss - Classification Loss
# y_hat = prediciton vector which has continuous values
# y_actual = ground truth vector which has binary values
y_pred = torch.tensor([0.1, 0.3, 0.4, 0.4, 0.2])
y_actual = torch.tensor([1,0,0,0,0])
cross_entropy_loss = torch.sum(-y_actual * torch.log(y_pred))
print(cross_entropy_loss)
# Can also use PyTorch's implementation
torch_cross_entropy_loss = nn.CrossEntropyLoss()
torch_cross_entropy_loss = torch_cross_entropy_loss(y_pred, y_actual)
print(torch_cross_entropy_loss)

# 2. Mean Absolute Error Loss

# 3. Hinge Loss
# 5. Kullback Leibler Divergence Loss
# 6. Negative Log Likelihood Loss
# 7. Poisson Loss
