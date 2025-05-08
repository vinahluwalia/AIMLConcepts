
import torch

class BatchNorm1dCustom:
    def __init__(self, num_features, eps = 1e-5, momentum = 0.1):
        """
        Args:
            num_features: Number of features in the input
            eps: Small constant for numerical stability
            momentum: Momentum for updating running mean and variance            
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters (gamma and beta)
        self.gamma = torch.ones(num_features, requires_grad = True)
        self.beta = torch.zeros(num_features, requires_grad = True)

        # Running statistics (not trainiable)
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

        # Track training mode
        self.training = True
    
    def forward(self, x):
        """
        Forward pass for batch norm
        Args:
            x: Input tensor for shape (batch_seize, num_features)
        Returns:
            Normalized and scaled output tensor            
        """
        if self.training:
            # Compute batch statistics
            batch_mean = x.mean(dim = 0)
            batch_var = x.var(dim = 0, unbiased = False)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
        
            # Normalize
            

