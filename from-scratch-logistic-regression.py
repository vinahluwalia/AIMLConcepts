import numpy as np

def sigmoid(z):
    """
    Compute the sigmoid function for input z.
    Parameters:
    z: array-like, shape (n_samples,) or (n_samples, n_features)
    Returns:
    sigmoid of z, same shape as z
    """
    # Clip z to avoid overflow in exp
    z = np.clip(z, -500, 500)
    # Compute the sigmoid function for input z
    return 1.0 / (1.0 + np.exp(-z))

def compute_cost(X, y, theta):
    # Compute the cost function (e.g., log-likelihood) for logistic regression
    m = y.shape[0]  # Number of samples
    h = sigmoid(X.dot(theta)) # Hypothesis function
    # Add epsion to avoid log(0)
    epsilon = 1e-15
    cost = -1/m * ( y.dot(np.log(h + epsilon)) + (1-y).dot(np.log(1 - h + epsilon)))

    return cost

def compute_gradient(X, y, theta):
    # Compute the gradient of the cost function for gradient descent
    m = y.shape[0]  # Number of samples
    h = sigmoid(X.dot(theta))  # Hypothesis function
    gradient = (1/m) * X.T.dot(h-y) # Gradient calculation
    # The gradient is the derivative of the cost function with respect to theta
    # It is the average of the product of the input features and the error (h - y)
    # where h is the predicted probability and y is the actual label.    
    return gradient

def logistic_regression(X, y, learning_rate=0.01, max_iters=100, tol=1e-4):
    # Main logistic regression algorithm using gradient descent
    # X: input features (n_samples, n_features)
    # y: target labels (n_samples,)
    # Initialize parameters (theta) to zeros
    theta = np.zeros(X.shape[1])
    
    for iteration in range(max_iters):
        # Compute cost and gradient
        cost = compute_cost(X, y, theta)
        gradient = compute_gradient(X, y, theta)
        
        # Update parameters using gradient descent
        new_theta = theta - learning_rate * gradient
        
        # Check for convergence (if parameter change is small)
        if np.all(np.abs(new_theta - theta) <= tol):
            print(f"Converged in {iteration+1} iterations; cost = {cost:.6f}")
            theta = new_theta            
            break
            
        # Update parameters for next iteration
        theta = new_theta
    
    else:
        print(f"Reached max_iters={max_iters}; cost = {cost:.6f}")
    
    return theta

# Example Usage
X = np.random.rand(100, 2)  # 100 samples, 2 features
y = np.random.randint(0, 2, 100)  # Binary labels (0 or 1)
theta = logistic_regression(X, y)
print("Parameters:", theta)