import numpy as np
import matplotlib.pyplot as plt

# Generate non-linear data
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 0.5 * X**3 + X**2 - 2 * X + 2 + np.random.randn(100, 1) * 3

# Split data
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# TODO: Implement a function to create polynomial features
def create_poly_features(X, degree):
    """
    Create polynomial features up to specified degree.
    
    Parameters:
    X (array): Input features of shape (n_samples, 1)
    degree (int): Maximum polynomial degree
    
    Returns:
    array: Polynomial features of shape (n_samples, degree)
    """
    # Your code here
    X_poly = np.zeros((X.shape[0], degree))
    for d in range(degree):
        X_poly[:, d] = X[:, 0]**(d+1)
    return X_poly

# TODO: Implement polynomial regression using linear regression
def poly_regression(X_train, y_train, X_test, degree, lambda_reg=0.1):

    # polynomial features
    X_train_poly = create_poly_features(X_train, degree)
    X_test_poly = create_poly_features(X_test, degree)
    
   
    X_train_b = np.c_[np.ones((X_train_poly.shape[0], 1)), X_train_poly]
    
    # regularization
    # n_features = X_train_b.shape[1]
    # regularization_matrix = lambda_reg * np.eye(n_features)
    # regularization_matrix[0, 0] = 0

    # normal equation
    theta = np.linalg.inv(X_train_b.T.dot(X_train_b)).dot(X_train_b.T).dot(y_train)
    # theta = np.linalg.inv(X_train_b.T.dot(X_train_b) + regularization_matrix).dot(X_train_b.T).dot(y_train)
    
    bias = theta[0, 0]
    coeffs = theta[1:, 0]
    
   
    train_pred = X_train_poly.dot(coeffs) + bias
    test_pred = X_test_poly.dot(coeffs) + bias
    
    return train_pred, test_pred, coeffs, bias

# Test with different polynomial degrees
degrees = [1, 3, 7, 15]
plt.figure(figsize=(15, 10))

for i, degree in enumerate(degrees):
    # Fit and predictions
    train_pred, test_pred, coeffs, bias = poly_regression(X_train, y_train, X_test, degree)
    
    # MSE
    train_mse = np.mean((y_train - train_pred.reshape(-1, 1))**2)
    test_mse = np.mean((y_test - test_pred.reshape(-1, 1))**2)
    
   
    plt.subplot(2, 2, i+1)
    plt.scatter(X_train, y_train, color='blue', alpha=0.6, label='Training data')
    plt.scatter(X_test, y_test, color='green', alpha=0.6, label='Testing data')
    
    # Sort
    X_sorted = np.sort(X, axis=0)
    X_poly = create_poly_features(X_sorted, degree)
    y_poly = X_poly.dot(coeffs) + bias
    
    plt.plot(X_sorted, y_poly, color='red', linewidth=2, label=f'Polynomial (degree={degree})')
    plt.title(f'Degree {degree}\nTrain MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()

plt.tight_layout()
plt.show()

