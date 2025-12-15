import numpy as np

np.random.seed(42)
m, n = 100, 2 
X = 2 * np.random.rand(m, n) 
true_w = np.array([3, -2]) 
true_b = 4  
y = X @ true_w + true_b + np.random.randn(m)  

split_idx = int(0.8 * m)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

class LinearRegression:
    def __init__(self):
        self.w = None 
        self.b = None 
        
    def fit(self, X, y):
        X = np.atleast_2d(X)
        y = np.ravel(y)
        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        self.b = theta[0]  
        self.w = theta[1:]  
        
    def predict(self, X):
        X = np.atleast_2d(X)
        return X @ self.w + self.b
    
model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"Model parameters: w = {model.w}, b = {model.b}")
print(f"Training MSE: {train_mse}")
print(f"Testing MSE: {test_mse}")