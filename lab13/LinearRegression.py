import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1) 


split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# TODO: Implement Linear Regression from scratch
class LinearRegression:
    def __init__(self):
        self.w = None
        self.b = None
        
    def fit(self, X, y):
       
        X_mean = np.mean(X)
        y_mean = np.mean(y)
        
        # Calculate the numerator and denominator for the slope
        numerator = np.sum((X - X_mean) * (y - y_mean))
        denominator = np.sum((X - X_mean)**2)
        
        # Calculate slope (w) and intercept (b)
        self.w = numerator / denominator
        self.b = y_mean - self.w * X_mean
        
    def predict(self, X):
        return self.w * X + self.b


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


plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='green', label='Testing data')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression')
plt.show()
