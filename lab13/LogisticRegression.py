import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)


split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# TODO: Implement Logistic Regression from scratch
class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
       
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.iterations):
           
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
           
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
           
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict_probability(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_probability(X) >= threshold).astype(int)


model = LogisticRegression(learning_rate=0.1, iterations=1000)
model.fit(X_train, y_train)


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate accuracy
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

train_acc = accuracy(y_train, y_train_pred)
test_acc = accuracy(y_test, y_test_pred)

print(f"Training accuracy: {train_acc:.4f}")
print(f"Testing accuracy: {test_acc:.4f}")


plt.figure(figsize=(10, 6))


plt.scatter(X_train[y_train[:, 0] == 0][:, 0], X_train[y_train[:, 0] == 0][:, 1], 
            color='blue', label='Class 0 (train)')
plt.scatter(X_train[y_train[:, 0] == 1][:, 0], X_train[y_train[:, 0] == 1][:, 1], 
            color='red', label='Class 1 (train)')


plt.scatter(X_test[y_test[:, 0] == 0][:, 0], X_test[y_test[:, 0] == 0][:, 1], 
            color='blue', marker='x', label='Class 0 (test)')
plt.scatter(X_test[y_test[:, 0] == 1][:, 0], X_test[y_test[:, 0] == 1][:, 1], 
            color='red', marker='x', label='Class 1 (test)')


x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='green', levels=[0.5], label='Decision Boundary')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Logistic Regression Decision Boundary')
plt.show()