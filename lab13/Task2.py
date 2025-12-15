import numpy as np
import matplotlib.pyplot as plt

# Generate classification data
np.random.seed(42)
X = np.random.randn(200, 2)
y = (np.sum(X**2, axis=1) < 2).astype(int).reshape(-1, 1)  # Circle pattern

# Split data
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# TODO: Implement a neural network with one hidden layer
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):

        self.learning_rate = learning_rate
       
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500))) 
    
    def sigmoid_derivative(self, x):
       
        s = self.sigmoid(x)
        return s * (1 - s)
    
    # def relu_derivative(self, x):
        # return (x > 0).astype(int)
    
    def forward(self, X):
        """Forward pass through the network"""
        # First layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        # ReLu : self.a1 = np.maximum(0, self.z1)
        
        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, output):
        """Backward pass to update weights"""
        m = X.shape[0]
        
        # Output layer gradients
        dz2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.z1)
        # ReLu: dz1 = np.dot(dz2, self.W2.T) * self.relu_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def train(self, X, y, epochs):
        """Train the neural network"""
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute loss
            loss = -np.mean(y * np.log(output + 1e-8) + (1 - y) * np.log(1 - output + 1e-8))
            losses.append(loss)
            
            # Backward pass
            self.backward(X, y, output)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
                
        return losses
    
    def predict(self, X, threshold=0.5):
        """Make predictions"""
        output = self.forward(X)
        return (output >= threshold).astype(int)


nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)
losses = nn.train(X_train, y_train, epochs=1000)


y_train_pred = nn.predict(X_train)
y_test_pred = nn.predict(X_test)

train_acc = np.mean(y_train == y_train_pred)
test_acc = np.mean(y_test == y_test_pred)

print(f"Training accuracy: {train_acc:.4f}")
print(f"Testing accuracy: {test_acc:.4f}")


plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.title('Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))


plt.scatter(X_train[y_train[:, 0] == 0][:, 0], X_train[y_train[:, 0] == 0][:, 1], 
            color='blue', label='Class 0 (train)')
plt.scatter(X_train[y_train[:, 0] == 1][:, 0], X_train[y_train[:, 0] == 1][:, 1], 
            color='red', label='Class 1 (train)')


x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))


Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.contourf(xx, yy, Z, alpha=0.3)
plt.contour(xx, yy, Z, colors='green', linewidths=0.5)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Neural Network Decision Boundary')
plt.show()

