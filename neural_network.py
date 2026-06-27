import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes, activation='relu', loss_function='mse', learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.loss_name = loss_function
        self.learning_rate = learning_rate
        
        self.weights = []
        self.biases = []
        self.activations = []
        self.z_values = []
        self.gradients = {
            'weights': [],
            'biases': [],
            'deltas': []
        }
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        np.random.seed(42)
        self.weights = []
        self.biases = []
        
        for i in range(len(self.layer_sizes) - 1):
            weight = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * 0.1
            bias = np.zeros((1, self.layer_sizes[i+1]))
            self.weights.append(weight)
            self.biases.append(bias)
    
    def activation(self, z):
        if self.activation_name == 'relu':
            return np.maximum(0, z)
        elif self.activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation_name == 'tanh':
            return np.tanh(z)
        return z
    
    def activation_derivative(self, z):
        if self.activation_name == 'relu':
            return (z > 0).astype(float)
        elif self.activation_name == 'sigmoid':
            s = 1 / (1 + np.exp(-z))
            return s * (1 - s)
        elif self.activation_name == 'tanh':
            return 1 - np.tanh(z) ** 2
        return np.ones_like(z)
    
    def loss(self, y_true, y_pred):
        if self.loss_name == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        elif self.loss_name == 'binary_crossentropy':
            y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return 0
    
    def loss_derivative(self, y_true, y_pred):
        if self.loss_name == 'mse':
            return 2 * (y_pred - y_true) / y_true.shape[0]
        elif self.loss_name == 'binary_crossentropy':
            y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
            return (y_pred - y_true) / (y_pred * (1 - y_pred)) / y_true.shape[0]
        return 0
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            if i == len(self.weights) - 1:
                if self.loss_name == 'binary_crossentropy':
                    a = 1 / (1 + np.exp(-z))
                else:
                    a = z
            else:
                a = self.activation(z)
            self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, y_true):
        self.gradients = {
            'weights': [],
            'biases': [],
            'deltas': []
        }
        
        delta = self.loss_derivative(y_true, self.activations[-1])
        self.gradients['deltas'].append(delta)
        
        for i in reversed(range(len(self.weights))):
            dw = np.dot(self.activations[i].T, delta)
            db = np.sum(delta, axis=0, keepdims=True)
            self.gradients['weights'].insert(0, dw)
            self.gradients['biases'].insert(0, db)
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(self.z_values[i-1])
                self.gradients['deltas'].insert(0, delta)
    
    def update_weights(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.gradients['weights'][i]
            self.biases[i] -= self.learning_rate * self.gradients['biases'][i]
    
    def train_step(self, X, y):
        y_pred = self.forward(X)
        loss = self.loss(y, y_pred)
        self.backward(y)
        self.update_weights()
        return loss, y_pred
    
    def get_gradient_norms(self):
        norms = []
        for dw in self.gradients['weights']:
            norms.append(np.linalg.norm(dw))
        return norms
    
    def get_weight_update_magnitude(self):
        total_update = 0
        for dw in self.gradients['weights']:
            total_update += np.sum(np.abs(dw * self.learning_rate))
        return total_update
