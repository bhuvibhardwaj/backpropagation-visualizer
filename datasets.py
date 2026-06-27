import numpy as np
from sklearn.datasets import make_moons, make_circles


def generate_dataset(name, n_samples=100):
    if name == 'xor':
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        return X, y
    elif name == 'and':
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [0], [0], [1]])
        return X, y
    elif name == 'or':
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [1]])
        return X, y
    elif name == 'spiral':
        return make_spiral(n_samples)
    elif name == 'two_moons':
        X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
        y = y.reshape(-1, 1)
        return X, y
    elif name == 'concentric_circles':
        X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=42)
        y = y.reshape(-1, 1)
        return X, y
    return None, None


def make_spiral(n_samples=100, noise=0.1):
    n = n_samples // 2
    X = []
    y = []
    
    for i in range(n):
        r = i / n * 5
        t = 1.75 * i / n * 2 * np.pi + np.random.normal(0, noise)
        X.append([r * np.sin(t), r * np.cos(t)])
        y.append([0])
        
        t = 1.75 * i / n * 2 * np.pi + np.pi + np.random.normal(0, noise)
        X.append([r * np.sin(t), r * np.cos(t)])
        y.append([1])
    
    return np.array(X), np.array(y)
