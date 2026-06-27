# Backpropagation Visualizer

Interactive neural network training visualizer showing gradient flow and weight updates in real time.

## Features

- **Network Configuration**: Adjust number of hidden layers, neurons per layer, learning rate, activation function, and loss function
- **Training Controls**: Initialize, train one epoch, or train all epochs at once
- **Real-time Visualization**:
  - Network graph with neuron activations, biases, and deltas
  - Loss curve over epochs
  - Gradient flow per layer
  - Weight distribution histogram
  - Decision boundary for 2D datasets
  - Gradient norms history per layer
- **Datasets**: XOR, AND, OR, Spiral, Two Moons, Concentric Circles

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the app:
```bash
streamlit run app.py
```

## License

MIT

## Topics

backpropagation, neural-networks, deep-learning, streamlit, python, machine-learning, visualization
