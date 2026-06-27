import streamlit as st
import numpy as np
import plotly.graph_objects as go
from neural_network import NeuralNetwork
from datasets import generate_dataset

st.set_page_config(
    page_title="Backpropagation Visualizer",
    layout="wide",
    page_icon=None,
    initial_sidebar_state="expanded"
)

# Custom CSS for professional design
st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #0B0F14;
        color: #F5F7FA;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0B0F14;
        border-right: 1px solid #232A33;
    }
    
    [data-testid="stSidebarContent"] {
        padding-top: 2rem;
    }
    
    /* Widget labels */
    .st-c7, .st-e2, .st-e0, .st-bu, .st-ec {
        color: #8D99A6;
        font-size: 13px;
        font-weight: 500;
    }
    
    /* Selectbox and inputs */
    .st-ay, .st-ae, .st-af {
        background-color: #11161C !important;
        border: 1px solid #232A33 !important;
        color: #F5F7FA !important;
        border-radius: 8px !important;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        height: 42px;
        background-color: #11161C;
        color: #F5F7FA;
        border: 1px solid #232A33;
        border-radius: 8px;
        font-weight: 500;
        font-size: 14px;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: #181E26;
        border-color: #2F3A47;
    }
    
    .stButton>button:active {
        background-color: #1F2730;
    }
    
    /* Primary button */
    div[data-testid="stVerticalBlock"] > div:nth-child(10) .stButton>button,
    div[data-testid="stVerticalBlock"] > div:nth-child(11) .stButton>button {
        background-color: #4F8EF7;
        border-color: #4F8EF7;
    }
    
    div[data-testid="stVerticalBlock"] > div:nth-child(10) .stButton>button:hover,
    div[data-testid="stVerticalBlock"] > div:nth-child(11) .stButton>button:hover {
        background-color: #3A7BD6;
        border-color: #3A7BD6;
    }
    
    /* Danger button */
    div[data-testid="stVerticalBlock"] > div:nth-child(13) .stButton>button {
        border-color: #E74C3C;
        color: #E74C3C;
    }
    
    div[data-testid="stVerticalBlock"] > div:nth-child(13) .stButton>button:hover {
        background-color: rgba(231,76,60,0.1);
    }
    
    /* Sliders */
    .st-d7, .st-d8, .st-d9, .st-da {
        color: #F5F7FA;
    }
    
    .st-dz {
        background-color: #232A33;
    }
    
    .st-dy {
        background-color: #4F8EF7;
    }
    
    /* Dividers */
    hr {
        border-color: #232A33;
        margin: 1.5rem 0;
    }
    
    /* Metric panels */
    [data-testid="stMetric"] {
        background-color: #11161C;
        border: 1px solid #232A33;
        border-radius: 10px;
        padding: 20px;
    }
    
    [data-testid="stMetricLabel"] {
        color: #8D99A6;
        font-size: 13px;
        font-weight: 500;
    }
    
    [data-testid="stMetricValue"] {
        color: #F5F7FA;
        font-size: 36px;
        font-weight: 700;
    }
    
    /* Hide default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Headings */
    h1 {
        font-size: 48px !important;
        font-weight: 700 !important;
        color: #F5F7FA !important;
        margin-bottom: 0.5rem !important;
    }
    
    h2 {
        font-size: 22px !important;
        font-weight: 600 !important;
        color: #F5F7FA !important;
        margin-top: 0 !important;
        margin-bottom: 1rem !important;
    }
    
    /* Subheader text */
    .subheader-text {
        color: #8D99A6;
        font-size: 16px;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.nn = None
        st.session_state.loss_history = []
        st.session_state.accuracy_history = []
        st.session_state.gradient_norms_history = []
        st.session_state.current_epoch = 0
        st.session_state.X = None
        st.session_state.y = None
        st.session_state.dataset_name = "xor"
        st.session_state.n_hidden_layers = 2
        st.session_state.neurons_per_layer = 4
        st.session_state.learning_rate = 0.1
        st.session_state.activation = "relu"
        st.session_state.loss_function = "mse"
        st.session_state.epochs = 1000
        st.session_state.batch_size = 4
        st.session_state.animation_speed = 1

init_session_state()

# Sidebar
with st.sidebar:
    st.markdown("### Dataset")
    st.session_state.dataset_name = st.selectbox(
        "",
        ["xor", "and", "or", "spiral", "two_moons", "concentric_circles"],
        index=["xor", "and", "or", "spiral", "two_moons", "concentric_circles"].index(st.session_state.dataset_name),
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("### Model")
    st.session_state.n_hidden_layers = st.slider(
        "Hidden Layers",
        1, 5,
        st.session_state.n_hidden_layers,
        label_visibility="visible"
    )
    st.session_state.neurons_per_layer = st.slider(
        "Neurons per Hidden Layer",
        1, 10,
        st.session_state.neurons_per_layer,
        label_visibility="visible"
    )
    st.session_state.learning_rate = st.select_slider(
        "Learning Rate",
        [0.001, 0.01, 0.1, 0.5, 1.0],
        st.session_state.learning_rate,
        label_visibility="visible"
    )
    st.session_state.activation = st.selectbox(
        "Activation",
        ["relu", "sigmoid", "tanh"],
        index=["relu", "sigmoid", "tanh"].index(st.session_state.activation),
        label_visibility="visible"
    )
    st.session_state.loss_function = st.selectbox(
        "Loss Function",
        ["mse", "binary_crossentropy"],
        index=["mse", "binary_crossentropy"].index(st.session_state.loss_function),
        label_visibility="visible"
    )
    
    st.markdown("---")
    
    st.markdown("### Training")
    st.session_state.epochs = st.slider(
        "Epochs",
        100, 5000,
        st.session_state.epochs,
        label_visibility="visible"
    )
    st.session_state.batch_size = st.select_slider(
        "Batch Size",
        [1, 4, 8, 16, 32],
        st.session_state.batch_size,
        label_visibility="visible"
    )
    st.session_state.animation_speed = st.select_slider(
        "Animation Speed",
        [0.25, 0.5, 1, 2, 5],
        st.session_state.animation_speed,
        label_visibility="visible"
    )
    
    st.markdown("---")
    
    st.markdown("### Actions")
    if st.button("Initialize"):
        st.session_state.X, st.session_state.y = generate_dataset(st.session_state.dataset_name)
        input_size = st.session_state.X.shape[1]
        output_size = st.session_state.y.shape[1]
        layer_sizes = [input_size] + [st.session_state.neurons_per_layer] * st.session_state.n_hidden_layers + [output_size]
        st.session_state.nn = NeuralNetwork(layer_sizes, st.session_state.activation, st.session_state.loss_function, st.session_state.learning_rate)
        st.session_state.nn.forward(st.session_state.X)
        st.session_state.loss_history = []
        st.session_state.accuracy_history = []
        st.session_state.gradient_norms_history = []
        st.session_state.current_epoch = 0
        st.rerun()
    
    if st.button("Train One Epoch"):
        if st.session_state.nn and st.session_state.X is not None:
            loss, y_pred = st.session_state.nn.train_step(st.session_state.X, st.session_state.y)
            st.session_state.loss_history.append(loss)
            st.session_state.gradient_norms_history.append(st.session_state.nn.get_gradient_norms())
            
            if st.session_state.loss_function == 'binary_crossentropy':
                y_pred_binary = (y_pred > 0.5).astype(int)
                accuracy = np.mean(y_pred_binary == st.session_state.y)
                st.session_state.accuracy_history.append(accuracy)
            st.session_state.current_epoch += 1
            st.rerun()
    
    if st.button("Train All"):
        if st.session_state.nn and st.session_state.X is not None:
            progress_bar = st.progress(0)
            for epoch in range(st.session_state.epochs):
                loss, y_pred = st.session_state.nn.train_step(st.session_state.X, st.session_state.y)
                st.session_state.loss_history.append(loss)
                st.session_state.gradient_norms_history.append(st.session_state.nn.get_gradient_norms())
                
                if st.session_state.loss_function == 'binary_crossentropy':
                    y_pred_binary = (y_pred > 0.5).astype(int)
                    accuracy = np.mean(y_pred_binary == st.session_state.y)
                    st.session_state.accuracy_history.append(accuracy)
                st.session_state.current_epoch += 1
                progress_bar.progress((epoch + 1) / st.session_state.epochs)
            st.rerun()
    
    st.markdown("---")
    
    if st.button("Reset"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Main content
st.markdown("# Backpropagation Visualizer")
st.markdown('<div class="subheader-text">Interactive visualization of gradient propagation in feedforward neural networks.</div>', unsafe_allow_html=True)

# Metrics row
st.markdown("## Training Metrics")
metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

with metrics_col1:
    st.metric("Current Epoch", st.session_state.current_epoch)

with metrics_col2:
    if st.session_state.loss_history:
        st.metric("Loss", f"{st.session_state.loss_history[-1]:.4f}")
    else:
        st.metric("Loss", "N/A")

with metrics_col3:
    if st.session_state.accuracy_history:
        st.metric("Accuracy", f"{st.session_state.accuracy_history[-1]:.2%}")
    else:
        st.metric("Accuracy", "N/A")

with metrics_col4:
    st.metric("Learning Rate", st.session_state.learning_rate)

st.markdown("---")

# Top row: Network + 1 graph
top_col1, top_col2 = st.columns([3, 2])

with top_col1:
    st.markdown("## Network")
    if st.session_state.nn:
        layer_sizes = st.session_state.nn.layer_sizes
        fig = go.Figure()
        
        node_x = []
        node_y = []
        node_labels = []
        node_colors = []
        
        max_neurons = max(layer_sizes)
        for layer_idx, n_neurons in enumerate(layer_sizes):
            x = layer_idx * 2
            for neuron_idx in range(n_neurons):
                y = (max_neurons - 1) / 2 - neuron_idx
                node_x.append(x)
                node_y.append(y)
                
                label_parts = [f"Neuron ID: L{layer_idx}N{neuron_idx}"]
                
                if layer_idx < len(st.session_state.nn.activations):
                    activation = st.session_state.nn.activations[layer_idx]
                    if activation is not None and len(activation) > 0:
                        val = activation[0, neuron_idx] if len(activation.shape) > 1 else activation[neuron_idx]
                        label_parts.append(f"Activation: {val:.4f}")
                
                # Biases are only for layers after input (len(biases) = len(layer_sizes)-1)
                if layer_idx > 0 and (layer_idx - 1) < len(st.session_state.nn.biases):
                    bias_layer_idx = layer_idx - 1
                    bias = st.session_state.nn.biases[bias_layer_idx][0, neuron_idx] if len(st.session_state.nn.biases[bias_layer_idx].shape) > 1 else st.session_state.nn.biases[bias_layer_idx][neuron_idx]
                    label_parts.append(f"Bias: {bias:.4f}")
                
                # Deltas are calculated for each layer after input (len(deltas) = len(layer_sizes)-1)
                if layer_idx > 0 and (layer_idx - 1) < len(st.session_state.nn.gradients.get('deltas', [])):
                    delta_layer_idx = layer_idx - 1
                    deltas = st.session_state.nn.gradients['deltas'][delta_layer_idx]
                    if deltas is not None and len(deltas) > 0:
                        delta = deltas[0, neuron_idx] if len(deltas.shape) > 1 else deltas[neuron_idx]
                        label_parts.append(f"Delta: {delta:.4f}")
                
                node_labels.append("<br>".join(label_parts))
                
                if layer_idx == 0:
                    color = "#4F8EF7"
                elif layer_idx == len(layer_sizes) - 1:
                    color = "#2ECC71"
                else:
                    color = "#8D99A6"
                node_colors.append(color)
        
        edge_x = []
        edge_y = []
        
        for layer_idx in range(len(layer_sizes) - 1):
            n_prev = layer_sizes[layer_idx]
            n_curr = layer_sizes[layer_idx + 1]
            
            for i in range(n_prev):
                for j in range(n_curr):
                    x0 = layer_idx * 2
                    y0 = (max_neurons - 1) / 2 - i
                    x1 = (layer_idx + 1) * 2
                    y1 = (max_neurons - 1) / 2 - j
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='rgba(141, 153, 166, 0.15)'),
            hoverinfo='none',
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(size=16, color=node_colors, line=dict(width=1, color='#232A33')),
            hovertext=node_labels,
            hoverinfo='text',
            name='Neurons'
        ))
        
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='#11161C',
            paper_bgcolor='#11161C',
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Initialize the model to view the network visualization.")

with top_col2:
    st.markdown("## Loss Curve")
    if st.session_state.loss_history:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=list(range(len(st.session_state.loss_history))),
            y=st.session_state.loss_history,
            mode='lines',
            line=dict(color='#4F8EF7', width=2)
        ))
        fig_loss.update_layout(
            plot_bgcolor='#11161C',
            paper_bgcolor='#11161C',
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title='Epoch',
            yaxis_title='Loss',
            height=400,
            xaxis=dict(
                gridcolor='#232A33',
                zerolinecolor='#232A33'
            ),
            yaxis=dict(
                gridcolor='#232A33',
                zerolinecolor='#232A33'
            )
        )
        st.plotly_chart(fig_loss, use_container_width=True)
    else:
        st.info("Start training to view loss curve.")

st.markdown("---")

# Bottom row: 2x2 grid
bottom_col1, bottom_col2 = st.columns(2)

with bottom_col1:
    st.markdown("## Gradient Flow")
    if st.session_state.gradient_norms_history and len(st.session_state.gradient_norms_history) > 0:
        latest_grads = st.session_state.gradient_norms_history[-1]
        layer_names = [f"Layer {i+1}" for i in range(len(latest_grads))]
        fig_grad = go.Figure([go.Bar(
            x=layer_names,
            y=latest_grads,
            marker_color='#4F8EF7'
        )])
        fig_grad.update_layout(
            plot_bgcolor='#11161C',
            paper_bgcolor='#11161C',
            margin=dict(l=20, r=20, t=20, b=20),
            height=300,
            xaxis=dict(
                gridcolor='#232A33',
                zerolinecolor='#232A33'
            ),
            yaxis=dict(
                gridcolor='#232A33',
                zerolinecolor='#232A33'
            )
        )
        st.plotly_chart(fig_grad, use_container_width=True)
    else:
        st.info("Start training to view gradient flow.")

with bottom_col2:
    st.markdown("## Weight Distribution")
    if st.session_state.nn:
        all_weights = []
        for w in st.session_state.nn.weights:
            all_weights.extend(w.flatten())
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=all_weights,
            nbinsx=30,
            marker_color='#4F8EF7'
        ))
        fig_hist.update_layout(
            plot_bgcolor='#11161C',
            paper_bgcolor='#11161C',
            margin=dict(l=20, r=20, t=20, b=20),
            height=300,
            xaxis=dict(
                gridcolor='#232A33',
                zerolinecolor='#232A33'
            ),
            yaxis=dict(
                gridcolor='#232A33',
                zerolinecolor='#232A33'
            )
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Initialize the model to view weight distribution.")

bottom_col3, bottom_col4 = st.columns(2)

with bottom_col3:
    st.markdown("## Decision Boundary")
    if st.session_state.nn and st.session_state.X is not None and st.session_state.X.shape[1] == 2:
        x_min, x_max = st.session_state.X[:, 0].min() - 1, st.session_state.X[:, 0].max() + 1
        y_min, y_max = st.session_state.X[:, 1].min() - 1, st.session_state.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        
        X_grid = np.c_[xx.ravel(), yy.ravel()]
        Z = st.session_state.nn.forward(X_grid)
        
        if st.session_state.loss_function == 'binary_crossentropy':
            Z = (Z > 0.5).astype(int)
        
        Z = Z.reshape(xx.shape)
        
        fig_db = go.Figure()
        fig_db.add_trace(go.Contour(
            x=np.arange(x_min, x_max, 0.1),
            y=np.arange(y_min, y_max, 0.1),
            z=Z,
            colorscale=[[0, '#11161C'], [1, '#4F8EF7']],
            opacity=0.7,
            showscale=False
        ))
        fig_db.add_trace(go.Scatter(
            x=st.session_state.X[:, 0],
            y=st.session_state.X[:, 1],
            mode='markers',
            marker=dict(
                color=st.session_state.y.flatten(),
                colorscale=[[0, '#E74C3C'], [1, '#2ECC71']],
                size=8,
                line=dict(width=1, color='#232A33')
            ),
            name='Data Points'
        ))
        fig_db.update_layout(
            plot_bgcolor='#11161C',
            paper_bgcolor='#11161C',
            margin=dict(l=20, r=20, t=20, b=20),
            height=300,
            showlegend=False,
            xaxis=dict(
                gridcolor='#232A33',
                zerolinecolor='#232A33'
            ),
            yaxis=dict(
                gridcolor='#232A33',
                zerolinecolor='#232A33'
            )
        )
        st.plotly_chart(fig_db, use_container_width=True)
    else:
        st.info("Initialize model with a 2D dataset to view decision boundary.")

with bottom_col4:
    st.markdown("## Gradient Norms History")
    if st.session_state.gradient_norms_history and len(st.session_state.gradient_norms_history) > 0:
        num_layers = len(st.session_state.gradient_norms_history[0])
        fig_grad_hist = go.Figure()
        
        for layer_idx in range(num_layers):
            layer_grads = [epoch[layer_idx] for epoch in st.session_state.gradient_norms_history]
            fig_grad_hist.add_trace(go.Scatter(
                x=list(range(len(layer_grads))),
                y=layer_grads,
                mode='lines',
                name=f'Layer {layer_idx + 1}',
                line=dict(width=2)
            ))
        
        fig_grad_hist.update_layout(
            plot_bgcolor='#11161C',
            paper_bgcolor='#11161C',
            margin=dict(l=20, r=20, t=20, b=20),
            height=300,
            xaxis_title='Epoch',
            yaxis_title='Gradient Norm',
            xaxis=dict(
                gridcolor='#232A33',
                zerolinecolor='#232A33'
            ),
            yaxis=dict(
                gridcolor='#232A33',
                zerolinecolor='#232A33'
            )
        )
        st.plotly_chart(fig_grad_hist, use_container_width=True)
    else:
        st.info("Start training to view gradient norms history.")
