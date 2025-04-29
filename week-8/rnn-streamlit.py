import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim

# Streamlit config
st.set_page_config(page_title="PyTorch RNN Tutorial", layout="wide")
st.title("Recurrent Neural Networks (RNN) Tutorial with PyTorch")

# Sidebar parameters
st.sidebar.header("RNN Parameters")
dataset_type = st.sidebar.selectbox(
    "Select Dataset", ["Sine Wave", "Square Wave", "Sawtooth Wave"]
)
hidden_size = st.sidebar.slider("Hidden Layer Size", 5, 100, 20)
epochs = st.sidebar.slider("Number of Epochs", 10, 200, 10)
learning_rate = st.sidebar.number_input(
    "Learning Rate", 0.0001, 0.1, 0.01, format="%.4f"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data generation
time_steps = np.linspace(0, 100, 100)
if dataset_type == "Sine Wave":
    data_np = np.sin(time_steps)
elif dataset_type == "Square Wave":
    data_np = np.sign(np.sin(time_steps))
else:
    data_np = (time_steps % 10) / 5 - 1

noise = np.random.normal(0, 0.1, data_np.shape)
data_np += noise
# Prepare sequences and targets
data = (
    torch.tensor(data_np, dtype=torch.float32).unsqueeze(1).to(device)
)  # shape [T, 1]
targets = torch.roll(data, shifts=-1, dims=0)


# PyTorch RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(
            input_size, hidden_size, batch_first=False, nonlinearity="tanh"
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        out, h_next = self.rnn(x, h)
        pred = self.fc(out)
        return pred, h_next


# Initialize model, loss, optimizer
model = RNNModel(input_size=1, hidden_size=hidden_size, output_size=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training section
st.header("Training Process")
if st.button("Train RNN"):
    progress_bar = st.progress(0)
    start_time = time.time()
    losses = []
    # Initialize hidden state
    h = torch.zeros(1, 1, hidden_size).to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        # Forward pass
        inputs_seq = data.unsqueeze(1)  # [T, batch=1, input]
        outputs, h = model(inputs_seq, h.detach())
        loss = criterion(outputs.squeeze(), targets.squeeze())
        # Backward pass
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        progress_bar.progress(epoch / epochs)

    training_time = time.time() - start_time
    st.success(f"Training completed in {training_time:.2f} seconds!")

    # Plot training loss
    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(losses)
    ax_loss.set_title("Training Loss Over Time")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    st.pyplot(fig_loss)

    # Predictions
    model.eval()
    h_pred = torch.zeros(1, 1, hidden_size).to(device)
    preds = []
    with torch.no_grad():
        for t in range(len(data)):
            x_t = data[t].unsqueeze(0).unsqueeze(1)  # [1, 1, 1]
            out_t, h_pred = model(x_t, h_pred)
            preds.append(out_t.item())

    # Plot predictions vs actual
    data_cpu = data.cpu().numpy().flatten()
    preds_np = np.array(preds)
    fig_pred, ax_pred = plt.subplots(figsize=(12, 6))
    ax_pred.plot(data_cpu, label="Actual Data")
    ax_pred.plot(preds_np, label="RNN Predictions", linestyle="--")
    ax_pred.set_title(f"RNN Predictions vs Actual {dataset_type} Data")
    ax_pred.set_xlabel("Time Steps")
    ax_pred.set_ylabel("Value")
    ax_pred.legend()
    st.pyplot(fig_pred)

# Implementation details
st.markdown(
    """
### Implementation Details

- **Model**: Built with `nn.RNN` layer + `nn.Linear` output head.
- **Loss**: Mean Squared Error (MSE).
- **Optimizer**: Adam.
- **Hidden State**: Passed between time steps with `detach()` to truncate gradients.
- **Training**: Full sequence processed each epoch.

### Tips for Better Results:

- Increase `hidden_size` for more capacity.
- Adjust `learning_rate` for training stability.
- More `epochs` for complex patterns.
- Use gradient clipping if gradients explode.
"""
)
