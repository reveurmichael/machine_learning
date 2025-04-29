import numpy as np
import matplotlib.pyplot as plt


class RNN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(
        self, inputs: np.ndarray, h_prev: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        h_next = np.tanh(np.dot(self.Wxh, inputs) + np.dot(self.Whh, h_prev) + self.bh)
        output = np.dot(self.Why, h_next) + self.by
        return h_next, output

    def backward(
        self,
        inputs: np.ndarray,
        h_prev: np.ndarray,
        h_next: np.ndarray,
        output: np.ndarray,
        target: np.ndarray,
        learning_rate: float,
    ) -> None:
        output_error = output - target
        dWhy = np.dot(output_error, h_next.T)
        dby = output_error
        dh_next = np.dot(self.Why.T, output_error) * (1 - h_next**2)
        dWxh = np.dot(dh_next, inputs.T)
        dWhh = np.dot(dh_next, h_prev.T)
        dbh = dh_next

        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby

    def train(
        self, data: np.ndarray, targets: np.ndarray, epochs: int, learning_rate: float
    ) -> None:
        for epoch in range(epochs):
            h_prev = np.zeros((self.hidden_size, 1))
            loss = 0

            for t in range(len(data)):
                inputs = data[t].reshape(-1, 1)
                target = targets[t].reshape(-1, 1)

                h_next, output = self.forward(inputs, h_prev)
                loss += np.sum((output - target) ** 2)
                self.backward(inputs, h_prev, h_next, output, target, learning_rate)
                h_prev = h_next

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")


# Example usage
if __name__ == "__main__":
    # Generate sine wave data
    time_steps = np.linspace(0, 100, 100)  # 100 time steps
    data = np.sin(time_steps)  # Sine wave data
    noise = np.random.normal(0, 0.1, data.shape)  # Add Gaussian noise
    data += noise  # Combine sine wave with noise
    data = data.reshape(-1, 1)  # Reshape to (100, 1)
    targets = np.roll(data, -1)  # Shifted target for next time step prediction

    # Initialize RNN
    rnn = RNN(input_size=1, hidden_size=10, output_size=1)

    # Train the RNN
    rnn.train(data, targets, epochs=1000, learning_rate=0.01)

    # Test the RNN
    h_prev = np.zeros((rnn.hidden_size, 1))
    predictions = []

    for i in range(100):  # Test on all data points
        input_data = data[i].reshape(-1, 1)
        h_prev, output = rnn.forward(input_data, h_prev)
        predictions.append(output.flatten())

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(data, label="Actual Data (Sine Wave)", color="blue")
    plt.plot(predictions, label="RNN Predictions", color="red", linestyle="--")
    plt.title("RNN Predictions vs Actual Sine Wave Data")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
