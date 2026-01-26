from functools import partial
import math
import jax
import jax.numpy as jnp
from jax import value_and_grad
from jax.nn import softmax
import flax.linen as nn
import optax

from data_util import load_mnist


class LSTMNet(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        # x has shape: [batch, time, input_dim]

        # 1. Define the LSTM layer using nn.scan over an LSTMCell.
        lstm = nn.scan(
            nn.LSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )(features=self.hidden_dim)

        # 2. Initialize the LSTM carry state (hidden state and cell state).
        batch_size = x.shape[0]
        carry = lstm.initialize_carry(
            jax.random.PRNGKey(42), (batch_size, self.hidden_dim)
        )

        # 3. Run the LSTM.
        # out_sequence shape: [batch, time, hidden_dim]
        _, out_sequence = lstm(carry, x)

        # 4. Select the output from the *last* time step for classification.
        h_final = out_sequence[:, -1, :]

        # 5. Add a Dense layer to map the final hidden state to output logits.
        logits = nn.Dense(features=self.output_dim)(h_final)

        return logits


def model_logits(params, model, X):
    return model.apply({"params": params}, X)


def loss_fn(params, model, X, y):
    logits = model_logits(params, model, X)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


@partial(jax.jit, static_argnames=("model", "optimizer"))
def train_step(params, opt_state, model, optimizer, X, y):
    loss, grads = value_and_grad(loss_fn)(params, model, X, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


@partial(jax.jit, static_argnames="model")
def predict(params, model, X):
    logits = model_logits(params, model, X)
    probs = softmax(logits)
    return jnp.argmax(probs, axis=-1)


def main():
    n_classes = 10
    seq_length = 28
    input_dim = 28
    hidden_dim = 128
    bz = 16
    lr = 1e-4
    epochs = 50
    rand_key = jax.random.PRNGKey(42)

    trX, teX, trY, teY = load_mnist(onehot=False)
    n_examples = trX.shape[0]

    # Reshape images to be sequences of rows
    # Shape becomes: [num_samples, seq_length, input_dim]
    trX = trX.reshape(-1, seq_length, input_dim)
    teX = teX.reshape(-1, seq_length, input_dim)

    trX = jnp.asarray(trX, dtype=jnp.float32)
    teX = jnp.asarray(teX, dtype=jnp.float32)
    trY = jnp.asarray(trY, dtype=jnp.int32)

    _, init_key = jax.random.split(rand_key)
    model = LSTMNet(hidden_dim=hidden_dim, output_dim=n_classes)

    dummy_input = jnp.ones((bz, seq_length, input_dim), dtype=jnp.float32)
    params = model.init(init_key, dummy_input)["params"]

    num_batches = math.ceil(n_examples / bz)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    for epoch in range(epochs):
        total_loss = 0
        for k in range(num_batches):
            start, end = k * bz, min((k + 1) * bz, n_examples)
            params, opt_state, loss = train_step(
                params, opt_state, model, optimizer, trX[start:end], trY[start:end]
            )
            total_loss += float(loss)

        avg_loss = total_loss / num_batches
        predY = predict(params, model, teX)
        acc = jnp.mean(predY == teY)
        print(f"Epoch {epoch+1}, cost ≈ {avg_loss:.6f}, acc ≈ {100 * acc:.2f}%")


if __name__ == "__main__":
    main()
