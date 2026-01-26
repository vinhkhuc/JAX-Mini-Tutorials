import math
import jax
import jax.numpy as jnp
from jax import value_and_grad


def model(w, X):
    return w * X


def loss_fn(w, X, y):
    y_pred = model(w, X)
    return jnp.mean((y_pred - y) ** 2)


@jax.jit
def train(w, X, y, lr):
    loss, grads = value_and_grad(loss_fn)(w, X, y)
    w -= lr * grads
    return w, loss


@jax.jit
def predict(w, X):
    return model(w, X)


def main():
    N = 100
    bz = 16
    lr = 0.01
    epochs = 50
    rand_key = jax.random.PRNGKey(42)

    X = jnp.linspace(-1, 1, N)

    rand_key, using_key = jax.random.split(rand_key)
    e = 0.33 * jax.random.normal(using_key, shape=(N,))
    y = 2 * X + e

    rand_key, using_key = jax.random.split(rand_key)
    w = jax.random.normal(using_key, shape=())  # scalar

    num_batches = math.ceil(N / bz)

    for epoch in range(epochs):
        total_loss = 0
        for k in range(num_batches):
            start, end = k * bz, min((k + 1) * bz, N)
            w, loss = train(w, X[start:end], y[start:end], lr)
            total_loss += float(loss)

        avg_loss = total_loss / num_batches
        predY = predict(w, X)
        mse = jnp.mean((predY - y) ** 2)
        print(
            f"Epoch {epoch+1: 3d}, cost ≈ {avg_loss:.6f}, mse ≈ {mse:.6f}, w ≈ {w:.4f}"
        )


if __name__ == "__main__":
    main()
