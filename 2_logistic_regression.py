import math
import jax
import jax.numpy as jnp
from jax import value_and_grad
from jax import nn
import optax

from data_util import load_mnist


def model_logits(w, X):
    return X @ w


def loss_fn(w, X, y):
    logits = model_logits(w, X)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


@jax.jit
def train(w, X, y, lr):
    loss, grads = value_and_grad(loss_fn)(w, X, y)
    w -= lr * grads
    return w, loss


@jax.jit
def predict(w, X):
    logits = model_logits(w, X)
    probs = nn.softmax(logits)
    return jnp.argmax(probs, axis=-1)


def main():
    n_classes = 10
    bz = 16
    lr = 0.01
    epochs = 50
    rand_key = jax.random.PRNGKey(42)

    trX, teX, trY, teY = load_mnist(onehot=False)
    trX = jnp.asarray(trX, dtype=jnp.float32)
    teX = jnp.asarray(teX, dtype=jnp.float32)
    trY = jnp.asarray(trY, dtype=jnp.int32)

    n_examples, n_features = trX.shape

    glorot_normal = nn.initializers.glorot_normal()
    w = glorot_normal(rand_key, shape=(n_features, n_classes))

    num_batches = math.ceil(n_examples / bz)

    for epoch in range(epochs):
        total_loss = 0
        for k in range(num_batches):
            start, end = k * bz, min((k + 1) * bz, n_examples)
            w, loss = train(w, trX[start:end], trY[start:end], lr)
            total_loss += float(loss)

        avg_loss = total_loss / num_batches
        predY = predict(w, teX)
        acc = jnp.mean(predY == teY)
        print(f"Epoch {epoch+1}, cost ≈ {avg_loss:.6f}, acc ≈ {100 * acc:.2f}%")


if __name__ == "__main__":
    main()
