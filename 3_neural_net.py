from functools import partial
import math
import jax
import jax.numpy as jnp
from jax import value_and_grad
from jax.nn import softmax
import flax.linen as nn
import optax

from data_util import load_mnist


class MLP(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, use_bias=False)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim, use_bias=False)(x)
        return x


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
    bz = 16
    lr = 0.01
    epochs = 50
    rng = jax.random.PRNGKey(42)

    trX, teX, trY, teY = load_mnist(onehot=False)
    trX = jnp.asarray(trX, dtype=jnp.float32)
    teX = jnp.asarray(teX, dtype=jnp.float32)
    trY = jnp.asarray(trY, dtype=jnp.int32)

    n_examples, n_features = trX.shape

    model = MLP(hidden_dim=512, output_dim=n_classes)
    params = model.init(rng, jnp.ones((1, n_features)))["params"]

    num_batches = math.ceil(n_examples / bz)

    optimizer = optax.sgd(lr)
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
