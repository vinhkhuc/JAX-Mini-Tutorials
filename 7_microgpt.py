# Adapted from: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
from functools import partial

import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

from data_util import load_names

n_embd = 16  # embedding dimension
n_head = 4  # number of attention heads
n_layer = 1  # number of layers
block_size = 16  # maximum sequence length
head_dim = n_embd // n_head  # dimension of each head

num_steps = 1000  # training steps
lr = 0.01

temperature = 0.5
num_samples = 20  # number of output samples

kernel_init = nn.initializers.normal(stddev=0.08)
Dense = partial(nn.Dense, use_bias=False, kernel_init=kernel_init)
Embed = partial(nn.Embed, embedding_init=kernel_init)
RMSNorm = partial(nn.RMSNorm, use_scale=False, epsilon=1e-5)


class Block(nn.Module):
    @nn.compact
    def __call__(self, x, mask):
        # 1) Multi-head attention block
        x_residual = x
        x = RMSNorm()(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=n_head,
            broadcast_dropout=False,
            use_bias=False,
            kernel_init=kernel_init,
        )(x, x, mask=mask)
        x = x + x_residual

        # 2) MLP block
        x_residual = x
        x = RMSNorm()(x)
        x = Dense(4 * n_embd)(x)
        x = nn.relu(x)
        x = Dense(n_embd)(x)
        x = x + x_residual
        return x


class GPT(nn.Module):
    vocab_size: int

    @nn.compact
    def __call__(self, tokens):
        T = tokens.shape[1]
        wte = Embed(num_embeddings=self.vocab_size, features=n_embd)
        wpe = Embed(num_embeddings=block_size, features=n_embd)

        pos_ids = jnp.arange(T)[None, :]
        x = wte(tokens) + wpe(pos_ids)
        x = RMSNorm()(x)

        mask = nn.make_causal_mask(tokens)
        for _ in range(n_layer):
            x = Block()(x, mask=mask)

        logits = Dense(self.vocab_size)(x)
        return logits


@jax.jit
def train_step(state, x, y, mask):

    def loss_fn(params, x, y, mask):
        logits = state.apply_fn({"params": params}, x)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y)
        loss = (loss * mask).sum() / mask.sum()
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params, x, y, mask)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def predict(state, x):
    logits = state.apply_fn({"params": state.params}, x)
    return logits[0]


def main():
    docs = load_names()
    uchars = sorted(set("".join(docs)))
    BOS, vocab_size = len(uchars), len(uchars) + 1
    print(f"num docs: {len(docs)}")
    print(f"vocab size: {vocab_size}")

    model = GPT(vocab_size=vocab_size)
    rng = jax.random.PRNGKey(42)
    variables = model.init(rng, jnp.ones((1, block_size), dtype=jnp.int32))
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(variables))
    print(f"num params: {num_params}")

    lr_schedule = optax.linear_schedule(
        init_value=lr, end_value=0.0, transition_steps=num_steps
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optax.adam(learning_rate=lr_schedule, b1=0.85, b2=0.99, eps=1e-8),
    )

    # Train
    for step in range(num_steps):
        # Take single document, tokenize it, surround it with BOS special token
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        # Prepare sequences padded to block_size
        x = jnp.array(tokens[:n] + [BOS] * (block_size - n))[None, :]
        y = jnp.array(tokens[1 : n + 1] + [BOS] * (block_size - n))[None, :]
        mask = jnp.array([1.0] * n + [0.0] * (block_size - n))[None, :]

        state, loss = train_step(state, x, y, mask)
        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss:.4f}")

    # Generate
    rng, sample_rng = jax.random.split(rng)
    print("\n--- generate (new, hallucinated names) ---")
    for sample_idx in range(num_samples):
        tokens = [BOS] + [0] * (block_size - 1)
        sample = []

        for pos_id in range(block_size):
            sample_rng, rand_key = jax.random.split(sample_rng)
            x = jnp.array(tokens)[None, :]
            logits_seq = predict(state, x)
            logits_step = logits_seq[pos_id] / temperature
            probs = jax.nn.softmax(logits_step)
            next_token = jax.random.choice(rand_key, vocab_size, p=probs).item()
            if next_token == BOS:
                break

            sample.append(uchars[next_token])
            if pos_id + 1 < block_size:
                tokens[pos_id + 1] = next_token

        print(f'sample {sample_idx+1:2d}: {"".join(sample)}')


if __name__ == "__main__":
    main()
