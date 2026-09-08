from functools import partial
import math
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

from data_util import load_mnist, save_image_grid


def ddpm_schedules(beta1, beta2, T):
    if not (0 < beta1 < beta2 < 1 and T >= 2):
        raise ValueError("Require 0 < beta1 < beta2 < 1 and T >= 2")

    # Index zero is clean data: beta[0] = 0, alpha_bar[0] = 1
    beta = jnp.concatenate([jnp.zeros(1), jnp.linspace(beta1, beta2, T)])
    alpha = 1.0 - beta
    ab = jnp.cumprod(alpha)
    prev_ab = jnp.concatenate([jnp.ones(1), ab[:-1]])
    denom = jnp.maximum(1.0 - ab, 1e-20)

    return dict(
        beta_t=beta,
        alpha_t=alpha,
        alphabar_t=ab,
        sqrtab=jnp.sqrt(ab),
        sqrtmab=jnp.sqrt(1.0 - ab),
        sqrt_beta_t=jnp.sqrt(beta),
        oneover_sqrta=jax.lax.rsqrt(alpha),
        mab_over_sqrtmab=beta / jnp.sqrt(denom),
        posterior_variance=beta * (1.0 - prev_ab) / denom,
        posterior_coef1=beta * jnp.sqrt(prev_ab) / denom,
        posterior_coef2=(1.0 - prev_ab) * jnp.sqrt(alpha) / denom,
    )


class Block(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x, temb):
        residual = x
        h = nn.Conv(self.out_channels, (3, 3), padding="SAME")(x)
        h = nn.silu(nn.GroupNorm(num_groups=8)(h))
        h = h + nn.Dense(self.out_channels)(nn.silu(temb))[:, None, None, :]
        h = nn.Conv(self.out_channels, (3, 3), padding="SAME")(h)
        h = nn.silu(nn.GroupNorm(num_groups=8)(h))

        if residual.shape[-1] != self.out_channels:
            residual = nn.Conv(self.out_channels, (1, 1))(residual)

        return (h + residual) / math.sqrt(2)


class EpsModel(nn.Module):
    n_channel: int = 1
    base: int = 32

    @nn.compact
    def __call__(self, x, t):
        # Time embedding
        t = jnp.broadcast_to(jnp.asarray(t, jnp.float32).reshape(-1), (x.shape[0],))
        frequencies = jnp.exp(-math.log(10000) * jnp.arange(32) / 31)
        angles = (t * 1000)[:, None] * frequencies[None, :]
        temb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
        temb = nn.Dense(128)(nn.silu(nn.Dense(128)(temb)))

        # 28 -> 14 -> 7, then resize exactly to each skip's shape
        skip1 = Block(self.base)(x, temb)
        h = nn.Conv(self.base * 2, (3, 3), strides=(2, 2), padding="SAME")(skip1)
        skip2 = Block(self.base * 2)(h, temb)
        h = nn.Conv(self.base * 4, (3, 3), strides=(2, 2), padding="SAME")(skip2)
        h = Block(self.base * 4)(h, temb)
        h = Block(self.base * 4)(h, temb)

        for skip, channels in [(skip2, self.base * 2), (skip1, self.base)]:
            h = jax.image.resize(
                h,
                (h.shape[0], skip.shape[1], skip.shape[2], h.shape[-1]),
                method="nearest",
            )
            h = nn.Conv(channels, (3, 3), padding="SAME")(h)
            h = Block(channels)(jnp.concatenate([h, skip], axis=-1), temb)

        return nn.Conv(self.n_channel, (3, 3), padding="SAME")(h)


class TrainState(train_state.TrainState):
    ema_params: object


def loss_fn(params, state, x, schedules, T, rng):
    rng_t, rng_eps = jax.random.split(rng)
    ts = jax.random.randint(rng_t, (x.shape[0],), 1, T + 1)
    eps = jax.random.normal(rng_eps, x.shape)
    xt = (
        schedules["sqrtab"][ts, None, None, None] * x
        + schedules["sqrtmab"][ts, None, None, None] * eps
    )
    pred = state.apply_fn({"params": params}, xt, ts / T)
    return jnp.mean(jnp.square(eps - pred))


@partial(jax.jit, static_argnames=["T"])
def train_step(state, x, schedules, T, rng):
    loss, grads = jax.value_and_grad(loss_fn)(state.params, state, x, schedules, T, rng)
    state = state.apply_gradients(grads=grads)

    # Warm up EMA so it does not lag far behind during the first epoch
    decay = jnp.minimum(0.999, (1.0 + state.step) / (10.0 + state.step))
    ema = jax.tree_util.tree_map(
        lambda e, p: decay * e + (1 - decay) * p, state.ema_params, state.params
    )

    return state.replace(ema_params=ema), loss


@partial(jax.jit, static_argnames=["n_sample", "size", "T"])
def sample(state, schedules, n_sample, size, T, rng):
    rng_x, rng_loop = jax.random.split(rng)
    x = jax.random.normal(rng_x, (n_sample, *size))

    def loop_step(step, carry):
        xt, key = carry
        key, noise_key = jax.random.split(key)
        t = T - step
        times = jnp.full((n_sample,), t / T, dtype=jnp.float32)
        eps = state.apply_fn({"params": state.ema_params}, xt, times)

        # Predict clean data and clip to its known range.
        # Use the matching posterior mean (not the epsilon mean after independently clipping)
        x0 = (xt - schedules["sqrtmab"][t] * eps) / schedules["sqrtab"][t]
        x0 = jnp.clip(x0, -1, 1)
        mean = (
            schedules["posterior_coef1"][t] * x0 + schedules["posterior_coef2"][t] * xt
        )
        noise = jax.random.normal(noise_key, xt.shape)
        sigma = jnp.sqrt(jnp.maximum(schedules["posterior_variance"][t], 0))

        return mean + jnp.where(t > 1, sigma, 0) * noise, key

    return jax.lax.fori_loop(0, T, loop_step, (x, rng_loop))[0]


def main():
    epochs = 100
    T = 1000
    bz = 128
    sample_every = 5

    schedules = ddpm_schedules(1e-4, 0.02, T)
    trX, *_ = load_mnist()  # (N, 28, 28), float32 in [0, 1]
    trX = trX.reshape(-1, 28, 28, 1) * 2 - 1  # (N, 28, 28, 1), float32 in [-1, 1]

    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    model = EpsModel()
    params = model.init(init_rng, jnp.zeros((1, 28, 28, 1)), jnp.ones((1,)))["params"]
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        ema_params=params,
        tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(2e-4)),
    )
    shuffle_rng = np.random.default_rng(42)

    # Use a fixed sampling key to make the progress easier to compare
    sample_rng = jax.random.PRNGKey(123)
    for epoch in range(epochs):
        indices = shuffle_rng.permutation(len(trX))
        loss_sum = 0
        seen = 0

        for step, start in enumerate(range(0, len(trX), bz)):
            x = trX[indices[start : start + bz]]
            rng, step_rng = jax.random.split(rng)
            state, loss = train_step(state, x, schedules, T, step_rng)
            loss = float(loss)
            loss_sum = loss_sum + loss * len(x)
            seen += len(x)

            if step % 100 == 0:
                print(f"Epoch {epoch:03d} | Step {step:03d} | Loss {loss:.4f}")

        print(f"Epoch {epoch:03d} | Mean loss {float(loss_sum) / seen:.4f}")
        if epoch % sample_every == 0 or epoch == epochs:
            x0 = sample(state, schedules, 16, (28, 28, 1), T, sample_rng)
            save_image_grid(x0, f"output/sample_{epoch:03d}.png", nrow=4)


if __name__ == "__main__":
    main()
