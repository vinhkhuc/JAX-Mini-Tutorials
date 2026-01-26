import jax.numpy as jnp
from jax import jit

@jit
def multiply(x, y):
    return x * y

a = jnp.array([2, 3, 4], dtype=jnp.float32)
b = jnp.array([3, 4, 5], dtype=jnp.float32)

m = multiply(a, b)  # element-wise product
print(m)
