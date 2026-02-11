import jax
import jax.numpy as jnp
from jax import grad, jacfwd

from time import time

grad_tanh = grad(grad(jnp.tanh))

start_time = time()
print(grad_tanh(2.0))
print(grad_tanh(3.0))
end_time = time()

print(f"time: {end_time - start_time}")
