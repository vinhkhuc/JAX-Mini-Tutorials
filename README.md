[![Build Status](https://travis-ci.org/vinhkhuc/JAX-Mini-Tutorials.svg?branch=master)](https://travis-ci.org/vinhkhuc/JAX-Mini-Tutorials)

Minimal tutorials for [JAX](https://github.com/jax-ml/jax) adapted
from Alec Radford's [Theano tutorials](https://github.com/Newmu/Theano-Tutorials).

```bash
python3 -m pip install -U jax jaxlib flax optax
```

For acceleration, replace `jax` with:
* `"jax[tpu]"` on TPU.
* `"jax[cuda12]"` on NVIDIA with CUDA 12.

<ol start="0">
  <li><a href="0_multiply.py">Tensor multiplication</a></li>
  <li><a href="1_linear_regression.py">Linear Regression</a></li>
  <li><a href="2_logistic_regression.py">Logistic Regression</a></li>
  <li><a href="3_neural_net.py">Neural Network</a></li>
  <li><a href="4_modern_neural_net.py">Modern Neural Network</a></li>
  <li><a href="5_convolutional_net.py">Convolutional Neural Network</a></li>
  <li><a href="6_lstm.py">Long Short-Term Memory</a></li>
</ol>
