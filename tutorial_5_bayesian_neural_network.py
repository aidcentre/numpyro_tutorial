# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: numpyro
#     language: python
#     name: python3
# ---

# %%
from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import vmap
from numpyro import handlers
from numpyro.infer import MCMC, NUTS

if TYPE_CHECKING:
    from collections.abc import Callable

# %% [markdown]
# ## Generate fake data


# %%
def get_data(
    N: int = 50,
    D_X: int = 3,
    sigma_obs: float = 0.05,
    N_test: int = 500,
    seed: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    rng = np.random.default_rng(seed)

    X = jnp.linspace(-1, 1, N)
    X = jnp.power(X[:, np.newaxis], jnp.arange(D_X))

    W = 0.5 * rng.standard_normal(D_X)

    Y = jnp.dot(X, W) + 0.5 * jnp.power(0.5 + X[:, 1], 2.0) * jnp.sin(4.0 * X[:, 1])
    Y += sigma_obs * rng.standard_normal(N)
    Y = Y[:, np.newaxis]
    Y -= jnp.mean(Y)
    Y /= jnp.std(Y)

    X_test = jnp.linspace(-1.3, 1.3, N_test)
    X_test = jnp.power(X_test[:, np.newaxis], jnp.arange(D_X))

    return X, Y, X_test


# %%
N = 100
D_X = 3
D_H = 5

X, Y, X_test = get_data(
    N=N,
    D_X=D_X,
)

# %%
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 1], Y[:, 0], color="black")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# %% [markdown]
# ## Define model


# %%
def nonlin(x: jax.Array) -> jax.Array:
    return jnp.tanh(x)


def model(
    X: jax.Array,
    Y: jax.Array | None = None,
    D_H: int = 5,
    D_Y: int = 1,
) -> None:
    N, D_X = X.shape

    w1 = numpyro.sample(
        "w1",
        dist.Normal(
            jnp.zeros((D_X, D_H)),
            jnp.ones((D_X, D_H)),
        ),
    )

    z1 = nonlin(jnp.matmul(X, w1))

    w2 = numpyro.sample(
        "w2",
        dist.Normal(
            jnp.zeros((D_H, D_H)),
            jnp.ones((D_H, D_H)),
        ),
    )

    z2 = nonlin(jnp.matmul(z1, w2))

    w3 = numpyro.sample(
        "w3",
        dist.Normal(
            jnp.zeros((D_H, D_Y)),
            jnp.ones((D_H, D_Y)),
        ),
    )

    z3 = jnp.matmul(z2, w3)

    prec_obs = numpyro.sample(
        "prec_obs",
        dist.Gamma(3.0, 1.0),
    )

    sigma_obs = 1.0 / jnp.sqrt(prec_obs)

    with numpyro.plate("data", N):
        numpyro.sample(
            "Y",
            dist.Normal(z3, sigma_obs).to_event(1),
            obs=Y,
        )


# %%
numpyro.render_model(
    model,
    model_kwargs={
        "X": X,
        "Y": Y,
        "D_H": D_H,
    },
    render_distributions=True,
)

# %% [markdown]
# ## Doing inference

# %%
kernel = NUTS(model)

mcmc = MCMC(
    kernel,
    num_warmup=1000,
    num_samples=2000,
    num_chains=1,
)

key = jax.random.PRNGKey(0)
key, sub_key = jax.random.split(key)

mcmc.run(
    sub_key,
    X=X,
    Y=Y,
    D_H=D_H,
)

mcmc.print_summary()

# %%
posterior_samples = mcmc.get_samples()


# %% [markdown]
# ## Posterior predictive


# %%
def predict(
    model: Callable,
    rng_key: jax.Array,
    samples: dict[str, jax.Array],
    X: jax.Array,
    D_H: int,
) -> jax.Array:
    model = handlers.substitute(
        handlers.seed(model, rng_key),
        samples,
    )

    model_trace = handlers.trace(model).get_trace(
        X=X,
        Y=None,
        D_H=D_H,
    )

    return model_trace["Y"]["value"]


# %%
key, sub_key = jax.random.split(key)

vmap_args = (
    posterior_samples,
    jax.random.split(
        sub_key,
        len(posterior_samples["prec_obs"]),
    ),
)

predictions = vmap(
    lambda samples, rng_key: predict(
        model,
        rng_key,
        samples,
        X_test,
        D_H,
    )
)(*vmap_args)

predictions = predictions[..., 0]

# %%
mean_prediction = jnp.mean(
    predictions,
    axis=0,
)

percentiles = np.percentile(
    predictions,
    [5.0, 95.0],
    axis=0,
)

# %%
plt.figure(figsize=(8, 6))

plt.scatter(
    X[:, 1],
    Y[:, 0],
    color="black",
    marker="x",
)

plt.fill_between(
    X_test[:, 1],
    percentiles[0],
    percentiles[1],
    alpha=0.3,
)

plt.plot(
    X_test[:, 1],
    mean_prediction,
    linewidth=2,
)

plt.xlabel("X")
plt.ylabel("Y")
plt.show()
