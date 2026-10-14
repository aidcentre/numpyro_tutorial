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
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

# %% [markdown]
# ## Define and show model


# %%
def one_throw_model(data: jax.Array | None = None) -> None:
    p = numpyro.sample("p", dist.Beta(1.0, 1.0))

    numpyro.sample("obs", dist.Bernoulli(p), obs=data)


# %%
numpyro.render_model(one_throw_model, render_distributions=True)

# %% [markdown]
# ## Sample from my model without data

# %%
key = jax.random.PRNGKey(0)

predictive = Predictive(one_throw_model, num_samples=10)
samples = predictive(key)

print(samples)


# %% [markdown]
# ## Several iid coin throws


# %%
def n_throws_model(data: jax.Array | None = None, n: int = 10) -> None:
    p = numpyro.sample("p", dist.Beta(1.0, 1.0))

    with numpyro.plate("throws", n):
        numpyro.sample("obs", dist.Bernoulli(p), obs=data)


# %%
numpyro.render_model(n_throws_model, render_distributions=True)

# %%
key = jax.random.PRNGKey(0)

predictive = Predictive(n_throws_model, num_samples=5)
samples = predictive(key, n=10)
print(samples)

# %% [markdown]
# ## Doing inference from data

# %%
data = jnp.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])

# %%
kernel = NUTS(n_throws_model)
mcmc = MCMC(kernel, num_warmup=500, num_samples=10000)
mcmc.run(jax.random.PRNGKey(0), data=data, n=len(data))
mcmc.print_summary()

# %%
posterior_samples = mcmc.get_samples()


# %%

plt.figure()
plt.hist(posterior_samples["p"], bins=40, density=True, alpha=0.7)
plt.xlabel("p")
plt.ylabel("density")
plt.show()

# %% [markdown]
# ## Sample from the posterior predictive distribution

# %%
posterior_predictive = Predictive(n_throws_model, posterior_samples=posterior_samples)
new_samples = posterior_predictive(jax.random.PRNGKey(1), n=10)

# %%

# %%
