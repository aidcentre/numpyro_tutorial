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
import arviz as az
import arviz_plots as azp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

# %% [markdown]
# ## Generate fake data

# %%
key = jax.random.PRNGKey(42)
n = 50

true_intercept = 2.0
true_slope = 3.0
true_sigma = 3.0

key, sub_key = jax.random.split(key)
x = jax.random.normal(sub_key, shape=(n,))
key, sub_key = jax.random.split(key)
y = (
    true_slope * x
    + true_intercept
    + true_sigma * jax.random.normal(sub_key, shape=(n,))
)

plt.figure()
plt.scatter(x, y, label="Data", color="k")
plt.legend()
plt.show()


# %% [markdown]
# ## Define model


# %%
def model(x: jax.Array, y: jax.Array | None = None) -> None:
    intercept = numpyro.sample("intercept", dist.Normal(0, 10))

    slope = numpyro.sample("slope", dist.Normal(0, 10))

    sigma = numpyro.sample("sigma", dist.HalfNormal(5))

    mu = intercept + slope * x

    with numpyro.plate("data", len(x)):
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)


# %%
numpyro.render_model(model, model_kwargs={"x": x}, render_distributions=True)

# %% [markdown]
# ## Doing inference

# %%
kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup=500, num_samples=10000)
key, sub_key = jax.random.split(key)
mcmc.run(sub_key, x=x, y=y)
mcmc.print_summary()

# %%
posterior_samples = mcmc.get_samples()
idata = az.from_numpyro(mcmc)

# %%
plt.figure()
az.plot_dist(idata, var_names=["intercept", "slope", "sigma"])
azp.plot_pair(idata, var_names=["intercept", "slope", "sigma"])
plt.gcf().set_size_inches(8, 8)
plt.show()

# %%
x_plot = jnp.linspace(x.min(), x.max(), 100)
y_true = true_intercept + true_slope * x_plot
n_lines = 1000

key, subkey = jax.random.split(key)
idx = jax.random.choice(
    subkey, len(posterior_samples["slope"]), shape=(n_lines,), replace=False
)

plt.figure()
for i in idx:
    intercept_i = posterior_samples["intercept"][i]
    slope_i = posterior_samples["slope"][i]
    y_line = intercept_i + slope_i * x_plot
    plt.plot(x_plot, y_line, alpha=0.01, color="red")
plt.plot(x_plot, y_true, linewidth=2, label="True regression line", color="black")
plt.legend()
plt.show()

# %% [markdown]
# ## Sample new points

# %%
key, sub_key = jax.random.split(key)
posterior_predictive = Predictive(model, posterior_samples=posterior_samples)
predictive_samples = posterior_predictive(sub_key, x=x)
y_new = predictive_samples["obs"][0, :]

# %%
plt.figure(figsize=(7, 5))

plt.scatter(x, y, color="black", label="Observed data")
plt.scatter(x, y_new, color="red", alpha=0.5, label="Posterior predictive sample")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# %%
