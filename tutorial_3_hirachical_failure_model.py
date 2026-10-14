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
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# %% [markdown]
# ## Generate data


# %%
def generate_data(  # noqa: PLR0913
    *,
    n_tanks: int = 50,
    sigma: float = 2.0,
    alpha: float = 0.8,
    tau: float = 0.5,
    c: float = 3.0,
    gamma: float = 0.4,
    delta: float = 0.5,
    seed: int = 42,
) -> dict[str, np.ndarray]:

    rng = np.random.default_rng(seed)

    V = rng.uniform(50, 200, size=n_tanks)

    V0 = 100.0
    v = V / V0

    b = rng.normal(0, tau, size=n_tanks)
    beta = np.exp(b)

    lambda_ = sigma * v**alpha * beta

    counts = rng.poisson(lambda_)

    mean_leak = c * v**gamma * beta**delta

    leak_sizes = []
    leak_tank_id = []

    for i in range(n_tanks):
        sizes = rng.exponential(
            scale=mean_leak[i],
            size=counts[i],
        )

        leak_sizes.extend(sizes)
        leak_tank_id.extend([i] * counts[i])

    return {
        "V": V,
        "counts": counts,
        "leak_sizes": np.asarray(leak_sizes),
        "leak_tank_id": np.asarray(leak_tank_id),
        # Ground truth
        "true_beta": beta,
        "true_b": b,
        "true_lambda": lambda_,
        "true_mean_leak": mean_leak,
    }


# %%
true_sigma = 2.0
true_alpha = 0.8
true_tau = 0.5

true_c = 3.0
true_gamma = 0.4

true_delta = 0.7

data = generate_data(
    sigma=true_sigma,
    alpha=true_alpha,
    tau=true_tau,
    c=true_c,
    gamma=true_gamma,
    delta=true_delta,
)


# %% [markdown]
# ## Define model


# %%
def model(
    V: np.ndarray,
    counts: np.ndarray,
    leak_sizes: np.ndarray,
    leak_tank_id: np.ndarray,
) -> None:

    n_tanks = len(V)

    V0 = 100.0
    v = V / V0

    sigma = numpyro.sample("sigma", dist.LogNormal(0.0, 1.0))
    alpha = numpyro.sample("alpha", dist.Normal(0.0, 1.0))
    tau = numpyro.sample("tau", dist.HalfNormal(1.0))

    with numpyro.plate("tanks", n_tanks):
        b = numpyro.sample("b", dist.Normal(0.0, tau))

    beta = jnp.exp(b)

    lambda_ = sigma * v**alpha * beta

    with numpyro.plate("tank_counts", n_tanks):
        numpyro.sample("counts", dist.Poisson(lambda_), obs=counts)

    c = numpyro.sample("c", dist.LogNormal(0.0, 1.0))

    gamma = numpyro.sample("gamma", dist.Normal(0.0, 1.0))
    delta = numpyro.sample("delta", dist.Normal(0.0, 1.0))

    mean_leak = c * v**gamma * beta**delta

    leak_mean = mean_leak[leak_tank_id]

    with numpyro.plate("leaks", len(leak_sizes)):
        numpyro.sample("leak_sizes", dist.Exponential(1.0 / leak_mean), obs=leak_sizes)


# %%
numpyro.render_model(
    model,
    model_kwargs={
        "V": jnp.asarray(data["V"]),
        "counts": jnp.asarray(data["counts"]),
        "leak_sizes": jnp.asarray(data["leak_sizes"]),
        "leak_tank_id": jnp.asarray(data["leak_tank_id"]),
    },
    render_distributions=True,
)

# %%
print("Number of tanks:", len(data["V"]))
print("Total number of leaks:", data["counts"].sum())
print("Number of leak sizes:", len(data["leak_sizes"]))

print()
print("First tank volumes:")
print(data["V"][:5])

print()
print("First leak counts:")
print(data["counts"][:5])

print()
print("First leak sizes:")
print(data["leak_sizes"][:10])

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

mcmc.run(
    jax.random.PRNGKey(0),
    V=jnp.asarray(data["V"]),
    counts=jnp.asarray(data["counts"]),
    leak_sizes=jnp.asarray(data["leak_sizes"]),
    leak_tank_id=jnp.asarray(data["leak_tank_id"]),
)

# %%
mcmc.print_summary()

# %%
samples = mcmc.get_samples()

parameters = {
    "sigma": true_sigma,
    "alpha": true_alpha,
    "tau": true_tau,
    "c": true_c,
    "gamma": true_gamma,
    "delta": true_delta,
}

for name, true_value in parameters.items():
    posterior_mean = samples[name].mean()

    print(
        f"{name:6s} | true = {true_value:6.3f} | posterior mean = {posterior_mean:6.3f}"
    )

# %%
idata = az.from_numpyro(mcmc)

# %%
posterior_samples = mcmc.get_samples()
idata = az.from_numpyro(mcmc)

# %% [markdown]
# ## Posterior distributions

# %%
azp.plot_pair(
    idata,
    var_names=[
        "sigma",
        "alpha",
        "tau",
        "c",
        "gamma",
        "delta",
    ],
)

plt.gcf().set_size_inches(12, 12)
plt.show()

# %%
fig, axes = plt.subplots(
    2,
    3,
    figsize=(12, 7),
)

axes = axes.flatten()

for ax, (name, true_value) in zip(axes, parameters.items(), strict=True):
    values = np.asarray(posterior_samples[name])

    ax.hist(
        values,
        bins=40,
        density=True,
        alpha=0.7,
    )

    ax.axvline(
        true_value,
        color="black",
        linestyle="--",
        linewidth=2,
        label="True value",
    )

    ax.set_title(name)
    ax.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Tank-specific effects

# %%
posterior_beta = np.exp(np.asarray(posterior_samples["b"]))

true_beta = np.asarray(data["true_beta"])

n_tanks = len(true_beta)

# %%
n_cols = 5
n_rows = int(np.ceil(n_tanks / n_cols))

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(15, 2.5 * n_rows),
)

axes = np.asarray(axes).flatten()

for i in range(n_tanks):
    ax = axes[i]

    ax.hist(
        posterior_beta[:, i],
        bins=30,
        density=True,
        alpha=0.7,
    )

    ax.axvline(
        true_beta[i],
        color="black",
        linestyle="--",
        linewidth=2,
        label="True beta",
    )

    ax.set_title(f"Tank {i}")
    ax.set_xlabel(r"$\beta_i$")

# Leere Subplots entfernen
for i in range(n_tanks, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

# %%
beta_mean = posterior_beta.mean(axis=0)

beta_lower = np.quantile(
    posterior_beta,
    0.05,
    axis=0,
)

beta_upper = np.quantile(
    posterior_beta,
    0.95,
    axis=0,
)

tank_id = np.arange(n_tanks)

# %%
plt.figure(figsize=(12, 6))

plt.errorbar(
    tank_id,
    beta_mean,
    yerr=[
        beta_mean - beta_lower,
        beta_upper - beta_mean,
    ],
    fmt="o",
    capsize=3,
    label="Posterior 90% interval",
)

plt.scatter(
    tank_id,
    true_beta,
    marker="x",
    s=50,
    label="True beta",
)

plt.axhline(
    1.0,
    linestyle="--",
    linewidth=1,
)

plt.xlabel("Tank")
plt.ylabel(r"$\beta_i$")
plt.title("Tank-specific failure parameters")

plt.legend()
plt.tight_layout()
plt.show()

# %%

# %%

# %%

# %%
