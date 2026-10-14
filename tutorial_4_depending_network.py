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
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from scipy.spatial import Delaunay

# %% [markdown]
# ## Generate graph


# %%
def generate_graph(n_nodes: int = 30, seed: int = 42) -> dict[str, np.ndarray]:

    rng = np.random.default_rng(seed)

    # Random node positions
    nodes = rng.uniform(
        0.0,
        1.0,
        size=(n_nodes, 2),
    )

    # Delaunay triangulation
    tri = Delaunay(nodes)

    # Extract unique edges
    edges = set()

    for triangle in tri.simplices:
        i, j, k = triangle

        edges.add(tuple(sorted((i, j))))
        edges.add(tuple(sorted((j, k))))
        edges.add(tuple(sorted((k, i))))

    edges = np.array(sorted(edges))

    # Euclidean edge lengths
    edge_lengths = np.linalg.norm(
        nodes[edges[:, 0]] - nodes[edges[:, 1]],
        axis=1,
    )

    return {
        "nodes": nodes,
        "edges": edges,
        "edge_lengths": edge_lengths,
    }


# %%
graph = generate_graph(
    n_nodes=30,
    seed=43,
)

nodes = graph["nodes"]
edges = graph["edges"]
edge_lengths = graph["edge_lengths"]

# %%
plt.figure(figsize=(7, 7))

# Edges
for i, j in edges:
    plt.plot(
        [nodes[i, 0], nodes[j, 0]],
        [nodes[i, 1], nodes[j, 1]],
        color="black",
        alpha=0.4,
        linewidth=1,
    )

# Nodes
plt.scatter(
    nodes[:, 0],
    nodes[:, 1],
    s=50,
    zorder=2,
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Delaunay graph")
plt.axis("equal")

plt.show()


# %% [markdown]
# ## Generate fake data


# %%
def generate_data(  # noqa: PLR0913
    graph: dict[str, np.ndarray],
    *,
    n_measurements: int = 100,
    lambda0: float = 2.0,
    ell: float = 0.20,
    tau: float = 3.0,
    sigma_z: float = 0.4,
    usage_min: float = 0.5,
    usage_max: float = 1.5,
    seed: int = 123,
) -> dict[str, np.ndarray]:

    rng = np.random.default_rng(seed)

    nodes = graph["nodes"]
    edges = graph["edges"]

    n_nodes = len(nodes)

    # ------------------------------------------------------------
    # 1. Pairwise Euclidean distances
    # ------------------------------------------------------------

    diff = nodes[:, None, :] - nodes[None, :, :]

    distances = np.linalg.norm(
        diff,
        axis=2,
    )

    # ------------------------------------------------------------
    # 2. Adjacency matrix
    # ------------------------------------------------------------

    adjacency = np.zeros((n_nodes, n_nodes))

    for i, j in edges:
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0

    # ------------------------------------------------------------
    # 3. Gaussian edge kernel
    # ------------------------------------------------------------

    kernel = np.exp(-(distances**2) / (2.0 * ell**2))

    # Keep only existing graph edges
    weights = adjacency * kernel

    # ------------------------------------------------------------
    # 4. Weighted graph Laplacian
    # ------------------------------------------------------------

    degree = weights.sum(axis=1)

    degree_matrix = np.diag(degree)

    laplacian = degree_matrix - weights

    # ------------------------------------------------------------
    # 5. GMRF precision matrix
    #
    # Q = (I + tau * L) / sigma_z^2
    # ------------------------------------------------------------

    precision = (np.eye(n_nodes) + tau * laplacian) / sigma_z**2

    # Needed only for simulating the data
    covariance = np.linalg.inv(precision)

    # ------------------------------------------------------------
    # 6. Fixed latent correlated node effects
    #
    # One GMRF realization for the complete experiment.
    # The latent effect z_i of each node stays constant
    # across all measurements.
    #
    # shape:
    # (n_nodes,)
    # ------------------------------------------------------------

    z = rng.multivariate_normal(
        mean=np.zeros(n_nodes),
        cov=covariance,
    )

    # ------------------------------------------------------------
    # 7. Known usage values
    #
    # Usage changes between measurements.
    #
    # shape:
    # (n_measurements, n_nodes)
    # ------------------------------------------------------------

    usage = rng.uniform(
        low=usage_min,
        high=usage_max,
        size=(n_measurements, n_nodes),
    )

    # ------------------------------------------------------------
    # 8. Poisson rates
    #
    # z has shape (n_nodes,)
    # usage has shape (n_measurements, n_nodes)
    #
    # NumPy broadcasts z over all measurements.
    # ------------------------------------------------------------

    lambda_ = lambda0 * usage * np.exp(z)

    # ------------------------------------------------------------
    # 9. Failure counts
    # ------------------------------------------------------------

    failures = rng.poisson(lambda_)

    return {
        "distances": distances,
        "adjacency": adjacency,
        "weights": weights,
        "laplacian": laplacian,
        "precision": precision,
        "covariance": covariance,
        "usage": usage,
        # Ground truth
        "true_z": z,
        "true_lambda": lambda_,
        # Observations
        "failures": failures,
    }


# %%
true_lambda0 = 10.0
true_ell = 0.20
true_tau = 3.0
true_sigma_z = 0.4

data = generate_data(
    graph,
    n_measurements=100,
    lambda0=true_lambda0,
    ell=true_ell,
    tau=true_tau,
    sigma_z=true_sigma_z,
    seed=123,
)

# %%
print("Number of nodes:", len(nodes))
print("Number of edges:", len(edges))

print()

print("Usage shape:")
print(data["usage"].shape)

print()

print("Latent effects shape:")
print(data["true_z"].shape)

print()

print("Poisson rates shape:")
print(data["true_lambda"].shape)

print()

print("Failure counts shape:")
print(data["failures"].shape)

# %%
measurement = 0

usage = data["usage"][measurement]
failures = data["failures"][measurement]

plt.figure(figsize=(8, 7))

# Edges
for i, j in edges:
    plt.plot(
        [nodes[i, 0], nodes[j, 0]],
        [nodes[i, 1], nodes[j, 1]],
        color="gray",
        alpha=0.4,
        linewidth=1,
        zorder=1,
    )


# Nodes
scatter = plt.scatter(
    nodes[:, 0],
    nodes[:, 1],
    c=usage,
    s=300,
    cmap="viridis",
    edgecolor="black",
    zorder=2,
)


# Failure counts inside nodes
for i, (x, y) in enumerate(nodes):
    plt.text(
        x,
        y,
        str(failures[i]),
        ha="center",
        va="center",
        fontsize=9,
        color="white",
        fontweight="bold",
        zorder=3,
    )


# Colorbar for usage
cbar = plt.colorbar(scatter)
cbar.set_label(r"Usage $u_i$")


plt.xlabel("x")
plt.ylabel("y")

plt.title(f"Network failures, measurement {measurement}")

plt.axis("equal")
plt.tight_layout()

plt.show()


# %% [markdown]
# ## Define model


# %%
def model(
    nodes: np.ndarray,
    edges: np.ndarray,
    usage: np.ndarray,
    failures: np.ndarray,
) -> None:

    n_nodes = nodes.shape[0]
    n_measurements = usage.shape[0]

    # ------------------------------------------------------------
    # 1. Global parameters
    # ------------------------------------------------------------

    lambda0 = numpyro.sample(
        "lambda0",
        dist.LogNormal(2.0, 0.5),
    )

    ell = numpyro.sample(
        "ell",
        dist.LogNormal(-1.5, 0.5),
    )

    tau = numpyro.sample(
        "tau",
        dist.LogNormal(1.0, 0.5),
    )

    sigma_z = numpyro.sample(
        "sigma_z",
        dist.HalfNormal(1.0),
    )

    # ------------------------------------------------------------
    # 2. Pairwise Euclidean distances
    # ------------------------------------------------------------

    diff = nodes[:, None, :] - nodes[None, :, :]

    distances = jnp.linalg.norm(
        diff,
        axis=2,
    )

    # ------------------------------------------------------------
    # 3. Adjacency matrix
    # ------------------------------------------------------------

    adjacency = jnp.zeros((n_nodes, n_nodes))

    adjacency = adjacency.at[
        edges[:, 0],
        edges[:, 1],
    ].set(1.0)

    adjacency = adjacency.at[
        edges[:, 1],
        edges[:, 0],
    ].set(1.0)

    # ------------------------------------------------------------
    # 4. Gaussian edge kernel
    # ------------------------------------------------------------

    kernel = jnp.exp(-(distances**2) / (2.0 * ell**2))

    weights = adjacency * kernel

    # ------------------------------------------------------------
    # 5. Weighted graph Laplacian
    # ------------------------------------------------------------

    degree = weights.sum(axis=1)

    laplacian = jnp.diag(degree) - weights

    # ------------------------------------------------------------
    # 6. GMRF precision matrix
    #
    # Q = (I + tau * L) / sigma_z^2
    # ------------------------------------------------------------

    precision = (jnp.eye(n_nodes) + tau * laplacian) / sigma_z**2

    # ------------------------------------------------------------
    # 7. Fixed latent correlated node effects
    #
    # One z_i for each node.
    # These effects stay constant across all measurements.
    # ------------------------------------------------------------

    z = numpyro.sample(
        "z",
        dist.MultivariateNormal(
            loc=jnp.zeros(n_nodes),
            precision_matrix=precision,
        ),
    )

    # ------------------------------------------------------------
    # 8. Poisson rates
    #
    # usage:
    # (n_measurements, n_nodes)
    #
    # z:
    # (n_nodes,)
    #
    # Broadcasting gives:
    # lambda_:
    # (n_measurements, n_nodes)
    # ------------------------------------------------------------

    lambda_ = lambda0 * usage * jnp.exp(z)

    # ------------------------------------------------------------
    # 9. Observed failures
    # ------------------------------------------------------------

    with (
        numpyro.plate(
            "measurements",
            n_measurements,
            dim=-2,
        ),
        numpyro.plate(
            "nodes",
            n_nodes,
            dim=-1,
        ),
    ):
        numpyro.sample(
            "failures",
            dist.Poisson(lambda_),
            obs=failures,
        )


# %%
nodes_jax = jnp.asarray(nodes)
edges_jax = jnp.asarray(edges)

usage_jax = jnp.asarray(data["usage"])

failures_jax = jnp.asarray(data["failures"])

numpyro.render_model(
    model,
    model_kwargs={
        "nodes": nodes_jax,
        "edges": edges_jax,
        "usage": usage_jax,
        "failures": failures_jax,
    },
    render_distributions=True,
    render_params=True,
)

# %% [markdown]
# ## Doing inference

# %%
# ------------------------------------------------------------
# Inference with NUTS / MCMC
# ------------------------------------------------------------

# Convert data to JAX arrays
nodes_jax = jnp.asarray(nodes)
edges_jax = jnp.asarray(edges)

usage_jax = jnp.asarray(data["usage"])

failures_jax = jnp.asarray(data["failures"])


# ------------------------------------------------------------
# NUTS kernel
# ------------------------------------------------------------

nuts_kernel = NUTS(
    model,
    target_accept_prob=0.9,
)


# ------------------------------------------------------------
# MCMC
# ------------------------------------------------------------

mcmc = MCMC(
    nuts_kernel,
    num_warmup=1000,
    num_samples=2000,
    num_chains=4,
)


# ------------------------------------------------------------
# Run inference
# ------------------------------------------------------------

rng_key = jax.random.PRNGKey(0)

mcmc.run(
    rng_key,
    nodes=nodes_jax,
    edges=edges_jax,
    usage=usage_jax,
    failures=failures_jax,
)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

mcmc.print_summary()

# %%
posterior_samples = mcmc.get_samples()

# %%
posterior_samples.keys()

# %%
idata = az.from_numpyro(mcmc)

# %%
az.summary(
    idata,
    var_names=[
        "lambda0",
        "ell",
        "tau",
        "sigma_z",
    ],
)

# %%
az.plot_trace(
    idata,
    var_names=[
        "lambda0",
        "ell",
        "tau",
        "sigma_z",
    ],
)

plt.tight_layout()
plt.show()

# %%
print("True values:")
print("lambda0 =", true_lambda0)
print("ell     =", true_ell)
print("tau     =", true_tau)
print("sigma_z =", true_sigma_z)

# %%
pairplot = az.plot_pair(
    idata,
    var_names=[
        "lambda0",
        "ell",
        "tau",
        "sigma_z",
    ],
    marginal=True,
    marginal_kind="kde",
)

plt.show()

# %%
divergences = mcmc.get_extra_fields()["diverging"]

print("Number of divergences:", divergences.sum())

# %% [markdown]
# ## Posterior dependence

# %%
lambda0_samples = posterior_samples["lambda0"]
z_samples = posterior_samples["z"]

# Mean latent effect for every posterior draw
mean_z_samples = z_samples.mean(axis=1)

plt.figure(figsize=(6, 5))

plt.scatter(
    lambda0_samples,
    mean_z_samples,
    alpha=0.2,
    s=10,
)

plt.xlabel(r"$\lambda_0$")
plt.ylabel(r"Mean latent effect $\bar{z}$")
plt.title(r"Posterior dependence: $\lambda_0$ vs. $\bar{z}$")

plt.tight_layout()
plt.show()

# %%
import inspect

import arviz as az

print(az.__version__)
print(inspect.signature(az.plot_pair))

# %%
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Posterior samples
# ------------------------------------------------------------

samples = {
    "lambda0": np.asarray(posterior_samples["lambda0"]),
    "ell": np.asarray(posterior_samples["ell"]),
    "tau": np.asarray(posterior_samples["tau"]),
    "sigma_z": np.asarray(posterior_samples["sigma_z"]),
}

# Ground-truth values used for simulation
true_values = {
    "lambda0": true_lambda0,
    "ell": true_ell,
    "tau": true_tau,
    "sigma_z": true_sigma_z,
}

var_names = [
    "lambda0",
    "ell",
    "tau",
    "sigma_z",
]

n_vars = len(var_names)


# ------------------------------------------------------------
# Pair plot
# ------------------------------------------------------------

fig, axes = plt.subplots(
    n_vars,
    n_vars,
    figsize=(10, 10),
)


for row in range(n_vars):
    for col in range(n_vars):
        ax = axes[row, col]

        # Upper triangle: empty
        if col > row:
            ax.axis("off")
            continue

        x_name = var_names[col]
        y_name = var_names[row]

        x = samples[x_name]

        # ----------------------------------------------------
        # Diagonal: posterior distribution
        # ----------------------------------------------------

        if row == col:
            ax.hist(
                x,
                bins=40,
                density=True,
                alpha=0.7,
            )

            # Ground truth
            ax.axvline(
                true_values[x_name],
                color="red",
                linestyle="--",
                linewidth=2,
                label="Ground truth",
            )

        # ----------------------------------------------------
        # Lower triangle: pairwise posterior samples
        # ----------------------------------------------------

        else:
            y = samples[y_name]

            ax.scatter(
                x,
                y,
                alpha=0.15,
                s=8,
            )

            # Ground truth
            ax.scatter(
                true_values[x_name],
                true_values[y_name],
                color="red",
                marker="x",
                s=100,
                linewidth=2.5,
                zorder=10,
            )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        if row == n_vars - 1:
            ax.set_xlabel(x_name)

        if col == 0 and row != 0:
            ax.set_ylabel(y_name)


# Legend
axes[0, 0].legend()

plt.tight_layout()
plt.show()

# %%
