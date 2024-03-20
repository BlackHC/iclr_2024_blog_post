#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

# Note on area element:
# The area element is used to normalize the densities and calculate the KL divergence.
# It is the product of the step sizes in the x and y directions.
# This is necessary because the densities are defined on a grid, and the area element
# ensures that the densities are properly normalized.
def normalize_log_density(log_density, area_element):
    """Normalize the log density in log space."""
    log_partition_constant = logsumexp(log_density) + np.log(area_element)
    return log_density - log_partition_constant


def gaussian_1D_log(x, mu, sigma):
    """Calculate the log density of the flat Gaussian for y-coordinate."""
    return -0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))

# Resolution
num_steps = 400

# Meshgrid for visualization
x, x_step = np.linspace(-2, 2, num_steps, dtype=np.float64, retstep=True)
y, y_step = np.linspace(-2, 2, num_steps, dtype=np.float64, retstep=True)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

area_element = x_step * y_step

# Calculate the log densities
log_density_upper_half = gaussian_1D_log(R, mu=1, sigma=0.1) + gaussian_1D_log(Y, mu=1, sigma=1)
log_density_lower_half = gaussian_1D_log(R, mu=1, sigma=0.1) + gaussian_1D_log(Y, mu=-1, sigma=1)

# Normalize the log densities in log space
log_density_upper_half = normalize_log_density(log_density_upper_half, area_element)
log_density_lower_half = normalize_log_density(log_density_lower_half, area_element)

# Convert log densities back to linear scale for visualization
density_upper_half = np.exp(log_density_upper_half)
density_lower_half = np.exp(log_density_lower_half)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
cmap = "viridis"

# Upper half
contour_upper = axes[0].contourf(X, Y, density_upper_half, levels=50, cmap=cmap)
axes[0].set_title('Upper Half-Circle Prior')
axes[0].axis('equal')
fig.colorbar(contour_upper, ax=axes[0], label='Density')

# Lower half
contour_lower = axes[1].contourf(X, Y, density_lower_half, levels=50, cmap=cmap)
axes[1].set_title('Lower Half-Circle Prior')
axes[1].axis('equal')
fig.colorbar(contour_lower, ax=axes[1], label='Density')
plt.show()
#%%

def kl_divergence_numerical_log(p_log, q_log, area_element=area_element):
    """Numerically approximate the KL divergence between two distributions in log space."""
    # Calculate the KL divergence in log space
    return np.sum(np.exp(p_log) * (p_log - q_log)) * area_element

# Compute the KL divergence from the upper half-circle prior to the lower half-circle prior in log space
kl_div_upper_to_lower = kl_divergence_numerical_log(log_density_upper_half, log_density_lower_half)

# Compute the KL divergence from the lower half-circle prior to the upper half-circle prior in log space
kl_div_lower_to_upper = kl_divergence_numerical_log(log_density_lower_half, log_density_upper_half)

print("KL divergence from upper to lower half-circle prior:", kl_div_upper_to_lower)
print("KL divergence from lower to upper half-circle prior:", kl_div_lower_to_upper)

# %%
# p((x', y') | x, y) = N(x'; x, 0.1) * N(y'; y, 0.1
log_likelihood = gaussian_1D_log(-X, mu=0, sigma=0.1) + gaussian_1D_log(-Y, mu=0, sigma=1)

# Normalize the log likelihood in log space
log_likelihood = normalize_log_density(log_likelihood, area_element)

# Convert log likelihood back to linear scale for visualization
likelihood = np.exp(log_likelihood)

# Plot the likelihood
plt.contourf(X, Y, likelihood, levels=50, cmap="viridis")
plt.title('Likelihood')
plt.axis('equal')
plt.show()

# %%
log_posterior_upper = log_likelihood * 1.0 + log_density_upper_half
log_posterior_lower = log_likelihood * 1.0 + log_density_lower_half

# Normalize the log posteriors in log space
log_posterior_upper = normalize_log_density(log_posterior_upper, area_element)
log_posterior_lower = normalize_log_density(log_posterior_lower, area_element)

# Plot the posteriors
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

posterior_upper = np.exp(log_posterior_upper)
posterior_lower = np.exp(log_posterior_lower)

# Upper half
contour_upper = axes[0].contourf(X, Y, posterior_upper, levels=50, cmap="viridis")
axes[0].set_title('Upper Half-Circle Posterior')
axes[0].axis('equal')
fig.colorbar(contour_upper, ax=axes[0], label='Probability Density')

# Lower half
contour_lower = axes[1].contourf(X, Y, posterior_lower, levels=50, cmap="viridis")
axes[1].set_title('Lower Half-Circle Posterior')
axes[1].axis('equal')
fig.colorbar(contour_lower, ax=axes[1], label='Probability Density')
plt.show()


# %% Compute the KL divergence from the upper half-circle posterior to the lower half-circle posterior in log space
kl_div_upper_to_lower_posterior = kl_divergence_numerical_log(log_posterior_upper, log_posterior_lower)

# Compute the KL divergence from the lower half-circle posterior to the upper half-circle posterior in log space
kl_div_lower_to_upper_posterior = kl_divergence_numerical_log(log_posterior_lower, log_posterior_upper)

print("KL divergence from upper to lower half-circle posterior:", kl_div_upper_to_lower_posterior)
print("KL divergence from lower to upper half-circle posterior:", kl_div_lower_to_upper_posterior)
# %%
# Create a table with the KL divergences
import pandas as pd

kl_divergences = pd.DataFrame({
    'Prior': [kl_div_upper_to_lower, kl_div_lower_to_upper],
    'Posterior': [kl_div_upper_to_lower_posterior, kl_div_lower_to_upper_posterior]
}, index=['Upper to Lower', 'Lower to Upper'])

kl_divergences
# %%
# Print dataframe for Tweet
print(kl_divergences.to_markdown())

# %%
