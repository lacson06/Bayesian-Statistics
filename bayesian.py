import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt

mu = np.linspace(1.65, 1.8, num = 50)
test = np.linspace(0, 2)

# Define the uniform distribution
uniform_dist = sts.uniform.pdf(mu, loc=1.65, scale=0.15) + 1  # Adjusted scale to match the range of mu

uniform_dist = uniform_dist/uniform_dist.sum()

# Define the beta distribution
beta_dist = sts.beta.pdf(mu, 2, 5, loc=1.65, scale=0.15)
beta_dist = beta_dist/beta_dist.sum()

# Plot the distributions
plt.plot(mu, beta_dist, label = 'Beta Dist')
plt.plot(mu, uniform_dist, label = 'Uniform Dist')
plt.xlabel("Value of $ \mu $ in meters")
plt.ylabel("Probability density")
plt.legend()

# Define the likelihood function
def likelihood_func(datum, mu):
    likelihood_out = sts.norm.pdf(datum, mu, scale=0.1)
    return likelihood_out/likelihood_out.sum()

# Calculate the likelihood for the observation (1.7)
likelihood_out = likelihood_func(1.7, mu)

# Plot the likelihood
plt.plot(mu, likelihood_out)
plt.title("Likelihood of $\mu$ given observation 1.7m")
plt.ylabel("Probability of Density/Likelihood")
plt.xlabel("Value of $\mu$")
plt.show()

# Calculate unnormalized posterior
unnormalized_posterior = likelihood_out * uniform_dist

# Plot the unnormalized posterior
plt.plot(mu, unnormalized_posterior)
plt.xlabel("$\mu$ in meters")
plt.ylabel("Unnormalized Posterior")
plt.show()
