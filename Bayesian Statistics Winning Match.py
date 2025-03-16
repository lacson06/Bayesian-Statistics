# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# Observed data for the football team's performance
num_games = 65  # Total number of games played
num_wins = 40   # Number of wins

# Prior hyperparameters for the Beta distribution
prior_alpha = 1  # Shape parameter (uniform prior belief about winning)
prior_beta = 1   # Shape parameter (uniform prior belief about losing)

# Update the prior with the observed data to get the posterior parameters
posterior_alpha = prior_alpha + num_wins
posterior_beta = prior_beta + (num_games - num_wins)

# Generate samples from the posterior Beta distribution
posterior_samples = np.random.beta(posterior_alpha, posterior_beta, size=10000)

# Plot the posterior distribution
plt.figure(figsize=(8, 6))
plt.hist(posterior_samples, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Posterior Distribution of the Probability of Winning a Match')
plt.xlabel('Winning Probability')
plt.ylabel('Density')
plt.xlim(0, 1)  # Limiting x-axis to focus on winning probability range
plt.show()

# Calculate summary statistics
mean_winning_probability = posterior_alpha / (posterior_alpha + posterior_beta)
mode_winning_probability = (posterior_alpha - 1) / (posterior_alpha + posterior_beta - 2)  # Mode of the Beta distribution

print("Mean winning probability:", mean_winning_probability)
print("Mode winning probability:", mode_winning_probability)

print(f"There is a {round(mean_winning_probability * 100, 2)}% chance of winning their next game based on their current standing.")
