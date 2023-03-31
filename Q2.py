import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from collections import Counter

# Define true GMM parameters
true_means = np.array([[0, 0], [3, 3], [-3, 3], [0, -3]])
true_covs = np.array([[[1, 0], [0, 1]], [[2, 0], [0, 2]], [[3, 0], [0, 3]], [[1, 0], [0, 1]]])
true_weights = np.array([0.25, 0.25, 0.25, 0.25])


def generate_data(n_samples):
    n_components = len(true_means)
    component_choices = np.random.choice(n_components, n_samples, p=true_weights)
    return (
    np.array([np.random.multivariate_normal(true_means[choice], true_covs[choice]) for choice in component_choices]),
    component_choices)


# Parameters
n_repeats = 50
sample_sizes = [10, 100, 1000, 10000]
max_components = 6
n_splits = 10

# Initialize results
results = np.zeros((len(sample_sizes), max_components))

# Define colors for each Gaussian component
colors = ['blue', 'green', 'red', 'yellow']

# Main loop
for i, n_samples in enumerate(sample_sizes):
    for _ in range(n_repeats):
        data, components = generate_data(n_samples)
        log_likelihoods = []

        for n_components in range(1, max_components + 1):
            kf = KFold(n_splits=n_splits)
            log_likelihood = 0

            for train_index, test_index in kf.split(data):
                train_data, test_data = data[train_index], data[test_index]
                gmm = GaussianMixture(n_components=n_components).fit(train_data)
                log_likelihood += gmm.score(test_data)

            log_likelihood /= n_splits
            log_likelihoods.append(log_likelihood)

        selected_order = np.argmax(log_likelihoods)
        results[i, selected_order] += 1

    # Plot dataset with specified colors for each Gaussian component
    for j in range(len(true_means)):
        plt.scatter(data[components == j][:, 0], data[components == j][:, 1], s=3, c=colors[j],
                    label=f'Gaussian {j + 1}')

    plt.title(f"Dataset with {n_samples} samples")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

# Convert to percentages
results /= n_repeats
results *= 100

# Display results
print("Results (percentages):")
print(results)

# Plot results
fig, ax = plt.subplots()
im = ax.imshow(results, cmap='plasma')

ax.set_xticks(np.arange(max_components))
ax.set_yticks(np.arange(len(sample_sizes)))

ax.set_xticklabels(np.arange(1, max_components + 1))
ax.set_yticklabels(sample_sizes)

ax.set_xlabel("GMM order")
ax.set_ylabel("Sample size")
ax.set_title("Model order selection")

for i in range(len(sample_sizes)):
    for j in range(max_components):
        text = ax.text(j, i, f"{results[i, j]:.1f}%", ha="center", va="center", color="w")

plt.show()