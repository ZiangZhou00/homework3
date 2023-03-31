import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import multivariate_normal as mvn
from sklearn.model_selection import KFold


def generate_data_from_gmm(N, pdf_params):
    # Determine dimensionality from mixture PDF parameters
    n = pdf_params['class_means'].shape[1]
    # Output samples and labels
    X = np.zeros([N, n])
    labels = np.zeros(N)

    # Decide randomly which samples will come from each component
    u = np.random.rand(N)
    # Determine the thresholds based on the mixture weights/priors for the GMM, which need to sum up to 1
    thresholds = np.cumsum(pdf_params['class_priors'])
    thresholds = np.insert(thresholds, 0, 0)  # For intervals of classes

    L = np.array(range(len(pdf_params['class_priors'])))
    for l in L:
        # Get randomly sampled indices for this component
        indices = np.argwhere((thresholds[l] <= u) & (u <= thresholds[l + 1]))[:, 0]
        # No. of samples in this component
        N_labels = len(indices)
        labels[indices] = l * np.ones(N_labels)
        X[indices, :] = mvn.rvs(pdf_params['class_means'][l], pdf_params['class_covariances'][l], N_labels)

    return X, labels

# Set seed to generate reproducible "pseudo-randomness" (handles scipy's "randomness" too)
np.random.seed(7)

# Number of samples
N = 10000

# Number of classes
C = 4

gmm_pdf = {}

# Class priors
gmm_pdf['class_priors'] = np.ones(C) / C  # uniform prior

# Mean and covariance of data PDFs conditioned on labels
gmm_pdf['class_means'] = np.array([[0, 0, 0],
                        [3, 0, 0],
                        [0, 3, 0],
                        [0, 0, 3]])

gmm_pdf['class_covariances'] = np.array([[[3, 0, 0],
                               [0, 3, 0],
                               [0, 0, 3]],
                              [[3, 1, 0],
                               [1, 3, 1],
                               [0, 1, 3]],
                              [[3, 0, 1],
                               [0, 3, 0],
                               [1, 0, 3]],
                              [[3, 0, 0],
                               [0, 3, 1],
                               [0, 1, 3]]])

# Generate data from Gaussian mixture model
X = np.zeros([N, 3])
labels = np.zeros(N)

for i in range(N):
    # Randomly select a class based on class priors
    label = np.random.choice(C, p=gmm_pdf['class_priors'])
    # Generate a sample from the selected class
    X[i, :] = mvn.rvs(mean=gmm_pdf['class_means'][label], cov=gmm_pdf['class_covariances'][label])
    labels[i] = label

# Plot the original data and their true labels
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'b', 'g', 'k']
labels_str = [f'Class {i}' for i in range(C)]

for i in range(C):
    ax.scatter(X[labels == i, 0], X[labels == i, 1], X[labels == i, 2], c=colors[i], label=labels_str[i], alpha=0.5)

ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_zlabel(r"$x_3$")
ax.set_title("Data and True Class Labels")
ax.legend()
plt.tight_layout()
plt.show()


class TwoLayerMLP(nn.Module):
    # Two-layer neural network class

    def __init__(self, n, P, C):
        super(TwoLayerMLP, self).__init__()
        # Fully connected layer WX + b mapping from n -> P
        self.input_fc = nn.Linear(n, P)
        # Output layer again fully connected mapping from P -> C
        self.output_fc = nn.Linear(P, C)

    def forward(self, X):
        # X = [batch_size, input_dim]
        X = self.input_fc(X)
        # ReLU
        X = F.relu(X)
        # X = [batch_size, P]
        y = self.output_fc(X)
        return y


# Number of training input samples for experiments
N_train = [100, 200, 500, 1000, 2000, 5000]
# Number of test samples for experiments
N_test = 100000

# Lists to hold the corresponding input matrices, target vectors and sample label counts per training set
X_train = []
y_train = []
for N_i in N_train:
    print("Generated the Ntrain = {} data set".format(N_i))

    # Modulus to plot in right locations, hacking it
    X_i, y_i = generate_data_from_gmm(N_i, gmm_pdf)

    # Add to lists
    X_train.append(X_i)
    y_train.append(y_i)

print("Generated the Ntest = {} test set".format(N_test))
X_test, y_test = generate_data_from_gmm(N_test, gmm_pdf)

print("All datasets generated!")


# Conditional likelihoods of each x given each class, shape (C, N)
class_cond_likelihoods = np.array([mvn.pdf(X_test, gmm_pdf['class_means'][i], gmm_pdf['class_covariances'][i]) for i in range(C)])
decisions = np.argmax(class_cond_likelihoods, axis=0)
misclass_samples = sum(decisions != y_test)
min_prob_error = (misclass_samples / N_test)
print("Probability of Error on Test Set using the True Data PDF: {:.4f}".format(min_prob_error))


# Plot the training datasets for each value of N_train
for i, N_i in enumerate(N_train):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for j in range(C):
        ax.scatter(X_train[i][y_train[i] == j, 0], X_train[i][y_train[i] == j, 1], X_train[i][y_train[i] == j, 2],
                   c=colors[j], label=labels_str[j], alpha=0.5)

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$x_3$")
    ax.set_title("Training Data for N_train={}".format(N_i))
    ax.legend()
    plt.tight_layout()
    plt.show()


# Lists to hold the corresponding error estimates
train_errors = []
test_errors = []

# Iterate over each training set size and train the model
for X_i, y_i, N_i in zip(X_train, y_train, N_train):
    print("Training the model on {} samples".format(N_i))

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_i).float(), torch.tensor(y_i).long())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Create model and optimizer
    model = TwoLayerMLP(n=3, P=10, C=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Train the model
    criterion = nn.CrossEntropyLoss()
    num_epochs = 500
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate the model on the training set
    with torch.no_grad():
        train_outputs = model(torch.tensor(X_i).float())
        train_loss = criterion(train_outputs, torch.tensor(y_i).long())
        train_pred = torch.argmax(train_outputs, dim=1)
        train_error = 1 - torch.sum(train_pred == torch.tensor(y_i)) / len(y_i)
        train_errors.append(train_error)

    # Evaluate the model on the test set
    with torch.no_grad():
        test_outputs = model(torch.tensor(X_test).float())
        test_loss = criterion(test_outputs, torch.tensor(y_test).long())
        test_pred = torch.argmax(test_outputs, dim=1)
        test_error = 1 - torch.sum(test_pred == torch.tensor(y_test)) / len(y_test)
        test_errors.append(test_error)

# Plot the error estimates
plt.plot(N_train, train_errors, label="Training")
plt.plot(N_train, test_errors, label="Test")
plt.xscale('log')
plt.xlabel("Number of Training Samples")
plt.ylabel("Error Estimate")
plt.title("Error Estimate vs. Number of Training Samples")
plt.legend()
plt.show()


def model_train(model, data, labels, optimizer, criterion=nn.CrossEntropyLoss(), num_epochs=100):
    # Set this "flag" before training
    model.train()
    # Optimize the model, e.g. a neural network
    for epoch in range(num_epochs):
        # These outputs represent the model's predicted probabilities for each class.
        outputs = model(data)
        # Criterion computes the cross entropy loss between input and target
        loss = criterion(outputs, labels)
        # Set gradient buffers to zero explicitly before backprop
        optimizer.zero_grad()
        # Backward pass to compute the gradients through the network
        loss.backward()
        # GD step update
        optimizer.step()
    return model, loss


def model_predict(model, data):
    # Similar idea to model.train(), set a flag to let network know your in "inference" mode
    model.eval()
    # Disabling gradient calculation is useful for inference, only forward pass!!
    with torch.no_grad():
        # Evaluate nn on test data and compare to true labels
        predicted_labels = model(data)
        # Back to numpy
        predicted_labels = predicted_labels.detach().numpy()
        return np.argmax(predicted_labels, 1)


def k_fold_cv_perceptrons(K, P_list, data, labels):
    # STEP 1: Partition the dataset into K approximately-equal-sized partitions
    kf = KFold(n_splits=K, shuffle=True)

    # Allocate space for CV
    error_valid_mk = np.zeros((len(P_list), K))

    # STEP 2: Iterate over all model options based on number of perceptrons
    # Track model index
    m = 0
    for P in P_list:
        # K-fold cross validation
        k = 0
        for train_indices, valid_indices in kf.split(data):
            # Extract the training and validation sets from the K-fold split
            # Convert numpy structures to PyTorch tensors, necessary data types
            X_train_k = torch.FloatTensor(data[train_indices])
            y_train_k = torch.LongTensor(labels[train_indices])

            model = TwoLayerMLP(X_train_k.shape[1], P, C)

            # Stochastic GD with learning rate and momentum hyperparameters
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

            # Trained model
            model, _ = model_train(model, X_train_k, y_train_k, optimizer)

            X_valid_k = torch.FloatTensor(data[valid_indices])
            y_valid_k = labels[valid_indices]

            # Evaluate the neural network on the validation fold
            predictions = model_predict(model, X_valid_k)
            # Retain the probability of error estimates
            error_valid_mk[m, k] = np.sum(predictions != y_valid_k) / len(y_valid_k)
            k += 1
        m += 1

    # STEP 3: Compute the average prob. error (across K folds) for that model
    error_valid_m = np.mean(error_valid_mk, axis=1)

    # Return the optimal choice of P* and prepare to train selected model on entire dataset
    optimal_P = P_list[np.argmin(error_valid_m)]

    return optimal_P, error_valid_m


# Number of folds for CV
K = 10
P_list = [2, 4, 8, 16, 24, 32, 48, 64, 128, 256, 512]
# List of best no. of perceptrons for MLPs per training set
P_best_list = []

fig, ax = plt.subplots(figsize=(10, 8))

print("\tnumber of Training Samples \tBest number of Perceptrons \tPr(error)")
for i in range(len(X_train)):
    P_best, P_CV_err = k_fold_cv_perceptrons(K, P_list, X_train[i], y_train[i])
    P_best_list.append(P_best)
    print("\t\t %d \t\t\t %d \t\t  %.3f" % (N_train[i], P_best, np.min(P_CV_err)))
    ax.plot(P_list, P_CV_err, label="N = {}".format(N_train[i]))

plt.axhline(y=min_prob_error, color="black", linestyle="--", label="Min. Pr(error)")
ax.set_title("No. Perceptrons vs Cross-Validation Pr(error)")
ax.set_xlabel(r"$P$")
ax.set_ylabel("Pr(error)")
ax.legend();
plt.show()

# List of trained MlPs for later testing
trained_mlps = []
# Number of times to re-train same model with random re-initializations
num_restarts = 10

for i in range(len(X_train)):
    print("Training model for N = {}".format(X_train[i].shape[0]))
    X_i = torch.FloatTensor(X_train[i])
    y_i = torch.LongTensor(y_train[i])

    restart_mlps = []
    restart_losses = []
    # Remove chances of falling into suboptimal local minima
    for r in range(num_restarts):
        model = TwoLayerMLP(X_i.shape[1], P_best_list[i], C)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        # Trained model
        model, loss = model_train(model, X_i, y_i, optimizer)
        restart_mlps.append(model)
        restart_losses.append(loss.detach().item())

    # Add best model from multiple restarts to list
    trained_mlps.append(restart_mlps[np.argmin(restart_losses)])


# First conver test set data to tensor suitable for PyTorch models
X_test_tensor = torch.FloatTensor(X_test)
pr_error_list = []

fig, ax = plt.subplots(figsize=(10, 8))

# Estimate loss (probability of error) for each trained MLP model by testing on the test data set
print("Probability of error results summarized below per trained MLP: \n")
print("\t number of Training Samples \t Pr(error)")
for i in range(len(X_train)):
    # Evaluate the neural network on the test set
    predictions = model_predict(trained_mlps[i], X_test_tensor)
    # Compute the probability of error estimates
    prob_error = np.sum(predictions != y_test) / len(y_test)
    print("\t\t %d \t\t   %.3f" % (N_train[i], prob_error))
    pr_error_list.append(prob_error)

plt.axhline(y=min_prob_error, color="black", linestyle="--", label="Min. Pr(error)")
ax.plot(np.log10(N_train), pr_error_list)
ax.set_title("No. Training Samples vs Test Set Pr(error)")
ax.set_xlabel("MLP Classifier")
ax.set_ylabel("Pr(error)")
ax.legend()
plt.show()

#Reference from Mark Zolotas