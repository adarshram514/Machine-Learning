import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Load the MNIST dataset (taking the first 30,000 samples)
mnist = datasets.fetch_openml('mnist_784', version=1)
X, y = mnist.data[:30000], mnist.target[:30000]

# Convert target to integer type
y = y.astype(int)

# Print the shape of the dataset
print(f'Shape of the dataset: {X.shape}')

# Apply t-SNE to MNIST data (reducing to 2 dimensions)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Visualize the t-SNE results
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', s=2)
plt.title('t-SNE Visualization of MNIST Dataset')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar()
plt.show()

# Save the t-SNE plot as an image
plt.savefig("SVMtsne_mnist_plot.png")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA to the standardized training data
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# After applying PCA as described previously
num_features = X_train_pca.shape[1]
print(f"Number of features after PCA: {num_features}")

# Print the shape of the dataset after applying PCA
print(X_train_pca.shape)

# Initialize the SVM model with RBF kernel
svm_model = SVC(kernel='rbf', random_state=42)

# Train the model
svm_model.fit(X_train_pca, y_train)

# Make predictions on the test set
y_pred_test = svm_model.predict(X_test_pca)

# Make predictions on the training set
y_pred_train = svm_model.predict(X_train_pca)

# Calculate and print the metrics
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test, average='weighted')
precision = precision_score(y_test, y_pred_test, average='weighted')
recall = recall_score(y_test, y_pred_test, average='weighted')

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Write the accuracies to a text file
with open("SVM_mnist_accuracies.txt", "w") as f:
    f.write('Training Accuracy: %.2f' % train_accuracy)
    f.write('\n')
    f.write('Test Accuracy: %.2f' % test_accuracy)
    f.write('\n')
    