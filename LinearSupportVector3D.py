from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load a simple dataset (e.g., iris dataset)
iris = datasets.load_iris()
X = iris.data[:, :3]  # Taking the first three features for 3D visualization
y = (iris.target != 0) * 1  # Binary classification (versus non-setosa)

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the SVM model
svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X_train, y_train)

# Plot decision boundary in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of training data
scatter = ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train, cmap='winter', s=50)

# Create grid to evaluate model
xx, yy = np.meshgrid(np.linspace(X_train[:, 0].min() - 1, X_train[:, 0].max() + 1, 50),
                     np.linspace(X_train[:, 1].min() - 1, X_train[:, 1].max() + 1, 50))

# Calculate the decision boundary plane
coef = svm_model.coef_.ravel()
intercept = svm_model.intercept_
zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]

# Plot the decision boundary
ax.plot_surface(xx, yy, zz, color='k', alpha=0.3)

# Set labels
ax.set_xlabel('Feature 1 (Sepal Length)')
ax.set_ylabel('Feature 2 (Sepal Width)')
ax.set_zlabel('Feature 3 (Petal Length)')

plt.title('3D Visualization of Linear SVM Decision Boundary')
plt.show()
