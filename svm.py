import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Define the data points (features)
x = [
    [0.1, 2.3],  # Example 1
    [-1.5, 2.5], # Example 2
    [2.0, -4.3], # Example 3
    [1.5, -2.0], # New Example to make it more interesting
    [-2.0, 1.0]  # Another new example
]

# Define the labels (targets)
y = [0, 1, 0, 1, 0]

# Convert to numpy arrays for easy manipulation
X = np.array(x)
Y = np.array(y)

# Create the SVM classifier with an RBF kernel
clf = svm.SVC(kernel='rbf', C=1, gamma='auto')
clf.fit(X, Y)

# Plotting the data points
plt.scatter(X[Y == 0][:, 0], X[Y == 0][:, 1], color='red', marker='o', label='Class 0')
plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], color='blue', marker='x', label='Class 1')

# Create a grid to plot decision boundaries
xx, yy = np.meshgrid(np.linspace(-3, 3, 500), np.linspace(-5, 5, 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary and margins
plt.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='black')
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')

# Additional plot settings
plt.legend()
plt.title('SVM with RBF Kernel: Decision Boundary and Support Vectors')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()
