from sklearn import svm

# Define the data points (features)
x = [
    [0.1, 2.3],  # Example 1
    [-1.5, 2.5], # Example 2
    [2.0, -4.3]  # Example 3
]

# Define the labels (targets)
y = [0, 1, 0]

# Create the SVM classifier
clf = svm.SVC()

# Train the model (fit the classifier to the data)
clf.fit(x, y)

# You can make predictions on new data using clf.predict(new_data)
clf.predict(new_data)