# -----------------------------------------------------------------------
# Author: Antonio Oviedo Paredes
# Date Created: 11/09/2023
# Description: This file trains a decision tree trying to find the best
# hiperparamters for the model.
# -----------------------------------------------------------------------

# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import pandas as pd

# ------------------------ Data manipulation ----------------------------

# Import CSV into a pandas DataFrame
data = pd.read_csv("./diabetes.csv")

# One-hot encode the "gender" and "smoking_history" columns, creating dummy variables
dummies = pd.get_dummies(data[["gender", "smoking_history"]])
X = data.drop(["diabetes", "gender", "smoking_history"], axis=1)

# Observations and target data
X = pd.concat([X, dummies], axis=1)
y = data["diabetes"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# ------------------- Test for best splitting criteria ----------------------

# Accuracy and precision values for different criteria (entropy and gini)
criterion_test = {"entropy": {"accuracy": 0, "precision": 0}, "gini": {"accuracy": 0, "precision": 0}}

# Loop through 50 iterations
for i in range(0, 50):
    # Create a Decision Tree Classifier with criterion "entropy" for the first 25 iterations,
    # and "gini" for the next 25 iterations
    tree_clf = DecisionTreeClassifier(
        criterion="entropy" if i < 25 else "gini",
    )

    # Fit tree
    tree_clf.fit(X_train, y_train)
    # Predict the target values
    y_pred = tree_clf.predict(X_test)
    
    # Calculate accuracy and precision scores for the predictions
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    # Update the accuracy and precision values
    if i < 25:
        criterion_test["entropy"]["accuracy"] += accuracy
        criterion_test["entropy"]["precision"] += precision
    else:
        criterion_test["gini"]["accuracy"] += accuracy
        criterion_test["gini"]["precision"] += precision

# Calculate the average accuracy and precision for both criteria
criterion_test["entropy"]["accuracy"] /= 25
criterion_test["entropy"]["precision"] /= 25
criterion_test["gini"]["accuracy"] /= 25
criterion_test["gini"]["precision"] /= 25

print("----- Best tree splitting criteria -----")
print("Entropy: ")
print("    Accuracy: ", criterion_test["entropy"]["accuracy"])
print("    Precision: ", criterion_test["entropy"]["precision"])
print("Gini: ")
print("    Accuracy: ", criterion_test["gini"]["accuracy"])
print("    Precision: ", criterion_test["gini"]["precision"])
print("Best tree splitting criteria: ", "Entropy")


# ---------------------- Test for best max depth ------------------------

# Accuracy and precision values for different maximum depths
max_depth_test = {"accuracy": [], "precision": []}

# Loop through maximum depth values from 1 to 30
for i in range(1, 31):
    # Create a Decision Tree Classifier with criterion "entropy" and the current maximum depth value
    tree_clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=i,
    )

    # Fit tree
    tree_clf.fit(X_train, y_train)
    
    # Predict the target values
    y_pred = tree_clf.predict(X_test)

    # Calculate accuracy and precision scores 
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    max_depth_test["accuracy"].append(accuracy)
    max_depth_test["precision"].append(precision)

# Get best accuracy an precision
best_accuracy = max(max_depth_test["accuracy"])
best_precision = max(max_depth_test["precision"])

# Get index (depth) of each metric (find last occurrence)
best_depth_acc = 30 - max_depth_test["accuracy"][::-1].index(best_accuracy)
best_depth_pre = 30 - max_depth_test["precision"][::-1].index(best_precision)

# Get best depth for both metrics
best_depth = min(best_depth_acc, best_depth_pre)

# Print the best maximum depth of the tree
print()
print("----- Best max depth -----")
print("Depth: ")
for i in range(30):
    print(f'{i + 1}:')
    print("    Accuracy: ", max_depth_test["accuracy"][i])
    print("    Precision: ", max_depth_test["precision"][i])

print("Best max depth: ", best_depth)


# ------------------- Tree with best hiperparameters --------------------

# Create tree classifier
tree_clf = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=8,
)

# Fit tree
tree_clf.fit(X_train, y_train)

# Predict the target values
y_pred = tree_clf.predict(X_test)

# Model avaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
confusion_mtx = confusion_matrix(y_test, y_pred)
print()
print("----- Final model ------")
print("Hiperparapeters: ")
print("    Criterion: entropy")
print("    Max depth: 8")
print("Metrics: ")
print("    Accuracy:", accuracy)
print("    Precision:", precision)
print("    F1 score:", f1)
print("    Confusion_matrix:")
print("    ", confusion_mtx)