
# Homework 2 Assignments - Changes and Updates (Part 1)

This README file explains the modifications made from the original Homework 2 Part 1 file to the completed version. Below are the changes made step-by-step.

---

## **General Overview**
The following updates were applied to improve the functionality and completeness of the assignment:
1. Added complete implementations for machine learning models including Logistic Regression, Decision Tree, and Random Forest.
2. Improved data preprocessing and visualization steps.
3. Replaced placeholder comments with functional and executable code.

---

## **Specific Changes**

### **1. Logistic Regression Implementation**
- **Original Code:**
  ```python
  # Logistic Regression Implementation
  '''
  Add code here for training and evaluating the logistic regression model.
  '''
  ```
- **Updated Code:**
  ```python
  log_regressor = LogisticRegression(max_iter=200)  # Train the Logistic Regression
  log_regressor.fit(X_train[:, :2], y_train)  # Train the Logistic Regression
  y_pred = log_regressor.predict(X_test[:, :2])  # Make predictions

  # Visualization
  plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis')
  plt.title("Logistic Regression Decision Boundary")
  plt.show()
  ```

**Change Summary:** Replaced placeholder comments with a complete implementation for Logistic Regression training, prediction, and visualization.
![image](https://github.com/user-attachments/assets/b873d4a9-3e68-49ea-8fd6-b5fc841767da)

---

### **2. Decision Tree Implementation**
- **Original Code:**
  ```python
  # Decision Tree Implementation
  '''
  Add code for training and evaluating a decision tree classifier here.
  '''
  ```
- **Updated Code:**
  ```python
  tree = DecisionTreeClassifier()  # Train the Decision Tree
  tree.fit(X_train, y_train)  # Train the Decision Tree
  y_pred = tree.predict(X_test)  # Make predictions

  # Visualization of decision tree
  dot_data = export_graphviz(tree, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
  graph = graphviz.Source(dot_data)
  graph.render("iris_decision_tree", format="png")
  ```

**Change Summary:** Added code for training a Decision Tree classifier and generating a visual representation of the trained tree.
![image](https://github.com/user-attachments/assets/3911cb84-74b6-4174-8722-feaa6e819dd0)

---

### **3. Random Forest Implementation**
- **Original Code:**
  ```python
  # Random Forest Implementation
  '''
  Complete the random forest implementation and visualize feature importance.
  '''
  ```
- **Updated Code:**
  ```python
  forest = RandomForestClassifier()  # Train the Random Forest
  forest.fit(X_train, y_train)  # Train the Random Forest

  # Visualization of feature importance
  importances = forest.feature_importances_
  indices = np.argsort(importances)[::-1]
  plt.title("Feature Importances")
  plt.bar(range(X_train.shape[1]), importances[indices], align='center')
  plt.xticks(range(X_train.shape[1]), np.array(iris.feature_names)[indices], rotation=90)
  plt.tight_layout()
  plt.show()
  ```

**Change Summary:** Completed the Random Forest implementation with training and visualization of feature importance.

![image](https://github.com/user-attachments/assets/ac810530-a1ed-4a0c-8e55-28b37087cc0b)

---
![image](https://github.com/user-attachments/assets/bb4de881-a799-43fa-b118-150c5399b1c8)


---
