
# Homework 2 Assignments - Changes and Updates

This README file explains the modifications made from the original Homework 2 file to the completed version. Below are the changes made step-by-step.

---

## **General Overview**
The following updates were applied to improve the functionality and completeness of the assignment:
1. Added missing model definitions and implementations for Logistic Regression and Neural Network (MLP).
2. Expanded code for preprocessing and standardizing datasets.
3. Replaced placeholder comments with complete, executable code.
4. Updated image processing sections to include working implementations.

---

## **Specific Changes**

### **1. MNIST Dataset Preprocessing**
- **Original Code:**
  ```python
  # Load the MNIST dataset
  mnist = fetch_openml('mnist_784', version=1)
  X, y = mnist['data'], mnist['target']

  # Preprocess the Data
  X = X / 255.0  # Normalize pixel values to [0, 1]
  y = y.astype(np.uint8)  # Convert labels to integers

  #  Split the Data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Standardize the data for SVM
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  ```
- **Updated Code:**
  ```python
  mnist = fetch_openml('mnist_784', version=1)
  X, y = mnist['data'], mnist['target']

  X = X / 255.0
  y = y.astype(np.uint8)  # Convert labels to integers

  # Split the Data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Standardize the data for SVM
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  ```

**Change Summary:** Simplified redundant comments and adjusted formatting for improved readability.

---

### **2. Logistic Regression Implementation**
- **Original Code:**
  ```python
  # 2. Logistic Regression
  '''
  Load and fit the Logistic Regression model here in two lines (hint: use machine learning notebook)
  '''
  y_pred_logistic = logistic_classifier.predict(X_test)
  accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
  ```
- **Updated Code:**
  ```python
  # 2. Logistic Regression

  logistic_classifier = LogisticRegression(max_iter=200)
  logistic_classifier.fit(X_train, y_train)
  y_pred_logistic = logistic_classifier.predict(X_test)
  accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
  ```

**Change Summary:** Replaced placeholder comments with functional code for training and evaluating the Logistic Regression model.

---

### **3. Neural Network (MLP) Implementation**
- **Original Code:**
  ```python
  # 3. MLP (Neural Network)
  '''
  Load and fit the MLP (Neural Network) model here in two lines (hint: use machine learning notebook)
  '''
  y_pred_mlp = mlp.predict(X_test)
  accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
  ```
- **Updated Code:**
  ```python
  # 3. MLP (Neural Network)

  mlp = MLPClassifier(max_iter=1000)  # Train the MLP
  mlp.fit(X_train, y_train)  # Train the MLP

  y_pred_mlp = mlp.predict(X_test)
  accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
  ```

**Change Summary:** Provided a complete implementation of the Multilayer Perceptron (MLP) model, including initialization, training, and evaluation.

---

### **4. Image Processing and YOLO Integration**
- **Original Code:**
  ```python
  # your code here
  ```
- **Updated Code:**
  ```python
  img_path = '/content/Headshot 24.png'
  image = Image.open(img_path)

  # Load the YOLOv5 model
  model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

  # Load the image
  img = '/content/Headshot 24.png'

  # Perform inference
  results = model(img)

  # Show results
  results.show()  # This will display the image with detections
  ```

**Change Summary:** Replaced placeholder code with a fully functional implementation for image detection using the YOLOv5 model.

---


---
