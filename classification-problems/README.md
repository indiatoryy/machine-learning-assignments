# Comparing Binary Linear Classifier and Support Vector Machines

## Introduction
In this assignment, the objective was to empirically compare the performance of a binary linear classifier and a Support Vector Machine (SVM) using the iris dataset. The comparison was done based on accuracy, decision boundaries, margins, support vectors, and adaptability to different dataset sizes.

## Background
### Binary Linear Classification
Binary Linear Classification, a fundamental concept in machine learning, aims to categorize data points into two classes based on their features. In the case of the iris dataset, the binary linear classifier uses sepal length and width to distinguish between two classes. Logistic Regression, a widely used algorithm for binary classification, models the probability of a sample belonging to a particular class. The decision boundary, determined by the weights and biases learned during training, separates the classes. Understanding logistic regression involves grasping concepts like the sigmoid function (which maps real values into the range [0,1]) and the log-likelihood function (used for optimizing the model). Binary linear classifiers serve as the foundation for more complex machine learning algorithms and are essential for various applications, including spam detection and medical diagnosis.

### Linear Support Vector Machines (SVMs)
Support Vector Machines are powerful supervised learning models used for classification and regression tasks. In the context of binary classification, SVMs work by finding the optimal hyperplane that maximizes the margin between classes, ensuring robust generalization to unseen data. The margin, the distance between the hyperplane and the nearest data points (support vectors), is a crucial concept in SVMs. Support vectors are the data points that determine the position of the hyperplane and are instrumental in maintaining the model's stability and accuracy. SVMs can be linear or non-linear, with the choice of the kernel function determining the model's complexity and its ability to handle non-linear separable data. Understanding SVMs requires familiarity with concepts like the kernel trick, dual optimization problem, and the importance of regularization in controlling overfitting. SVMs are widely used in image classification, text mining, and bioinformatics due to their versatility and robustness.

## Implementation Details:
1. **Binary Linear Classifier (Logistic Regression):**
   - Implemented a binary linear classifier using Logistic Regression from sklearn.
   - Utilized the sigmoid function for probabilistic predictions and cross-entropy loss for optimization.
   - Visualized the decision boundary to understand the classifier's separation capabilities.

2. **Linear SVM Classifier:**
   - Implemented a linear SVM classifier using the SVM module from sklearn.
   - Optimized the SVM using the hinge loss function, aiming to maximize the margin.
   - Visualized the decision boundary and support vectors to comprehend the SVM's classification.

## Conclusion
This assignment provided valuable insights into the performance disparities between a binary linear classifier and an SVM. Through analysis and visualization, it was demonstrated how the choice of classifier and dataset split size can significantly impact decision boundaries and accuracy. Understanding these nuances is crucial for choosing the right model for specific datasets and real-world applications.
