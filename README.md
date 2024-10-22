# Machine Learning Models from Scratch

This project demonstrates the implementation of three powerful machine learning models for regression tasks: Support Vector Machine (SVM), Multi-Layer Perceptron (MLP) with two hidden layers, and Random Forest. All models are built from scratch, showcasing the underlying algorithms and principles without relying on high-level libraries like scikit-learn. Additionally, Principal Component Analysis (PCA) was employed to preprocess the data and enhance model performance.

## Project Overview

The primary objective of this project is to predict continuous outcomes based on a given dataset, utilizing the three different models mentioned above. Each model was meticulously crafted to handle regression tasks, allowing for a comprehensive understanding of their functionality and performance.

### Models Implemented
Support Vector Machine (SVM):
The SVM model was designed to find the optimal hyperplane that minimizes prediction error. It incorporates a custom kernel function to enable non-linear regression. Hyperparameter tuning was performed to adjust parameters such as the regularization strength and kernel type, ensuring the model is well-fitted to the data.
Multi-Layer Perceptron (MLP):
The MLP consists of an input layer, two hidden layers, and an output layer. It utilizes backpropagation for training and gradient descent optimization. Hyperparameter tuning was conducted to refine aspects like the learning rate, the number of epochs, and the number of neurons in each layer, ultimately enhancing the model’s performance.
Random Forest:
As an ensemble learning method, Random Forest was implemented to leverage the power of multiple decision trees. Each tree is trained on a bootstrap sample of the data, and hyperparameter tuning was performed to optimize parameters such as the number of trees, maximum depth of each tree, and the number of features considered at each split. This model achieved the lowest error among all implementations, demonstrating its robustness in regression tasks.
### Principal Component Analysis (PCA)
PCA was employed as a preprocessing step to reduce the dimensionality of the dataset. By transforming the original features into a smaller set of uncorrelated variables (principal components), PCA helps in retaining the most significant variance in the data. This reduction not only speeds up the training process but also helps in mitigating issues related to overfitting.

### Performance Evaluation
The performance of each model was rigorously evaluated using Mean Squared Error (MSE), which provided a quantitative measure for comparing their predictive accuracy. Hyperparameter tuning was systematically applied across all models to identify the optimal configurations, leading to enhanced performance. The Random Forest model emerged as the most effective, exhibiting the least error compared to the SVM and MLP models.

## Conclusion
This project serves as an insightful exploration into the mechanics of various regression models, highlighting the importance of hyperparameter tuning and preprocessing techniques like PCA. By building these models from scratch, we gain a deeper appreciation for the algorithms that power machine learning, ultimately equipping ourselves with the knowledge to apply these techniques in real-world scenarios.
