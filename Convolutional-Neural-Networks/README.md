# Deep Learning with Convolutional Neural Networks

## Part 1: Building a CNN

1. **Data Understanding:**
   - Examine data split ratios for training, validation, and test sets.
   - Calculate the number of iterations needed to process the entire training set during gradient descent and within 30 epochs.

2. **Custom Convolution Filter:**
   - Implement a custom convolution filter and validate its output against Objax's convolution routine.

3. **Linear Layer Implementation:**
   - Implement a linear layer and verify its output against Objax's linear layer.

4. **Training vs Validation Set:**
   - Explain the distinction between the training and validation sets.

## Part 2: Training and Tuning a CNN

1. **Optimizer Implementation:**
   - Complete the stochastic gradient descent optimizer definition.

2. **Batch Sampling:**
   - Implement batch sampling using train indices and val indices.

3. **Training and Validation Monitoring:**
   - Train the model for a few epochs, monitoring loss and accuracy. Include plots in the PDF.

4. **Understanding Hyperparameters:**
   - Define "hyperparameter" and explain why evaluating accuracy on the test set should be deferred until hyperparameters are tuned.

5. **Hyperparameter Selection:**
   - Choose 4 hyperparameters, including CNN architecture, and create two different sets.
   
6. **Model Creation and Training:**
   - Create two additional models (M1 and M2) with the selected hyperparameters.
   - Report best validation accuracy and corresponding epoch for Base Model, M1, and M2.

7. **Final Model Selection:**
   - Discuss which model is chosen as the final model based on validation accuracy and why.
   - Evaluate the final model on the test set and report the accuracy.

## Part 3: Trying Out a New Dataset

1. **Data Import and Partition:**
   - Import and partition the new dataset.

2. **Base Model Creation:**
   - Create a base model for the new dataset.

3. **Hyperparameter Tuning:**
   - Tune hyperparameters to achieve a validation accuracy 5-10% better than the base model.
   - Discuss the design procedure and the impact of the tuned hyperparameters on accuracy.

4. **Final Model and Test Accuracy:**
   - Select the final model and report the test accuracy.

## Part 4: Open-Ended Exploration
**Additional Hyperparameter Tuning:**
   - Experiment with new hyperparameters and achieve a 5-10% increase in validation accuracy.
   - Discuss the performance on the test set compared to the base model.
