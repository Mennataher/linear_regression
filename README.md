Assignment 1 AI

# linear_regression
10 types of optimizers in Machine Learning
1. Gradient Descent (GD)

Updates weights using the average gradient computed over the entire dataset.

2. Stochastic Gradient Descent (SGD)

Updates weights after every single training sample, making updates fast but noisy.

3. Mini-Batch Gradient Descent

Updates weights using a small batch of samples, balancing speed and stability.

4. Momentum

Accelerates gradient descent by adding a fraction of the previous update to the current one.

5. Nesterov Accelerated Gradient (NAG)

Improves Momentum by calculating the gradient after making a “look-ahead” step.

6. AdaGrad

Adjusts learning rate individually for each parameter based on past gradients.

7. RMSProp

Uses a moving average of squared gradients to normalize and stabilize learning rates.

8. Adam (Adaptive Moment Estimation)

Combines Momentum and RMSProp, making it one of the most popular optimizers.

9. AdaDelta

An improved version of AdaGrad that prevents learning rate from shrinking too much.

10. AdamW

A variation of Adam that separates weight decay from the gradient update for better regularization.
