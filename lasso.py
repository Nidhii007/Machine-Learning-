import numpy as np

def projection_oracle_l1(w, l1_norm):
  """Projects a vector onto the L1 norm ball."""
  signs = np.sign(w)
  w = w * signs
  d = len(w)
  if np.sum(w) <= l1_norm:
    return w * signs

  for i in range(d):
    w_next = w.copy()
    w_next[w > 1e-7] -= np.min(w[w > 1e-7])
    if np.sum(w_next) <= l1_norm:
      w = (l1_norm - np.sum(w_next)) * w + (np.sum(w) - l1_norm) * w_next
      return w * signs
    else:
      w = w_next
  raise ValueError("Failed to project onto L1 ball")

def train_lasso(X, y, learning_rate, l1_norm, max_iter=1000):
  """Trains a Lasso regression model using projected gradient descent."""
  n, d = X.shape
  w = np.zeros(d)
  for _ in range(max_iter):
    # Calculate gradients
    gradient = -2 * X.T.dot(y - X.dot(w)) / n
    # Update with projection
    w = projection_oracle_l1(w - learning_rate * gradient, l1_norm)
  return w

def evaluate(X, y, w):
  """Evaluates the mean squared error of the model."""
  return np.mean((y - X.dot(w))**2)

# Load data (assuming LassoReg_data.npz is loaded)
data = np.load('LassoReg_data.npz')
X = data['arr_0']
y = data['arr_1']

# Split data into train/validation/test sets (50%/25%/25%)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

# Define learning rates and L1 norm parameters to try
learning_rates = [0.01, 0.001, 0.0001]
l1_norms = [0.1, 1.0, 10.0]

# Train models with different hyperparameters and evaluate on validation set
best_model = None
best_val_error = np.inf
for lr in learning_rates:
  for l1 in l1_norms:
    model = train_lasso(X_train, y_train, lr, l1)
    val_error = evaluate(X_val, y_val, model)
    if val_error < best_val_error:
      best_model = model
      best_val_error = val_error

# Evaluate the best model on the test set
test_error = evaluate(X_test, y_test, best_model)

# Find top 10 weights (indices and values)
top_10_indices = np.argsort(np.abs(best_model))[-10:]
top_10_weights = best_model[top_10_indices]

# Print results
print(f"Test error: {test_error:.4f}")
print(f"Top 10 weight indices: {top_10_indices}")
print(f"Top 10 weight values: {top_10_weights}")

# Conclusions
# This code performs Lasso regression using projected gradient descent. It searches for the best hyperparameters (learning rate and L1 norm) based on the validation set performance. The final results include the test error, the indices of the top 10 weights by absolute value, and their corresponding values.

# This approach helps in achieving feature selection as Lasso regression drives some weights to zero. Analyzing the top weights can provide insights into the most important features for prediction. The optimal learning rate and L1 norm value will depend on the specific dataset. Experimenting with different hyperparameters is crucial
