import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
from sklearn.model_selection import train_test_split  # Only for split

# Load data
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv("housing.csv", header=None, delimiter=r"\s+", names=column_names)
target_col = 'MEDV'

X = data.drop(columns=[target_col])
y = data[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape}, Test: {X_test.shape}") 

np.random.seed(42)  
mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma  
y_train, y_test = y_train.values, y_test.values  

def gradient_descent(X_train, y_train, X_test, y_test, alpha=0.001, epochs=1200, tol=1e-4):
    m_train, n = X_train.shape
    w = np.random.randn(n) * 0.01
    b = 0.0
    losses = []
    
    for epoch in range(epochs):
        y_pred_train = np.dot(X_train, w) + b
        loss = (1 / m_train) * np.sum((y_pred_train - y_train) ** 2)
        losses.append(loss)
        
        dw = (2 / m_train) * np.dot(X_train.T, (y_pred_train - y_train))
        db = (2 / m_train) * np.sum(y_pred_train - y_train)
        w -= alpha * dw
        b -= alpha * db
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Train Loss = {loss:.4f}")
        
        # Early stopping
        if epoch > 10 and abs(losses[-1] - losses[-2]) < tol:
            print(f"Converged at epoch {epoch}")
            break
    
    y_pred_test = np.dot(X_test, w) + b
    
    def compute_metrics(y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}
    
    train_metrics = compute_metrics(y_train, np.dot(X_train, w) + b)
    test_metrics = compute_metrics(y_test, y_pred_test)
    
    # Print weights and metrics
    feature_names = X.columns
    for name, weight in zip(feature_names, w):
        print(f"{name}: {weight:.4f}")
    print(f"Bias: {b:.4f}")
    print("\nTrain Metrics:", train_metrics)
    print("Test Metrics:", test_metrics)
    
    return w, b, losses, train_metrics, test_metrics


w, b, losses, train_metrics, test_metrics = gradient_descent(X_train, y_train, X_test, y_test)

# Test predictions (recompute for plotting)
X_test_np = X_test.values  # Ensure numpy
y_pred_test = np.dot(X_test_np, w) + b

# Train loss plot
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))
plt.plot(losses, linewidth=2.5, color='steelblue')
plt.title('GD Train Loss Convergence')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.savefig('gd_loss.png', dpi=300, bbox_inches='tight')
plt.show()

# Test scatter
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred_test, alpha=0.6)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('True MEDV')
ax.set_ylabel('Predicted MEDV')
ax.set_title(f'Test Predictions (R²={test_metrics["R²"]:.3f})')
plt.savefig('test_scatter.png', dpi=300, bbox_inches='tight')
plt.show()
