import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv("housing.csv", header=None, delimiter=r"\s+", names=column_names)
 

def gradient_descent(points, alpha, epochs):
    tol=1e-6
    target_col='MEDV'
    x = points.drop(columns=[target_col]).values  
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    y = points[target_col].values
    m_samples, n = len(y), x.shape[1]
    w = np.random.randn(n)*0.01
    b = 0.0
    prev_loss = float('inf')
    losses = []
    
    for epoch in range(epochs):
        y_pred = np.dot(x, w) + b
        gradients_w = (2/m_samples) * np.dot(np.transpose(x), (y_pred - y))
        gradients_b = (2/m_samples) * np.sum(y_pred- y) 
        w -= alpha * gradients_w
        b -= alpha * gradients_b

        loss =  (1 / m_samples) * np.sum((y_pred - y) ** 2) 

        if abs(prev_loss - loss) < tol:
            break

        prev_loss = loss
        losses.append(prev_loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    return w, b, loss, losses


w, b, loss, losses = gradient_descent(data,0.001, 1200)

# After GD call
plt.style.use('seaborn-v0_8-whitegrid')  # Seaborn theme
plt.figure(figsize=(10, 6))
epochs_range = range(len(losses))
sns.lineplot(x=epochs_range, y=losses, linewidth=2.5, color='steelblue')
plt.title('Gradient Descent: MSE Convergence', fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.grid(True, alpha=0.3)
plt.savefig('gd_loss_convergence.png', dpi=300, bbox_inches='tight')
plt.show()


# Recompute predictions with final w, b
target_col = 'MEDV'
x = data.drop(columns=[target_col]).values
x = (x - x.mean(axis=0)) / x.std(axis=0)
y_true = data[target_col].values
y_pred = np.dot(x, w) + b

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.scatterplot(x=y_pred, y=y_true, ax=axes[0,0])
axes[0,0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
axes[0,0].set_title('Predicted vs Actual MEDV')

residuals = y_true - y_pred
sns.histplot(residuals, kde=True, ax=axes[0,1])
axes[0,1].set_title('Residuals Histogram')

sns.boxplot(y=residuals, ax=axes[1,0])
axes[1,0].set_title('Residuals Boxplot')

sns.scatterplot(x=y_pred, y=residuals, ax=axes[1,1])
axes[1,1].axhline(0, color='red', ls='--')
axes[1,1].set_title('Residuals vs Predicted')

plt.tight_layout()
plt.savefig('gd_diagnostics.png', dpi=300, bbox_inches='tight')
plt.show()


