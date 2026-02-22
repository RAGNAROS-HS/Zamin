import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
from sklearn.model_selection import train_test_split  # Only for split
import tensorflow as tf

os.makedirs('output/nn', exist_ok=True)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

np.random.seed(42)
tf.random.set_seed(42)

# Load data
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv("housing.csv", header=None, delimiter=r"\s+", names=column_names)
target_col = 'MEDV'

X = data.drop(columns=[target_col])
y = data[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape}, Test: {X_test.shape}") 

mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma 
y_train = y_train.values
y_test = y_test.values


model = Sequential([
    Dense(22, activation='relu', input_shape=(13,)),  
    Dense(22, activation='relu'),                     
    Dense(22, activation='relu'),                     
    Dense(1, activation='linear')                     
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])  
model.summary()


def compute_metrics(y_true, y_pred):
    """Exact same metrics function from GD code"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}

def nn_training_loop(epoch, batch):
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)]
    history = model.fit(
        X_train, y_train,
        epochs=epoch,  
        batch_size=batch,  
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    model.save('output/nn/nnmod1.keras')
    
    # NEW: Compute and print exact GD metrics post-training
    y_pred_train = model.predict(X_train, verbose=0).flatten()
    y_pred_test = model.predict(X_test, verbose=0).flatten()
    
    train_metrics = compute_metrics(y_train, y_pred_train)
    test_metrics = compute_metrics(y_test, y_pred_test)
    
    print("\nNN Train Metrics:", train_metrics)
    print("NN Test Metrics:", test_metrics)
    

    plt.style.use('seaborn-v0_8-whitegrid')
    

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], linewidth=2.5, color='steelblue', label='Train Loss')
    plt.plot(history.history['val_loss'], linewidth=2.5, color='orange', label='Val Loss')
    plt.title('NN Train/Val Loss Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig('output/nn/nn_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('True MEDV')
    plt.ylabel('Predicted MEDV')
    plt.title(f'NN Test Predictions (R²={test_metrics["R²"]:.3f})')
    plt.savefig('output/nn/nn_test_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return history, train_metrics, test_metrics  


history, train_metrics, test_metrics = nn_training_loop(50, 32)
