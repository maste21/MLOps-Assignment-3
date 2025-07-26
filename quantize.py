import numpy as np
import torch
import torch.nn as nn
from joblib import load, dump
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

sklearn_model = load('linear_regression.joblib')

weights = sklearn_model.coef_
bias = sklearn_model.intercept_

unquant_params = {
    'weights': weights,
    'bias': bias
}
dump(unquant_params, 'unquant_params.joblib')

def quantize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    if min_val == max_val:
        return np.zeros_like(arr, dtype=np.uint8), 1.0, 0.0
    scale = (max_val - min_val) / 255
    zero_point = np.round(-min_val / scale)
    quantized = np.round(arr / scale + zero_point).astype(np.uint8)
    return quantized, scale, zero_point

quant_weights, w_scale, w_zero = quantize(weights)
quant_bias, b_scale, b_zero = quantize(np.array([bias]))  

quant_params = {
    'weights': quant_weights,
    'bias': quant_bias[0],  
    'w_scale': w_scale,
    'w_zero': w_zero,
    'b_scale': b_scale,
    'b_zero': b_zero
}
dump(quant_params, 'quant_params.joblib')

def dequantize(quantized, scale, zero_point):
    return (quantized.astype(np.float32) - zero_point) * scale

dequant_weights = dequantize(quant_weights, w_scale, w_zero)
dequant_bias = dequantize(np.array([quant_bias]), b_scale, b_zero)[0] 

class SingleLayerNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
    
    def forward(self, x):
        return self.linear(x)

data = fetch_california_housing()
model = SingleLayerNN(data.data.shape[1])

with torch.no_grad():
    model.linear.weight.data = torch.from_numpy(dequant_weights).float().unsqueeze(0)
    model.linear.bias.data = torch.tensor(dequant_bias).float()

torch.save(model.state_dict(), 'quantized_model.pth')

X, y = data.data, data.target
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sklearn_score = sklearn_model.score(X_test, y_test)

X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()
with torch.no_grad():
    predictions = model(X_test_tensor).squeeze()
    ss_res = torch.sum((y_test_tensor - predictions) ** 2)
    ss_tot = torch.sum((y_test_tensor - torch.mean(y_test_tensor)) ** 2)
    pytorch_score = 1 - ss_res / ss_tot

print(f"Sklearn R² score: {sklearn_score:.4f}")
print(f"Quantized PyTorch R² score: {pytorch_score.item():.4f}")

import os
unquant_size = os.path.getsize('unquant_params.joblib') / 1024
quant_size = os.path.getsize('quant_params.joblib') / 1024

print("\nComparison Table:")
print("| Metric               | Original Sklearn Model | Quantized Model |")
print("|----------------------|------------------------|-----------------|")
print(f"| R² Score            | {sklearn_score:.4f}            | {pytorch_score.item():.4f}     |")
print(f"| Model Size (KB)     | {unquant_size:.2f} KB          | {quant_size:.2f} KB    |")