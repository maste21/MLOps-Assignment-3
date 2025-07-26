import numpy as np
import torch
import torch.nn as nn
from joblib import load, dump
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import os

def quantize(params):
    params = np.array(params, dtype=np.float32)
    min_val = np.min(params)
    max_val = np.max(params)
    
    if np.isclose(min_val, max_val, atol=1e-6):
        return np.zeros_like(params, dtype=np.uint8), 1.0, min_val
    
    scale = 255.0 / (max_val - min_val)
    quantized = np.clip(np.round(scale * (params - min_val)), 0, 255).astype(np.uint8)
    return quantized, (max_val - min_val)/255.0, min_val

def get_file_size_kb(filename):
    return round(os.path.getsize(filename) / 1024, 3)

def main():
    sklearn_model = load('linear_regression.joblib')
    weights = sklearn_model.coef_.astype(np.float32)
    bias = np.float32(sklearn_model.intercept_)
    
    dump({'weights': weights, 'bias': bias}, 'unquant_params.joblib', compress=3)
    unquant_size = get_file_size_kb('unquant_params.joblib')
    
    quant_w, w_scale, w_min = quantize(weights)
    quant_b, b_scale, b_min = quantize(bias)
    
    quant_params = {
        'weights': quant_w,
        'bias': quant_b,
        'w_scale': np.float32(w_scale),
        'w_min': np.float32(w_min),
        'b_scale': np.float32(b_scale),
        'b_min': np.float32(b_min)
    }
    dump(quant_params, 'quant_params.joblib', compress=3)
    quant_size = get_file_size_kb('quant_params.joblib')
    
    dequant_w = w_scale * quant_w.astype(np.float32) + w_min
    dequant_b = b_scale * quant_b.astype(np.float32) + b_min
    
    model = nn.Linear(weights.shape[0], 1)
    with torch.no_grad():
        model.weight.data = torch.from_numpy(dequant_w).float().unsqueeze(0)
        model.bias.data = torch.tensor(dequant_b).float()
    
    X, y = fetch_california_housing(return_X_y=True)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    sklearn_score = sklearn_model.score(X_test, y_test)
    
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()
    with torch.no_grad():
        predictions = model(X_test_tensor).squeeze()
        ss_res = torch.sum((y_test_tensor - predictions) ** 2)
        ss_tot = torch.sum((y_test_tensor - torch.mean(y_test_tensor)) ** 2)
        pytorch_score = 1 - (ss_res / ss_tot).item()
    
    reduction = (unquant_size - quant_size) / unquant_size * 100
    
    print("\n=== Quantization Results ===")
    print("| Metric               | Original Model | Quantized Model |")
    print("|----------------------|----------------|-----------------|")
    print(f"| R² Score            | {sklearn_score:.6f} | {pytorch_score:.6f} |")
    print(f"| Model Size (KB)     | {unquant_size:.3f} KB | {quant_size:.3f} KB |")
    print(f"| Size Reduction      | {reduction:.1f}%          |")

if __name__ == "__main__":
    main()