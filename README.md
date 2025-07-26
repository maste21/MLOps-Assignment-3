# MLOps-Assignment-3

This project implements an end-to-end MLOps pipeline for California Housing price prediction using:

     Scikit-learn Linear Regression
     PyTorch Neural Network
     Docker containerization
     GitHub Actions 
     Manual model quantization

### Clone locally:

     git clone https://github.com/maste21/MLOps-Assignment-3.git

     cd MLOps-Assignment-3

 Install requirements:
   
     pip install -r requirements.txt

### Quantization Results:     
Run the below command in quantization branch

    python quantize.py

=== Quantization Results ===
| Metric               | Original Model | Quantized Model |
|----------------------|----------------|-----------------|
| R² Score            | 0.575788 | -0.179779 |
| Model Size (KB)     | 0.288 KB | 0.346 KB |

 Size Reduction ->  -20.1%               

# Branching Strategy

     main: Initial setup and documentation
     dev: Model development and training
     docker_ci: Docker containerization and pipeline
     quantization: Model quantization and optimization

