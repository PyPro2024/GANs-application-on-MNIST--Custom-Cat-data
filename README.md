# Generative Adversarial Networks (GANs) with PyTorch

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-GANs-blue)

## Project Overview
This repository contains implementations of **Generative Adversarial Networks (GANs)** built from scratch using **PyTorch**. The project explores the capability of neural networks to generate synthetic data that mimics real-world distributions.

The project is divided into two major phases:
1.  **Vanilla GAN & DCGAN on MNIST:** Generating handwritten digits.
2.  **Custom DCGAN on Private Dataset:** Generating 64x64 color images (Cats) using a custom-annotated dataset.

##  Architectures Implemented

### 1. MNIST Generator (Digits)
* **Evolution:** Started with a simple Fully Connected (Dense) GAN.
* **Optimization:** Upgraded to a **Deep Convolutional GAN (DCGAN)** to reduce image noise and improve structural coherence.
* **Hyperparameters:** Tuned the Adam optimizer and increased training from 50 to 100 epochs for stability.

### 2. Custom DCGAN (Object Generation)
* **Dataset:** Custom dataset of cat images.
* **Architecture:**
    * **Generator:** Uses Transposed Convolutions to upsample latent vectors into 64x64x3 RGB images. Features a deep network with 512 filters to capture color and texture details.
    * **Discriminator:** A Convolutional Classifier that distinguishes real images from generated fakes.
* **Preprocessing:** Robust data augmentation (Resize, Normalize) applied to prevent overfitting on the small custom dataset.

## 🛠️ Tech Stack
* **Framework:** PyTorch
* **Data Processing:** Torchvision, PIL, NumPy
* **Visualization:** Matplotlib

## Key Results
* **MNIST:** Successfully transitioned from noisy gray outputs to sharp digit generation by replacing Dense layers with Convolutional layers.
* **Custom Data:** Stabilized training on a small dataset by optimizing the learning rate and using robust augmentation techniques.

##  How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/PyPro2024/GANs-application-on-MNIST--Custom-Cat-data.git]
    ```
2.  **Install dependencies:**
    ```bash
    pip install torch torchvision matplotlib numpy
    ```
3.  **Run the Notebook:**
    Open `GANs.ipynb` in Jupyter Notebook or Google Colab.
    * *Note: For the custom dataset section, ensure you update the image path to your local directory or mount your Google Drive.*

---
*If you find this project helpful, feel free to ⭐ the repo!*
