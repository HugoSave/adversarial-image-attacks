# Adversarial Examples in the Physical World - Reproduction

<img src="./images/adveserial_example_FGSM.png" alt="Project Banner" width="80%">


This repository contains a PyTorch-based partial reproduction and analysis of the paper:

**Adversarial Examples in the Physical World**  
*Alexey Kurakin, Ian Goodfellow, Samy Bengio* (2017)  
[Paper Link](https://arxiv.org/abs/1607.02533)

---

## 📖 Project Overview

Adversarial examples are carefully crafted inputs designed to fool machine learning models. This project reproduces key experiments from the paper, generating and evaluating adversarial perturbations on image classification models (ResNet, and VGG16) with the CIFAR-10 dataset.

The goals include:  
- Understanding the robustness of models to adversarial attacks  
- Comparing different attack methods (FGSM, Iterative FGSM)  
- Analyzing transferability of attacks across models  
- Visualizing original vs adversarial images  

---

## 🖼️ Illustrative Examples

<img src="./images/model_evaluation.png" alt="Project Banner" width="40%">


---


## 🚀 How to Run

### Google Colab (Recommended)

This project is fully compatible with Google Colab for easy setup and GPU acceleration.

1. Open the Colab notebook here:  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HugoSave/adversarial-image-attacks/blob/main/AdversarialAttacks.ipynb)

2. Follow the notebook cells step-by-step:  
   - Install dependencies  
   - Load models and the cifar-10 dataset 
   - Generate adversarial datasets  
   - Run evaluations and visualize results  

3. Feel free to tweak parameters interactively for your own experiments.

---


