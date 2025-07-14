# DeepLearningCA1

# COVID-19 CT Scan Classifier (Julia)
This project replicates the core contributions of the research paper **“Deep Convolutional Neural Network–Based Computer-Aided Detection System for COVID-19 Using Multiple Lung Scans”** by *Ghaderzadeh et al. (2021)*, implemented entirely in the Julia programming language using open-source tools. The goal is to build a computer-aided detection (CAD) system that classifies CT scan images into **COVID-19** or **Non-COVID-19** categories using deep convolutional neural networks.


## Project Summary

- **Paper Source**: Ghaderzadeh et al., *JMIR Med Inform 2021* ([Link](https://doi.org/10.2196/25075))
- **Models Used**: ResNet50, DenseNet201
- **Data**: 100 CT scan images (50 COVID, 50 Non-COVID)
- **Frameworks**: Flux.jl, Metalhead.jl
- **Julia Version**: 1.x

## Introduction
This project is a re-implementation of the paper _"Deep Convolutional Neural Network–Based Computer-Aided Detection System for COVID-19 Using Multiple Lung Scans"_ by Ghaderzadeh et al. (2021), using the Julia programming language and open-source libraries like Flux.jl and Metalhead.jl.

## Objective

To build a deep learning-based computer-aided detection (CAD) system that classifies lung CT scans as either **COVID-19** or **Non-COVID-19** using pretrained CNN models in Julia.

## Features

-  Pretrained CNNs using Metalhead.jl (ResNet50 & DenseNet201)
-  Real-time image augmentation: flip, rotate, zoom
-  Grayscale to RGB image conversion
-  Cosine annealing learning rate scheduler
-  GPU acceleration via CUDA (if available)
-  Custom DataLoader for efficient batch training

## Evaluation

Both DenseNet201 and ResNet50 models achieved 100% validation accuracy on a small dataset of 100 CT images (50 COVID, 50 Non-COVID). While this confirms that the implementation and training pipeline are functioning correctly, the result clearly indicates overfitting due to the deep models and limited data. A confusion matrix showed perfect classification, but these results are not generalizable and must be interpreted cautiously.

## Conclusion

This project successfully replicates the key methodology of a COVID-19 CAD system in Julia using Flux.jl and Metalhead.jl. Despite data limitations, the pipeline—from preprocessing to model training—worked effectively, proving Julia's viability for deep learning in medical imaging. With access to larger datasets and further tuning, this system could be extended into a clinically relevant tool.


