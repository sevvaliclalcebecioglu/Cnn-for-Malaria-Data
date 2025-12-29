# Image Classification with CNN for Malaria Detection

🔗 **Live Demo (Hugging Face Space):**  
https://huggingface.co/spaces/sevvaliclal/MaleriaModel

---

## Project Overview
The objective of this project is to develop image classification models capable of detecting **malaria infection** from microscopic cell images.  
The study first implements a **custom Convolutional Neural Network (CNN)** and then compares its performance with **transfer learning models** such as **VGG16** and **ResNet50**, which are pre-trained on the ImageNet dataset.

---

## Dataset Description
The dataset contains **27,558 microscopic cell images**, evenly distributed across two classes:

- **Uninfected cells:** 13,780 images  
- **Parasitized (infected) cells:** 13,780 images  

All images are RGB images and resized according to the input requirements of the selected model architectures.

---

## Data Preprocessing and Augmentation
To enhance model generalization and reduce overfitting, the following preprocessing and augmentation techniques were applied:

- Pixel value normalization (rescaling to the range [0,1])
- Random rotation, zoom, shear, and horizontal flipping
- Dataset split into **70% training** and **30% validation**

The `ImageDataGenerator` class was used to perform real-time data augmentation during training.

---

## Custom CNN Model
A custom Convolutional Neural Network was implemented using TensorFlow and Keras with the following architecture:

- Input layer: 170×170 RGB images  
- Two convolutional layers (Conv2D) with ReLU activation  
- MaxPooling layers for spatial downsampling  
- Fully connected Dense layer with 128 neurons  
- Sigmoid output layer for binary classification  

### Custom CNN Results
- **Training Accuracy:** ~95%  
- **Validation Accuracy:** ~95%  
- Training and validation loss values decreased consistently

These results indicate that the custom CNN successfully learned discriminative features for malaria detection with minimal overfitting.

---

## Transfer Learning Approach
To further improve performance, a **VGG16** model pre-trained on ImageNet was used as a feature extractor:

- The convolutional base was frozen to preserve learned features
- A custom classification head was added:
  - Flatten layer  
  - Dense layer with 1024 neurons (ReLU)  
  - Sigmoid output layer  

### Transfer Learning Results (VGG16)
- **Training Accuracy:** ~96%  
- **Validation Accuracy:** ~92%  

The transfer learning model achieved strong performance with faster convergence, demonstrating the effectiveness of using pre-trained deep learning models for medical image classification tasks.

---

## Conclusion
Both the custom CNN and transfer learning approaches produced high classification accuracy for malaria detection.  
While the custom CNN achieved balanced training and validation performance, the VGG16-based transfer learning model provided competitive results with reduced training time.  
This project highlights the potential of deep learning techniques in supporting automated and reliable medical image diagnosis.
