# 👤 Sprint 18 Project — Customer Age Estimation with Computer Vision (ResNet50)

---

## 🧠 Project Overview

In this project, I worked as a Data Scientist for a retail company interested in leveraging **Computer Vision** to estimate customer age from facial images.

The objective was to build a deep learning model capable of predicting a person's age using photographs. Accurate age estimation can support customer analytics, demographic studies, and marketing strategies without requiring customers to provide personal information directly.

A **ResNet50 Convolutional Neural Network (CNN)** pre-trained on ImageNet was used through transfer learning to perform age prediction.

---

## 🎯 Project Objectives

* Load and explore the customer image dataset.
* Analyze age distribution and image characteristics.
* Preprocess image data for deep learning.
* Build and train a ResNet50-based regression model.
* Evaluate model performance using Mean Absolute Error (MAE).
* Assess whether computer vision can solve the client's business problem.
* Identify additional practical applications of the model.

---

## 📁 Dataset Description

The dataset contains facial photographs and corresponding age labels.

### Features

| Feature   | Description           |
| --------- | --------------------- |
| file_name | Image filename        |
| real_age  | Customer age in years |

### Target Variable

| Variable | Description         |
| -------- | ------------------- |
| real_age | Age to be predicted |

### Dataset Size

* Approximately **7,600 images**
* Images of individuals across multiple age groups
* Real-world variability in lighting, pose, and image quality

---

## 🧩 Project Workflow

### Step 1 — Data Loading

* Loaded image labels from `labels.csv`.
* Linked image files with corresponding age labels.
* Verified dataset integrity and image availability.

---

### Step 2 — Exploratory Data Analysis (EDA)

Analyzed:

* Age distribution.
* Dataset balance.
* Image quality.
* Demographic representation.

### Key Findings

* Most observations correspond to young and middle-aged adults.
* Fewer examples exist for children and elderly individuals.
* Images present variability in lighting conditions and facial orientation.
* Age distribution is not perfectly balanced.

---

### Step 3 — Data Preprocessing

Image preparation included:

* Image resizing to 224 × 224 pixels.
* Pixel normalization using:

```python
rescale = 1./255
```

* Training-validation split using `ImageDataGenerator`.
* Horizontal image augmentation during training.

---

## 🤖 Model Development

### Architecture

The model was built using:

* ResNet50 (pre-trained on ImageNet)
* Global Average Pooling Layer
* Dense Output Layer

### Training Configuration

* Optimizer: Adam
* Loss Function: Mean Squared Error (MSE)
* Metric: Mean Absolute Error (MAE)
* Epochs: 20
* Transfer Learning with ResNet50

---

## 📊 Model Performance

### Training Results

| Epoch | Training MAE | Validation MAE |
| ----- | ------------ | -------------- |
| 1     | 8.8          | 8.1            |
| 5     | 6.7          | 6.3            |
| 10    | 5.9          | 5.7            |
| 20    | 5.2          | 5.5            |

### Final Result

✅ **Validation MAE ≈ 5.5 years**

✅ **Project Requirement Met (MAE < 8)**

The model achieved strong predictive performance and demonstrated stable convergence throughout training.

---

## 📈 Evaluation Metric

### Mean Absolute Error (MAE)

The primary metric used for evaluation.

```text
MAE = Σ |y_true − y_pred| / n
```

An MAE of approximately **5.5 years** means that, on average, the model's age predictions differ from the actual age by about 5.5 years.

---

## 🔍 Model Analysis

The ResNet50 model produced accurate age estimates while maintaining stable validation performance.

Key observations:

* No significant signs of overfitting.
* Consistent reduction in validation error during training.
* Strong generalization capability on unseen data.

The model performs best for age groups that are well represented in the training data.

---

## 💡 Business Applications

### Can Computer Vision Help the Client?

Yes.

The model can provide useful age estimates for customer analytics and segmentation purposes without requiring manual demographic data collection.

However, age estimation should be considered approximate and should not be used for legal verification or high-stakes decision-making.

---

### Additional Practical Applications

#### Customer Segmentation

* Analyze customer age groups.
* Improve marketing targeting.

#### Personalized Advertising

* Deliver age-relevant promotions.
* Improve campaign effectiveness.

#### Retail Analytics

* Study demographic trends.
* Optimize product placement.

#### Recommendation Systems

* Tailor recommendations based on estimated age.

#### Store Performance Monitoring

* Compare customer demographics across locations.

---

## ⚠️ Limitations

* Imbalanced age distribution.
* Fewer samples at age extremes.
* Sensitivity to image quality and lighting conditions.
* Age estimation remains an approximation.

---

## 🚀 Potential Improvements

* Fine-tune deeper layers of ResNet50.
* Increase training data for underrepresented age groups.
* Apply additional image augmentation techniques.
* Experiment with EfficientNet or Vision Transformers (ViT).

---

## 🧰 Tools & Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* TensorFlow
* Keras
* ResNet50
* ImageDataGenerator
* Jupyter Notebook

---

## 📚 Machine Learning Concepts Applied

* Computer Vision
* Deep Learning
* Convolutional Neural Networks (CNNs)
* Transfer Learning
* ResNet50
* Regression Modeling
* Image Preprocessing
* TensorFlow/Keras
* MAE Optimization

---

## 👤 Author

**Jonathan Peña**
