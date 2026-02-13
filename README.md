# 🌾 Rice Leaf Disease Detection

**Project Code:** PRCP-1001

A deep learning-based system for classifying rice leaf diseases using Convolutional Neural Networks (CNN) and Transfer Learning. This project helps farmers and agricultural experts detect diseases early to prevent crop loss.

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Disease Classes](#disease-classes)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Challenges and Solutions](#challenges-and-solutions)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

Rice is one of the most important food crops worldwide, feeding over half of the global population. Various leaf diseases can significantly reduce crop yield if not detected early. This project implements a machine learning solution to automatically classify rice leaf diseases from images, enabling:

- **Early disease detection** for timely intervention
- **Accurate classification** of disease types
- **Scalable deployment** for agricultural applications
- **Support for farmers** in disease management

## 🔍 Problem Statement

Rice crops are vulnerable to various diseases that can devastate yields. Manual disease detection is:
- Time-consuming and labor-intensive
- Requires expert knowledge
- Prone to human error
- Not scalable for large farms

**Our Solution:** Build an automated deep learning system that:
1. Analyzes rice leaf images
2. Classifies diseases with high accuracy
3. Provides quick and reliable predictions
4. Can be deployed on mobile devices for field use

## 📊 Dataset

The dataset contains **120 RGB images** distributed equally among three rice leaf disease classes.

### Dataset Distribution
- **Total Images:** 120
- **Classes:** 3
- **Images per class:** ~40
- **Image Format:** RGB (color images)
- **Dataset Split:**
  - Training: 80%
  - Validation: 20%

### Dataset Structure
```
Dataset/
├── Bacterial leaf blight/  (40 images)
├── Brown spot/             (40 images)
└── Leaf smut/              (39 images)
```

## 🦠 Disease Classes

### 1. Bacterial Leaf Blight
- **Pathogen:** *Xanthomonas oryzae*
- **Symptoms:** Water-soaked lesions that turn yellow to white
- **Impact:** Can cause up to 50% yield loss in severe cases

### 2. Brown Spot
- **Pathogen:** *Bipolaris oryzae*
- **Symptoms:** Circular or oval brown spots with gray centers
- **Impact:** Reduces grain quality and can cause 50-90% yield loss

### 3. Leaf Smut
- **Pathogen:** *Entyloma oryzae*
- **Symptoms:** Small black spots on leaves
- **Impact:** Affects photosynthesis and plant health

## 🤖 Models Implemented

### 1. Baseline CNN
A simple convolutional neural network without data augmentation.
- **Validation Accuracy:** 73.91%
- **Architecture:** 3 Conv layers + 2 Dense layers
- **Purpose:** Establish baseline performance

### 2. Improved CNN with Data Augmentation
Enhanced CNN with augmentation techniques to improve generalization.
- **Validation Accuracy:** 82.61%
- **Augmentation:** Rotation, flip, zoom, shift
- **Improvement:** +8.7% over baseline

### 3. MobileNetV2 (Transfer Learning) ⭐
Pre-trained model fine-tuned on rice disease dataset.
- **Validation Accuracy:** 91.30% 🏆
- **Architecture:** MobileNetV2 + Custom classifier
- **Advantage:** Leverages ImageNet pre-training

## 📈 Results

### Model Comparison

| Model | Validation Accuracy | Notes |
|-------|-------------------|-------|
| Baseline CNN | 73.91% | Simple architecture |
| Improved CNN + Augmentation | 82.61% | Better generalization |
| **MobileNetV2 (Final Model)** | **91.30%** | **Best performance** |

### Why MobileNetV2?

✅ **Highest accuracy** (91.3%)  
✅ **Robust generalization** to unseen data  
✅ **Efficient inference** suitable for mobile deployment  
✅ **Transfer learning** leverages ImageNet features  
✅ **Production-ready** for real-world applications  

## 🚀 Installation

### Prerequisites
- Python 3.9+
- TensorFlow 2.x
- CUDA (optional, for GPU acceleration)

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/rice-disease-detection.git
cd rice-disease-detection
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the dataset**
Place your dataset in the `Dataset/` directory following the structure shown above.

## 💻 Usage

### Training the Model

```python
# Run the Jupyter notebook
jupyter notebook Rice_app__2_.ipynb
```

Or execute the training script:

```python
python train.py --model mobilenetv2 --epochs 50 --batch_size 32
```

### Making Predictions

```python
from model import predict_disease

# Predict on a single image
result = predict_disease('path/to/leaf_image.jpg')
print(f"Predicted disease: {result['disease']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Model Evaluation

```python
python evaluate.py --model_path models/mobilenetv2_best.h5
```

## 📁 Project Structure

```
rice-disease-detection/
│
├── Dataset/                      # Training data
│   ├── Bacterial leaf blight/
│   ├── Brown spot/
│   └── Leaf smut/
│
├── models/                       # Saved models
│   ├── baseline_cnn.h5
│   ├── improved_cnn.h5
│   └── mobilenetv2_best.h5
│
├── notebooks/
│   └── Rice_app__2_.ipynb       # Main analysis notebook
│
├── src/
│   ├── data_preprocessing.py    # Data loading and augmentation
│   ├── model.py                 # Model architectures
│   ├── train.py                 # Training script
│   └── evaluate.py              # Evaluation utilities
│
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── LICENSE                      # License file
```

## 🛠️ Technologies Used

### Core Libraries
- **TensorFlow/Keras** - Deep learning framework
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **Matplotlib** - Visualization
- **PIL/Pillow** - Image processing

### Key Techniques
- **Convolutional Neural Networks (CNN)**
- **Transfer Learning** (MobileNetV2)
- **Data Augmentation**
- **Dropout Regularization**
- **Early Stopping**

## 🎯 Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| **Small Dataset** (120 images) | Data augmentation (rotation, flip, zoom) |
| **Overfitting** | Dropout layers + Early stopping |
| **Model Selection** | Systematic comparison of 3 architectures |
| **Limited Training Data** | Transfer learning from ImageNet |

## 🔮 Future Improvements

- [ ] Expand dataset to 1000+ images per class
- [ ] Add more disease classes
- [ ] Implement real-time detection using webcam
- [ ] Deploy as mobile app (Android/iOS)
- [ ] Create web interface for easy access
- [ ] Add explainability features (Grad-CAM)
- [ ] Multi-language support for farmers
- [ ] Integration with IoT sensors for automated monitoring

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Guidelines
- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Keep commits atomic and well-described

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset providers and agricultural research institutions
- TensorFlow and Keras teams for excellent deep learning tools
- The open-source community for inspiration and support
- Agricultural experts for domain knowledge

## 📧 Contact

For questions, suggestions, or collaborations:

- **Project Link:** [https://github.com/yourusername/rice-disease-detection](https://github.com/yourusername/rice-disease-detection)
- **Email:** your.email@example.com
- **LinkedIn:** [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)

---

**⭐ If you find this project useful, please consider giving it a star!**

**Made with ❤️ for sustainable agriculture**
