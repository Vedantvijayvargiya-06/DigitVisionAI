# 🔢 DigitVision AI – Handwritten Digit Recognition

<p align="center">
  <strong>Recognize handwritten digits instantly using Deep Learning.</strong>
</p>

<p align="center">
  <a href="https://digitvision-ai-project.streamlit.app/">
    <img src="https://img.shields.io/badge/Live%20Demo-Open%20App-brightgreen?style=for-the-badge&logo=streamlit" alt="Live Demo">
  </a>
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.18-orange?style=for-the-badge&logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-red?style=for-the-badge&logo=streamlit" alt="Streamlit">
</p>

---

## 🌐 Live Demo

**Try the application here:**

👉 https://digitvision-ai-project.streamlit.app/

Upload an image of a handwritten digit and let the CNN model predict the digit from 0 to 9 with a confidence score.

## 📌 About the Project

DigitVision AI is a deep learning-based handwritten digit recognition application built using a Convolutional Neural Network (CNN).

The model is trained on the MNIST dataset, which contains 60,000 training images and 10,000 testing images of handwritten digits.

Users can upload their own handwritten digit images and receive predictions along with confidence scores and probability distributions.

## ✨ Features

- 🔢 Handwritten digit recognition (0–9)
- 🧠 CNN model built using TensorFlow and Keras
- 📊 Prediction confidence and probability visualization
- 🖼️ Upload handwritten digit images (PNG, JPG, JPEG)
- 🎯 MNIST sample image demonstration
- 📈 Model performance evaluation
- 📉 Accuracy and loss visualization
- 📊 Confusion matrix and classification metrics
- 🌐 Interactive Streamlit web interface

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming |
| TensorFlow | Deep learning framework |
| Keras | CNN model development |
| Streamlit | Web application |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Model evaluation |
| Pillow (PIL) | Image processing |

## 🧠 CNN Architecture

The model uses the following architecture:

1. Input layer – 28 × 28 grayscale images
2. Convolutional layer – 32 filters
3. Max Pooling layer
4. Convolutional layer – 64 filters
5. Max Pooling layer
6. Dropout layer
7. Flatten layer
8. Dense layer – 128 neurons
9. Dropout layer
10. Output layer – 10 neurons with Softmax activation

### Training Configuration

- Dataset: MNIST
- Training images: 60,000
- Testing images: 10,000
- Image dimensions: 28 × 28
- Epochs: 8
- Batch size: 128
- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy

## 📊 Model Training

The CNN model is trained using the MNIST handwritten digit dataset.

During training, the application tracks:
- Training accuracy
- Validation accuracy
- Training loss
- Validation loss

Model performance can be explored through the application's Model Performance section.

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Vedantvijayvargiya-06/DigitVisionAI.git
cd DigitVisionAI
```

### 2. Create a virtual environment

```bash
py -3.12 -m venv .venv
```

### 3. Activate the environment (Windows)

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

The application will open in your browser at:

`http://localhost:8501`

## 📂 Project Structure

```text
DigitVisionAI/
│
├── app.py
├── requirements.txt
├── README.md
├── sample_digit_7.png
├── VedantVijay_CNN_Streamlit_Report.pdf
└── .gitignore
```

## 🖼️ Sample Input

The repository includes a sample handwritten digit image for testing.

![Sample handwritten digit 7](sample_digit_7.png)

**Expected prediction:** 7

## 📄 Project Report

Detailed project documentation and analysis:

[View CNN Streamlit Project Report](VedantVijay_CNN_Streamlit_Report.pdf)

## 👨‍💻 Author

**Vedant Vijay**

- GitHub: [@Vedantvijayvargiya-06](https://github.com/Vedantvijayvargiya-06)
- Project Repository: [DigitVisionAI](https://github.com/Vedantvijayvargiya-06/DigitVisionAI)
- Live Application: [DigitVision AI](https://digitvision-ai-project.streamlit.app/)

## ⭐ Support

If you find this project interesting, consider giving it a ⭐ on GitHub!

---

<p align="center">
  Built with ❤️ using Python, TensorFlow and Streamlit.
</p>
