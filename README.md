# CNN Character Recognition & CAPTCHA Solver

An end-to-end Convolutional Neural Network (CNN) trained to recognize handwritten/font characters with high precision. This project serves as a foundational backbone for solving multi-character CAPTCHAs.

## 🚀 Features
- **Deep Learning Architecture:** Utilizes a multi-layer CNN with Dropout and MaxPool for spatial feature extraction.
- **Massive Dataset:** Trained on the `character_font.npz` dataset containing ~390,000 labeled 32x32 images.
- **High Accuracy:** Optimized for 26-class alphabet classification (A-Z).
- **Modular Pipeline:** Separate scripts for training (`train.py`) and inference (`predict.py`) to minimize compute overhead.

## 🛠️ Tech Stack
- **Python 3.10+**
- **TensorFlow/Keras** (Neural Network Backend)
- **Scikit-Learn** (Data Stratification & Splitting)
- **Matplotlib** (Result Visualization)

## 📊 Model Architecture
1. **Conv2D (32 filters, 3x3):** Detects basic edges and textures.
2. **MaxPooling2D (2x2):** Reduces spatial dimensions while preserving key features.
3. **Conv2D (64 filters, 3x3):** Identifies complex shapes/curves.
4. **Flatten:** Converts 2D feature maps to 1D vectors.
5. **Dense (128 units, ReLU):** Fully connected "thinking" layer.
6. **Dense (26 units, Softmax):** Outputs probability distribution for A-Z.

## 📈 Results
The model achieves high confidence across all 26 character classes.

## 💻 Usage
1. **Prepare Data:** Ensure `character_font.npz` is in the data directory.
2. **Run Inference:**
   ```bash
   python predict.py