A deep learning approach for platelet classification using CNNs and GAN-based data augmentation. This repository includes dataset preprocessing, synthetic data generation, and transfer learning models like DenseNet, Inception, and VGG. WGAN-GP is used to enhance dataset diversity, improving platelet detection and classification accuracy.


# Platelet Detection and Classification Using CNN & GAN-Based Augmentation

## 📌 Overview
This project explores **Convolutional Neural Networks (CNNs)** and **Generative Adversarial Networks (GANs)** for platelet detection and classification. The study enhances dataset availability through **synthetic data generation using GANs** and traditional augmentation techniques, improving deep learning model performance for medical imaging.

## 📂 Project Structure
- **`Platelet Detection (Original) and (GAN_Data) Dataset 100epochs_FN.ipynb`** - Jupyter Notebook containing platelet classification models.
- **`Platelet Detection (Augmentation) Level 1 and Level 2 Dataset 100epochs_FN.ipynb`** - Notebook for augmented dataset training.
- **`AUGMENTATION to over 1400 Images and GAN VALIDATION CODE.ipynb`** - Notebook detailing dataset augmentation and GAN-generated validation.
- **`Platelet_Gen.pdf`** - A scientific report covering platelet classification methodology, GAN-based augmentation, and model comparisons.
- **`figures/`** - Directory containing training accuracy/loss plots and confusion matrices.

## 🛠️ Technologies Used
- **Python (TensorFlow, Keras, OpenCV, NumPy, Matplotlib, Seaborn)**
- **Jupyter Notebook**
- **Deep Learning Models:**
  - Convolutional Neural Networks (CNN)
  - Wasserstein GAN with Gradient Penalty (WGAN-GP)
  - Transfer learning architectures (DenseNet, Inception, VGG)
- **Data Augmentation:**
  - Traditional augmentation (flipping, rotation, scaling)
  - GAN-based synthetic data generation
- **Evaluation Metrics:**
  - Accuracy, Precision, Recall, F1-score
  - Fréchet Inception Distance (FID) and Inception Score (IS) for GAN validation

## 🔹 Key Features
### **1. Dataset Processing & Augmentation**
- **Original dataset:** Limited to 71 platelet images across three classes.
- **Level 1 augmentation:** Increased dataset to **141 images** using traditional techniques.
- **Level 2 augmentation:** Expanded dataset to **1,463 images** with advanced augmentation.
- **GAN augmentation:** **300 synthetic images generated using WGAN-GP**.

### **2. CNN-Based Platelet Classification**
- **Custom CNN architectures** developed for platelet classification.
- **Pre-trained models** (DenseNet121, DenseNet169, DenseNet201, VGG16, VGG19, InceptionV3, InceptionResNetV2, AlexNet).
- Fine-tuning of **hyperparameters (batch size, learning rate, dropout rate)** for optimal performance.

### **3. Generative Adversarial Networks (GANs) for Data Generation**
- **WGAN-GP model** trained over **5000 epochs per class**.
- **GAN loss metrics** computed to validate synthetic images.
- **Comparison of real vs synthetic images** to assess data generation effectiveness.

### **4. Model Evaluation & Performance Metrics**
- **Best-performing model:** **DenseNet201 with 98% accuracy** on Level 2 Augmented Dataset.
- **GAN-augmented dataset** improved model accuracy and robustness.
- **Confusion matrices and training curves** analyzed for model effectiveness.

## 📊 Results & Insights
- **Original dataset results:** Limited accuracy due to dataset scarcity.
- **Augmented dataset results:** Significant improvement in classification accuracy.
- **GAN-generated dataset results:** Improved data diversity, enhancing model performance.
- **Comparison of traditional augmentation vs. GAN augmentation.**

## 🚀 Getting Started
### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/iabidoye/Platelet_Detection.git
   cd Platelet_Detection
   
2. Install required dependencies:
   ```bash
   pip install tensorflow keras opencv-python numpy matplotlib seaborn

3. Open the Jupyter Notebook and run the analysis:
   ```bash
   jupyter notebook "Platelet Detection (Original) and (GAN_Data) Dataset 100epochs_FN.ipynb"
   
🤝 Contribution
Contributions are welcome! If you have suggestions for model improvements or dataset expansions, feel free to submit a pull request.

📧 Contact
For inquiries or collaborations, please reach out.
