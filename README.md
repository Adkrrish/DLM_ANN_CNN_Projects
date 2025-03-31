# ✍️ MNIST Handwritten Digit Recognizer using Deep CNN

## 📌 Contributors
- **Krishnendu Adhikary (055022)**
- **Mohit Agarwal (055024)**

## 🎯 Objective
This project develops a **Deep Convolutional Neural Network (CNN)** to classify handwritten digits from the **MNIST dataset**, improving recognition efficiency for applications like postal mail sorting and bank check processing.

## 🏆 Problem Statement
Traditional methods struggle with handwriting diversity, leading to errors. Our **CNN model**, based on the **LeNet-5 architecture**, enhances digit classification by leveraging deep learning techniques.

## 🔑 Project Structure
1. **Data Preparation**:
   - **Normalization**: Scaling pixel values to `[0,1]`
   - **Reshaping**: Converting to `(28,28,1)` for CNN input
   - **One-Hot Encoding**: Label encoding for classification
   - **Train-Test Split**: 80% training, 20% validation

2. **Model Building**:
   - **Architecture**: LeNet-5 (Convolutional & Pooling layers + Dropout)
   - **Data Augmentation**: Rotation, zooming, flipping to improve generalization
   - **Optimizer**: `RMSProp` for stable convergence
   - **Learning Rate Scheduler**: `ReduceLROnPlateau` to prevent stagnation

3. **Model Training**:
   - **Trained with GPU acceleration** on Kaggle for efficiency
   - **Monitored loss & accuracy** to prevent overfitting
   - **Final accuracy ~99%**, confirming strong performance

4. **Testing & Predictions**:
   - Model evaluated on test dataset
   - Predictions saved in CSV format for competition submission

## 📊 Key Insights
- The **LeNet-5 model** effectively captures digit patterns, achieving **high accuracy**.
- **Automation potential**: Useful for banking, postal services, and digital form processing.
- **Cost-effective**: Reduces manual transcription efforts & errors.
- **Scalability**: Can be adapted for multi-language handwriting recognition.
