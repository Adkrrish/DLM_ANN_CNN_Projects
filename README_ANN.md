# 📌 Heart Disease Prediction Using Artificial Neural Networks (ANN)
### 🔗 For the Streamlit Dashboard Link: https://dlmprojects-awzttghmyh6xdcrs75xuzq.streamlit.app/

## 🏆 Project Details
This project aims to **predict heart disease** using a deep learning model built with **Artificial Neural Networks (ANN)**. The model is trained on a structured dataset containing patient health records and cardiovascular attributes. The primary focus is on **hyperparameter tuning** to enhance model performance. The dataset is provided in this repository.("Dataset of Heart.csv")

## 👥 Contributors
**Krishnendu Adhikary** - 055022

**Mohit Agarwal** - 055024

---
## 🔑 Key Activities
- **Data Preprocessing:** Handling missing values, encoding categorical variables, and normalizing numerical attributes.
- **Model Development:** Constructing an ANN with tunable hyperparameters for optimization.
- **Hyperparameter Tuning:** Investigating the impact of different hyperparameter values on accuracy.
- **Visualization & Insights:** Analyzing model performance using loss and accuracy plots.
- **Managerial Interpretation:** Extracting actionable insights for healthcare decision-making.

---
## 💻 Technologies Used
- **Python**
- **TensorFlow & Keras** (for ANN modeling)
- **Pandas & NumPy** (for data manipulation)
- **Matplotlib & Seaborn** (for visualization)
- **Scikit-learn** (for preprocessing and evaluation)
- **Streamlit** (for interactive dashboard development)

---
## 📊 Nature of Data
The dataset consists of **structured clinical data** with a mix of **categorical and numerical variables** representing patients' medical history and test results.

---
## 📌 Variable Information
| Feature | Type | Description |
|---------|------|-------------|
| **Age** | Continuous | Age of the patient (years) |
| **Sex** | Categorical | Gender (Male/Female) |
| **CP (Chest Pain Type)** | Categorical | Chest pain type ('Asymptomatic', 'Atypical Angina', etc.) |
| **TRTBPS (Resting Blood Pressure)** | Continuous | Blood pressure (mm Hg) at hospital admission |
| **Chol (Serum Cholesterol)** | Continuous | Serum cholesterol level (mg/dl) |
| **FBS (Fasting Blood Sugar)** | Binary | 1 if fasting blood sugar > 120 mg/dl, else 0 |
| **Rest ECG (Resting Electrocardiographic Results)** | Categorical | Normal/ST Elevation/Other |
| **Thalachh (Max Heart Rate Achieved)** | Continuous | Maximum heart rate achieved |
| **Exng (Exercise Induced Angina)** | Binary | 1 if angina occurs during exercise, else 0 |
| **Oldpeak (ST Depression Induced by Exercise)** | Continuous | ST depression caused by exercise |
| **Slope (Slope of Peak Exercise ST Segment)** | Categorical | 'Flat', 'Up Sloping', etc. |
| **CA (Number of Major Vessels Colored by Fluoroscopy)** | Continuous | Ranges from 0 to 3 |
| **Thall (Thalassemia Type)** | Categorical | Different types of thalassemia |
| **Target** | Binary | 1 = Heart disease present, 0 = No heart disease |

---
## 🎯 Problem Statements
- How can **ANN models** effectively classify patients at risk of heart disease?
- What impact does **hyperparameter tuning** have on model accuracy?
- Can the model achieve high accuracy without **overfitting**?

---
## 🏗️ Model Information
- **Input Layer:** Accepts all **numerical and encoded categorical features**.
- **Hidden Layers:** Number of layers and neurons **customizable via Streamlit UI**.
- **Activation Functions:** **ReLU, Tanh, Sigmoid** (chosen dynamically by the user).
- **Dropout Rate:** Adjustable to **prevent overfitting**.
- **Optimizer Options:** **Adam, SGD, RMSprop**.
- **Loss Function Options:** **Binary Cross-Entropy, Mean Squared Error, Hinge Loss**.
- **Output Layer:** Single neuron with **Sigmoid activation** for binary classification.

---
## 📉 Observations from Hyperparameter Tuning
### 1️⃣ Number of Hidden Layers
- **1-2 layers:** Moderate accuracy (~85%).
- **3-4 layers:** Optimal accuracy (~91%) without overfitting.
- **5+ layers:** Slight improvement but increased **computational cost**.

### 2️⃣ Neurons per Layer
- **10-50 neurons:** Stable and consistent training.
- **>50 neurons:** Marginal improvement, but risk of **overfitting** increases.

### 3️⃣ Activation Functions
- **ReLU:** Performs best in hidden layers.
- **Sigmoid:** Used in the output layer for binary classification.
- **Tanh:** Works well but slightly inferior to ReLU.

### 4️⃣ Optimizer Comparison
- **Adam:** **Best performance**, balances speed and accuracy.
- **SGD:** Slower, requires more epochs for convergence.
- **RMSprop:** Works well but sometimes unstable.

### 5️⃣ Dropout Rate
- **0-0.2:** **Best accuracy (~91%)**.
- **0.3-0.5:** Reduces overfitting but may hurt performance.

### 6️⃣ Epochs
- **50 epochs:** Sufficient for convergence.
- **100+ epochs:** No significant improvement, risk of **overfitting**.

---
## 📈 Managerial Insights
### 🔹 Healthcare Applications
- This ANN model can help **doctors and healthcare professionals** make **early diagnoses** of heart disease.
- **Automating heart disease detection** reduces **human error** and speeds up treatment decisions.

### 🔹 Business Value
- Hospitals and insurance companies can use **AI-driven risk assessment** for better patient outcomes.
- Implementing **ANN-based diagnostic tools** can reduce **costs and workloads** for cardiologists.

### 🔹 Future Improvements
- **Feature Engineering:** Extracting more relevant medical features for better accuracy.
- **Hybrid Models:** Combining ANN with **other machine learning techniques** (e.g., XGBoost, Random Forest).
- **Explainability:** Using techniques like **SHAP values** to understand feature importance in predictions.

---
## 🚀 Conclusion
This project successfully demonstrates how **deep learning** can be leveraged for **heart disease prediction**. The ANN model achieves **~91% accuracy** after hyperparameter tuning, making it a **powerful tool for healthcare analytics**. 

---
## 📜 License
This project is open-source and available for further improvement. Feel free to contribute!

---
## 🤝 Contributions
Suggestions, improvements, and forks are welcome! Open a pull request or raise an issue. 💡
