import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Define ANN model
class ANN(nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_units):
        super(ANN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_units))
        layers.append(nn.ReLU())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_units, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Streamlit UI
def main():
    st.title("ANN-Based Prediction Dashboard")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df.head())
        
        # Feature Selection
        target_column = st.selectbox("Select Target Column", df.columns)
        feature_columns = [col for col in df.columns if col != target_column]
        
        # Data Preprocessing
        X = df[feature_columns]
        y = df[target_column]
        
        # Encoding categorical variables
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col])
        
        # Scaling numerical features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = LabelEncoder().fit_transform(y)  # Ensure binary output (0 or 1)
        
        # Splitting dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
        y_train, y_test = torch.tensor(y_train, dtype=torch.float32).view(-1, 1), torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        
        # Hyperparameter Tuning
        hidden_layers = st.slider("Hidden Layers", 1, 5, 2)
        hidden_units = st.slider("Neurons per Layer", 5, 100, 10)
        learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, step=0.001)
        epochs = st.slider("Epochs", 10, 200, 50)
        
        # Model Training
        model = ANN(X_train.shape[1], hidden_layers, hidden_units)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train model
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
        
        # Evaluate model
        with torch.no_grad():
            predictions = model(X_test)
            accuracy = ((predictions.round() == y_test).float().mean()).item()
        
        st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
        
if __name__ == "__main__":
    main()