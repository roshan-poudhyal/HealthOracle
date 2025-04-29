import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

def build_model(input_dim):
    model = Sequential()
    model.add(Dense(16, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Heart Disease Prediction Model
def train_heart_model():
    # Load data from UCI ML Repository
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", 
        "oldpeak", "slope", "ca", "thal", "target"
    ]
    data = pd.read_csv(url, names=column_names, na_values="?")
    
    # Handle missing values
    data = data.dropna()
    
    # Prepare features and target
    X = data.drop('target', axis=1).values
    y = (data['target'] > 0).astype(int).values  # Convert to binary classification
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    joblib.dump(scaler, 'heart_scaler.pkl')
    
    # Build and train model
    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('heart_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    return model

# Lung Disease Prediction Model
def train_lung_model():
    # Load and preprocess data from CSV file
    df = pd.read_csv('HealthOracle/lung_cancer_examples.csv')
    
    # Prepare features and target
    X = df[['Age', 'Smokes', 'AreaQ', 'Alkhol']].values
    y = df['Result'].values
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    joblib.dump(scaler, 'lung_scaler.pkl')
    
    # Build and train model
    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('lung_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    return model

# Liver Disease Prediction Model
def train_liver_model():
    # Load and preprocess data from CSV file
    df = pd.read_csv('HealthOracle/indian_liver_patient.csv')
    
    # Convert Gender to numeric (Female: 0, Male: 1)
    df['Gender'] = (df['Gender'] == 'Male').astype(int)
    
    # Prepare features and target
    X = df[['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
            'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens',
            'Albumin', 'Albumin_and_Globulin_Ratio']].values
    y = (df['Dataset'] == 1).astype(int).values  # 1 for liver disease, 2 for no liver disease in dataset
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    joblib.dump(scaler, 'liver_scaler.pkl')
    
    # Build and train model
    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('liver_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    return model

# Diabetes Prediction Model
def train_diabetes_model():
    # Load and preprocess data from Pima Indians Diabetes dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    data = pd.read_csv(url, names=column_names)
    
    # Prepare features and target
    X = data.drop('Outcome', axis=1).values
    y = data['Outcome'].values
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    joblib.dump(scaler, 'diabetes_scaler.pkl')
    
    # Build and train model
    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('diabetes_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    return model

# Initialize all models and scalers
heart_model = train_heart_model()
lung_model = train_lung_model()
liver_model = train_liver_model()
diabetes_model = train_diabetes_model()

# Load scalers
heart_scaler = joblib.load('heart_scaler.pkl')
lung_scaler = joblib.load('lung_scaler.pkl')
liver_scaler = joblib.load('liver_scaler.pkl')
diabetes_scaler = joblib.load('diabetes_scaler.pkl')

# Initialize TFLite interpreters
def load_tflite_interpreter(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

heart_interpreter = load_tflite_interpreter('heart_model.tflite')
lung_interpreter = load_tflite_interpreter('lung_model.tflite')
liver_interpreter = load_tflite_interpreter('liver_model.tflite')
diabetes_interpreter = load_tflite_interpreter('diabetes_model.tflite')

# Prediction functions using TFLite
def predict_with_tflite(interpreter, features):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], features.astype(np.float32))
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return bool(prediction[0][0] > 0.5)

def predict_heart_disease(features):
    scaled_features = heart_scaler.transform([features])
    return predict_with_tflite(heart_interpreter, scaled_features)

def predict_lung_disease(features):
    scaled_features = lung_scaler.transform([features])
    return predict_with_tflite(lung_interpreter, scaled_features)

def predict_liver_disease(features):
    scaled_features = liver_scaler.transform([features])
    return predict_with_tflite(liver_interpreter, scaled_features)

def predict_diabetes(features):
    scaled_features = diabetes_scaler.transform([features])
    return predict_with_tflite(diabetes_interpreter, scaled_features)

