import pandas as pd
import joblib

model = joblib.load('models/heart_model.pkl')
imputer = joblib.load('models/imputer.pkl')
scaler = joblib.load('models/scaler.pkl')
try:
    encoder = joblib.load('models/encoder.pkl')
except:
    encoder = None

novo_dado = pd.DataFrame([{
    'age': 55,
    'sex': 1,
    'cp': 2,
    'trestbps': 140,
    'chol': 230,
    'fbs': 0,
    'restecg': 1,
    'thalach': 150,
    'exang': 0,
    'oldpeak': 1.5,
    'slope': 2,
    'ca': 0,
    'thal': 1
}])

numeric_cols = novo_dado.select_dtypes(include='number').columns.tolist()
novo_dado[numeric_cols] = imputer.transform(novo_dado[numeric_cols])
novo_dado[numeric_cols] = scaler.transform(novo_dado[numeric_cols])

if encoder:
    categorical_cols = [c for c in novo_dado.columns if c not in numeric_cols]
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    novo_dado[encoded_cols] = encoder.transform(novo_dado[categorical_cols])
    X_final = novo_dado[numeric_cols + encoded_cols]
else:
    X_final = novo_dado

pred = model.predict(X_final)
prob = model.predict_proba(X_final)

print("Predição:", pred[0])
print("Probabilidade:", round(prob[0][1]*100,2), "%")
