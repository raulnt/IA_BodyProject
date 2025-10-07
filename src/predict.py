import pandas as pd
import numpy as np
import joblib

#CARREGAR
def load_artifacts():
    model = joblib.load('models/body_model.pkl')
    imputer = joblib.load('models/imputer.pkl')
    scaler = joblib.load('models/scaler.pkl')
    encoder = joblib.load('models/encoder.pkl')
    return model, imputer, scaler, encoder

#PREVISÃO
def predict_single(data: dict):
    model, imputer, scaler, encoder = load_artifacts()

    df = pd.DataFrame([data])
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    
    df[numeric_cols] = imputer.transform(df[numeric_cols])
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    df_encoded = pd.DataFrame(encoder.transform(df[categorical_cols]), columns=encoded_cols)
    
    X = pd.concat([df[numeric_cols], df_encoded], axis=1)
    
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][list(model.classes_).index(pred)]
    
    return pred, proba

#TESTE
if __name__ == "__main__":
    sample = {
        'Idade': 27,
        'Sexo': 'Masculino',
        'IMC': 25.5,
        'TempoTreinoSemanal': 5,
        'ConsumoProteinaDia': 120,
        'Sono': 7.5,
        'IngestaoAgua': 3.0,
        'CaloriasDiarias': 2200,
        'NivelExperiencia': 'Intermediário'
    }
    resultado, prob = predict_single(sample)
    print("Predição:", resultado)
    print(f"Probabilidade: {prob*100:.2f}%")

    novo_aluno = {
    'Idade': 22,
    'Sexo': 'Masculino',
    'IMC': 23.5,
    'TempoTreinoSemanal': 4,
    'ConsumoProteinaDia': 110,
    'Sono': 7.0,
    'IngestaoAgua': 2.5,
    'CaloriasDiarias': 2400,
    'NivelExperiencia': 'Iniciante'
    }
    resultado, probabilidade = predict_single(novo_aluno)
    print("Predição:", resultado)
    print(f"Probabilidade: {probabilidade*100:.2f}%")
