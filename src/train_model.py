import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib

def load_dataset():
    n = 1000
    np.random.seed(42)
    df = pd.DataFrame({
        'Idade': np.random.randint(18, 60, n),
        'Sexo': np.random.choice(['Masculino', 'Feminino'], n),
        'IMC': np.random.uniform(18, 35, n).round(1),
        'TempoTreinoSemanal': np.random.randint(1, 8, n),
        'ConsumoProteinaDia': np.random.uniform(50, 200, n).round(1),
        'Sono': np.random.uniform(4, 9, n).round(1),
        'IngestaoAgua': np.random.uniform(1.5, 4.0, n).round(1),
        'CaloriasDiarias': np.random.randint(1500, 3500, n),
        'NivelExperiencia': np.random.choice(['Iniciante','Intermediário','Avançado'], n)
    })
    df['AtingiuMeta'] = np.where(
        (df['TempoTreinoSemanal'] >= 4) &
        (df['ConsumoProteinaDia'] >= 100) &
        (df['Sono'] >= 7) &
        (df['IngestaoAgua'] >= 2.5) &
        (df['CaloriasDiarias'] <= 2500),
        'Yes', 'No'
    )
    return df

def preprocess(df):
    X = df.drop('AtingiuMeta', axis=1)
    y = df['AtingiuMeta']

    #Dividir em treino, validação e teste
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
    
    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_train.select_dtypes(include='object').columns.tolist()
    
    #Imputação
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(X_train[numeric_cols])
    X_train[numeric_cols] = imputer.transform(X_train[numeric_cols])
    X_val[numeric_cols] = imputer.transform(X_val[numeric_cols])
    X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])
    
    #Escalonamento
    scaler = MinMaxScaler()
    scaler.fit(X_train[numeric_cols])
    X_train[numeric_cols] = scaler.transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    #Codificação
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(X_train[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    X_train[encoded_cols] = encoder.transform(X_train[categorical_cols])
    X_val[encoded_cols] = encoder.transform(X_val[categorical_cols])
    X_test[encoded_cols] = encoder.transform(X_test[categorical_cols])
    
    #Seleção final de colunas
    X_train_final = X_train[numeric_cols + encoded_cols]
    X_val_final = X_val[numeric_cols + encoded_cols]
    X_test_final = X_test[numeric_cols + encoded_cols]
    
    return X_train_final, y_train, X_val_final, y_val, X_test_final, y_test, imputer, scaler, encoder

def train_and_save():
    df = load_dataset()
    X_train, y_train, X_val, y_val, X_test, y_test, imputer, scaler, encoder = preprocess(df)
    
    model = DecisionTreeClassifier(max_depth=6, max_leaf_nodes=20, random_state=42)
    model.fit(X_train, y_train)
    
    print("Acurácia Treino:", model.score(X_train, y_train))
    print("Acurácia Validação:", model.score(X_val, y_val))
    print("Acurácia Teste:", model.score(X_test, y_test))
    
    #Criar pasta 'models' se não existir
    os.makedirs('models', exist_ok=True)

    #Salvar modelo e objetos de pré-processamento
    joblib.dump(model, 'models/body_model.pkl')
    joblib.dump(imputer, 'models/imputer.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(encoder, 'models/encoder.pkl')

    print("Modelo e objetos salvos em 'models/' com sucesso!")

if __name__ == "__main__":
    train_and_save()
