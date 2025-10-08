import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report, accuracy_score, roc_curve, auc
import joblib
from scipy.stats import randint

def load_dataset():
    df = pd.read_csv("data/heart.csv")  
    print("Primeiras linhas do dataset:\n", df.head())
    return df

def preprocess(df):
    X = df.drop('target', axis=1)
    y = df['target']

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
    
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    imputer.fit(X_train[numeric_cols])
    scaler.fit(X_train[numeric_cols])
    encoder.fit(X_train[categorical_cols])

    #Transformar
    def transform(X):
        X_num = imputer.transform(X[numeric_cols])
        X_num = scaler.transform(X_num)
        X_cat = encoder.transform(X[categorical_cols])
        X_final = np.hstack([X_num, X_cat])
        return X_final

    return transform(X_train), y_train, transform(X_val), y_val, transform(X_test), y_test, imputer, scaler, encoder


#Vanilla Mod
def train_crude_model(X_train, y_train, X_val, y_val, X_test, y_test):
    model = DecisionTreeClassifier(max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test, average='weighted')

    print("\n--- Modelo Cru ---")
    print("Acurácia Teste:", acc)
    print("F1-score Teste:", f1)
    print("Relatório completo:\n", classification_report(y_test, y_pred_test))

    return model, acc, f1, y_test, y_pred_proba


#OP Mod

def train_optimized_model(X_train, y_train, X_val, y_val, X_test, y_test):
    param_dist = {
        "max_depth": randint(3, 10),
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 5),
        "max_leaf_nodes": randint(10, 50)
    }
    
    dt = DecisionTreeClassifier(random_state=42)
    
    random_search = RandomizedSearchCV(
        dt, param_distributions=param_dist,
        n_iter=20, scoring='f1_weighted',
        cv=5, random_state=42, n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    y_pred_test = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test, average='weighted')

    print("\n--- Modelo Otimizado ---")
    print("Melhores parâmetros:", random_search.best_params_)
    print("Acurácia Teste:", acc)
    print("F1-score Teste:", f1)
    print("Relatório completo:\n", classification_report(y_test, y_pred_test))

    return best_model, acc, f1, y_test, y_pred_proba

def plot_comparison(acc_cru, f1_cru, acc_opt, f1_opt):
    labels = ['Acurácia', 'F1-Score']
    cru = [acc_cru, f1_cru]
    opt = [acc_opt, f1_opt]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(x - width/2, cru, width, label='Modelo Cru', color='lightcoral')
    ax.bar(x + width/2, opt, width, label='Modelo Otimizado', color='seagreen')

    ax.set_ylabel('Pontuação')
    ax.set_title('Comparação entre Modelos')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_roc(y_test, proba_cru, proba_opt):
    fpr_cru, tpr_cru, _ = roc_curve(y_test, proba_cru)
    fpr_opt, tpr_opt, _ = roc_curve(y_test, proba_opt)

    auc_cru = auc(fpr_cru, tpr_cru)
    auc_opt = auc(fpr_opt, tpr_opt)

    plt.figure(figsize=(6,4))
    plt.plot(fpr_cru, tpr_cru, color='red', lw=2, label=f'Modelo Cru (AUC = {auc_cru:.2f})')
    plt.plot(fpr_opt, tpr_opt, color='green', lw=2, label=f'Modelo Otimizado (AUC = {auc_opt:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('Falsos Positivos')
    plt.ylabel('Verdadeiros Positivos')
    plt.title('Curva ROC - Comparação de Modelos')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def train_and_save():
    df = load_dataset()
    X_train, y_train, X_val, y_val, X_test, y_test, imputer, scaler, encoder = preprocess(df)
    
    #Treino
    crude_model, acc_cru, f1_cru, y_t_cru, proba_cru = train_crude_model(X_train, y_train, X_val, y_val, X_test, y_test)
    best_model, acc_opt, f1_opt, y_t_opt, proba_opt = train_optimized_model(X_train, y_train, X_val, y_val, X_test, y_test)

    plot_comparison(acc_cru, f1_cru, acc_opt, f1_opt)
    plot_roc(y_t_cru, proba_cru, proba_opt)

    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/heart_model_best.pkl')
    joblib.dump(imputer, 'models/imputer.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(encoder, 'models/encoder.pkl')
    print("\nModelos e objetos salvos com sucesso!")

if __name__ == "__main__":
    train_and_save()
