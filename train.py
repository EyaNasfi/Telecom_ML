das as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib 

def prepare_data(file_path_80, file_path_20):
    """
    Charge et prétraite les données à partir de deux fichiers CSV.
    
    Args:
        file_path_80 (str): Chemin vers le fichier churn-bigml-80.csv
        file_path_20 (str): Chemin vers le fichier churn-bigml-20.csv
    
    Returns:
        X_scaled (array): Features mises à l'échelle
        y (array): Variable cible
        feature_names (list): Noms des features
    """
    # Charger les données
    data1 = pd.read_csv(file_path_80)
    data2 = pd.read_csv(file_path_20)
    data = pd.concat([data1, data2], ignore_index=True)
    
    # Vérifier les doublons
    if data.duplicated().sum() > 0:
        print(f"Suppression de {data.duplicated().sum()} doublons")
        data = data.drop_duplicates()
    else:
        print("Aucun doublon trouvé")
    
    # Séparer les features et la cible
    X = data.drop(['Churn', 'State'], axis=1)  # Exclure 'State' car non utilisé dans l'entraînement
    y = data['Churn'].astype(int)  # Convertir True/False en 1/0
    
    # Encoder les variables catégoriques
    categorical_cols = ['International plan', 'Voice mail plan']
    le = LabelEncoder()
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col])
    
    # Gérer les données déséquilibrées avec SMOTE
    #smote = SMOTE(random_state=42)
    #X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Mettre à l'échelle les features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, X.columns.tolist()


def train_model(X, y, model_type='rf'):
    """
    Entraîne un modèle de classification.
    
    Args:
        X (array): Features
        y (array): Cible
        model_type (str): Type de modèle ('rf' pour Random Forest, 'svm' pour SVM)
    
    Returns:
        model: Modèle entraîné
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'rf':
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
    elif model_type == 'svm':
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 0.01, 0.1, 1],
            'kernel': ['linear', 'rbf']
        }
        model = GridSearchCV(SVC(random_state=42), param_grid, scoring='accuracy', cv=3, verbose=1)
        model.fit(X_train, y_train)
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Évalue les performances du modèle et retourne les métriques
    
    Args:
        model: Modèle entraîné
        X_test (array): Features de test
        y_test (array): Target de test
    
    Returns:
        dict: Dictionnaire des métriques d'évaluation
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    # Affichage des résultats
    print(f"Accuracy: {metrics['accuracy']:.2f}")
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"Recall: {metrics['recall']:.2f}")
    print(f"F1 Score: {metrics['f1']:.2f}")
    print("\nMatrice de confusion:")
    print(metrics['confusion_matrix'])
    
    return metrics

def save_model(model, filename):
    """
    Sauvegarde le modèle entraîné.
    
    Args:
        model: Modèle à sauvegarder
        filename (str): Nom du fichier
    """
    joblib.dump(model, filename)
    print(f"Modèle sauvegardé sous {filename}")

def load_model(filename):
    """
    Charge un modèle sauvegardé.
    
    Args:
        filename (str): Nom du fichier
    
    Returns:
        model: Modèle chargé
    """
    model = joblib.load(filename)
    print(f"Modèle chargé depuis {filename}")
    return model

def main():
    file_path_80 = "churn-bigml-80.csv"
    file_path_20 = "churn-bigml-20.csv"
    
    X_scaled, y, feature_names = prepare_data(file_path_80, file_path_20)
    
    rf_model, X_test, y_test = train_model(X_scaled, y, model_type='rf')
    
    metrics = evaluate_model(rf_model, X_test, y_test)
    print("Métriques Random Forest :", metrics)
    
    save_model(rf_model, "rf_model.joblib")
    
    svm_model, X_test, y_test = train_model(X_scaled, y, model_type='svm')
    metrics_svm = evaluate_model(svm_model, X_test, y_test)
    print("Métriques SVM :", metrics_svm)
    save_model(svm_model, "svm_model.joblib")

if __name__ == "_main_":
    main()import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib 

def prepare_data(file_path_80, file_path_20):
    """
    Charge et prétraite les données à partir de deux fichiers CSV.
    
    Args:
        file_path_80 (str): Chemin vers le fichier churn-bigml-80.csv
        file_path_20 (str): Chemin vers le fichier churn-bigml-20.csv
    
    Returns:
        X_scaled (array): Features mises à l'échelle
        y (array): Variable cible
        feature_names (list): Noms des features
    """
    # Charger les données
    data1 = pd.read_csv(file_path_80)
    data2 = pd.read_csv(file_path_20)
    data = pd.concat([data1, data2], ignore_index=True)
    
    # Vérifier les doublons
    if data.duplicated().sum() > 0:
        print(f"Suppression de {data.duplicated().sum()} doublons")
        data = data.drop_duplicates()
    else:
        print("Aucun doublon trouvé")
    
    # Séparer les features et la cible
    X = data.drop(['Churn', 'State'], axis=1)  # Exclure 'State' car non utilisé dans l'entraînement
    y = data['Churn'].astype(int)  # Convertir True/False en 1/0
    
    # Encoder les variables catégoriques
    categorical_cols = ['International plan', 'Voice mail plan']
    le = LabelEncoder()
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col])
    
    # Gérer les données déséquilibrées avec SMOTE
    #smote = SMOTE(random_state=42)
    #X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Mettre à l'échelle les features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, X.columns.tolist()


def train_model(X, y, model_type='rf'):
    """
    Entraîne un modèle de classification.
    
    Args:
        X (array): Features
        y (array): Cible
        model_type (str): Type de modèle ('rf' pour Random Forest, 'svm' pour SVM)
    
    Returns:
        model: Modèle entraîné
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'rf':
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
    elif model_type == 'svm':
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 0.01, 0.1, 1],
            'kernel': ['linear', 'rbf']
        }
        model = GridSearchCV(SVC(random_state=42), param_grid, scoring='accuracy', cv=3, verbose=1)
        model.fit(X_train, y_train)
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Évalue les performances du modèle et retourne les métriques
    
    Args:
        model: Modèle entraîné
        X_test (array): Features de test
        y_test (array): Target de test
    
    Returns:
        dict: Dictionnaire des métriques d'évaluation
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    # Affichage des résultats
    print(f"Accuracy: {metrics['accuracy']:.2f}")
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"Recall: {metrics['recall']:.2f}")
    print(f"F1 Score: {metrics['f1']:.2f}")
    print("\nMatrice de confusion:")
    print(metrics['confusion_matrix'])
    
    return metrics

def save_model(model, filename):
    """
    Sauvegarde le modèle entraîné.
    
    Args:
        model: Modèle à sauvegarder
        filename (str): Nom du fichier
    """
    joblib.dump(model, filename)
    print(f"Modèle sauvegardé sous {filename}")

def load_model(filename):
    """
    Charge un modèle sauvegardé.
    
    Args:
        filename (str): Nom du fichier
    
    Returns:
        model: Modèle chargé
    """
    model = joblib.load(filename)
    print(f"Modèle chargé depuis {filename}")
    return model

def main():
    file_path_80 = "churn-bigml-80.csv"
    file_path_20 = "churn-bigml-20.csv"
    
    X_scaled, y, feature_names = prepare_data(file_path_80, file_path_20)
    
    rf_model, X_test, y_test = train_model(X_scaled, y, model_type='rf')
    
    metrics = evaluate_model(rf_model, X_test, y_test)
    print("Métriques Random Forest :", metrics)
    
    save_model(rf_model, "rf_model.joblib")
    
    svm_model, X_test, y_test = train_model(X_scaled, y, model_type='svm')
    metrics_svm = evaluate_model(svm_model, X_test, y_test)
    print("Métriques SVM :", metrics_svm)
    save_model(svm_model, "svm_model.joblib")

if __name__ == "_main_":

    main()
