import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.modelo.dataset import carregar_dados_brutos, carregar_vocabulario


def extrair_features_simples(sequencias: np.ndarray) -> np.ndarray:
    """(N, 30, 258) -> (N, 258*2), concatenando media e desvio-padrao
    de cada feature ao longo do tempo."""
    media = sequencias.mean(axis=1)
    desvio = sequencias.std(axis=1)
    return np.concatenate([media, desvio], axis=1)


def main() -> None:
    vocabulario = carregar_vocabulario()
    sequencias, rotulos = carregar_dados_brutos()

    x = extrair_features_simples(sequencias)
    x_treino, x_val, y_treino, y_val = train_test_split(
        x, rotulos, test_size=0.2, random_state=42, stratify=rotulos
    )

    scaler = StandardScaler()
    x_treino = scaler.fit_transform(x_treino)
    x_val = scaler.transform(x_val)

    for nome, modelo in [
        ("SVM (RBF)", SVC(kernel="rbf", C=1.0)),
        ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42)),
    ]:
        modelo.fit(x_treino, y_treino)
        preds = modelo.predict(x_val)
        acc = accuracy_score(y_val, preds)
        print(f"\n=== {nome} — acuracia: {acc:.4f} ===")
        print(classification_report(y_val, preds, zero_division=0))


if __name__ == "__main__":
    main()
