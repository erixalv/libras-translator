import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.modelo.dataset import carregar_npz


def extrair_features_simples(X: np.ndarray) -> np.ndarray:
    """(N, 30, 260) -> (N, 260*2), concatenando media e desvio-padrao
    de cada feature ao longo do tempo."""
    media = X.mean(axis=1)
    desvio = X.std(axis=1)
    return np.concatenate([media, desvio], axis=1)


def main() -> None:
    dados = carregar_npz()
    classes = dados["classes"]

    x_treino = extrair_features_simples(dados["X_train"])
    x_teste = extrair_features_simples(dados["X_test"])
    y_treino, y_teste = dados["y_train"], dados["y_test"]

    scaler = StandardScaler()
    x_treino = scaler.fit_transform(x_treino)
    x_teste = scaler.transform(x_teste)

    nomes_classes = [classes[i] for i in sorted(set(y_treino.tolist()) | set(y_teste.tolist()))]

    for nome, modelo in [
        ("SVM (RBF)", SVC(kernel="rbf", C=1.0)),
        ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42)),
    ]:
        modelo.fit(x_treino, y_treino)
        preds = modelo.predict(x_teste)
        acc = accuracy_score(y_teste, preds)
        print(f"\n=== {nome} — acuracia no holdout de sinalizador: {acc:.4f} ===")
        print(classification_report(y_teste, preds, labels=sorted(set(y_teste.tolist())),
                                     target_names=[classes[i] for i in sorted(set(y_teste.tolist()))],
                                     zero_division=0))


if __name__ == "__main__":
    main()