import os

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.modelo.dataset import LibrasLandmarksDataset, adicionar_features_velocidade, carregar_npz
from src.modelo.inferencia import carregar_ensemble

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR_PROCESSED = os.path.join(RAIZ, "data", "processed")
DIR_FIGURAS = os.path.join(RAIZ, "docs", "figuras")


def avaliar() -> None:
    """Avalia o ensemble de producao (carregar_ensemble(), ver inferencia.py)
    no holdout de teste: media do softmax dos N modelos do CV, mesma logica
    usada por predict() e por avaliar_ensemble() em treino.py."""
    os.makedirs(DIR_FIGURAS, exist_ok=True)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dados = carregar_npz()
    classes = dados["classes"]

    modelos, scalers, vocabulario, usar_velocidade = carregar_ensemble(dispositivo)
    assert list(vocabulario) == list(classes), "vocabulario do checkpoint nao bate com o do dataset atual"

    X_test = adicionar_features_velocidade(dados["X_test"]) if usar_velocidade else dados["X_test"]

    probs_por_modelo = []
    with torch.no_grad():
        for modelo, scaler in zip(modelos, scalers):
            ds = LibrasLandmarksDataset(X_test, dados["y_test"], scaler=scaler, fit_scaler=False)
            dl = DataLoader(ds, batch_size=16, shuffle=False)
            probs_modelo = []
            for x, _ in dl:
                probs_modelo.append(torch.softmax(modelo(x.to(dispositivo)), dim=1).cpu())
            probs_por_modelo.append(torch.cat(probs_modelo))

    probs_media = torch.stack(probs_por_modelo).mean(dim=0)
    y_pred = probs_media.argmax(dim=1).tolist()
    y_true = dados["y_test"].tolist()

    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 12))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, xticks_rotation=90, colorbar=False)
    plt.tight_layout()
    caminho_fig = os.path.join(DIR_FIGURAS, "matriz_confusao.png")
    plt.savefig(caminho_fig, dpi=150)
    print(f"Matriz de confusao salva em: {caminho_fig}")


if __name__ == "__main__":
    avaliar()
