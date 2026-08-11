import os
import pickle

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.modelo.arquiteturas import ClassificadorLSTM
from src.modelo.dataset import LibrasLandmarksDataset, adicionar_features_velocidade, carregar_npz

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR_PROCESSED = os.path.join(RAIZ, "data", "processed")
DIR_FIGURAS = os.path.join(RAIZ, "docs", "figuras")


def avaliar() -> None:
    os.makedirs(DIR_FIGURAS, exist_ok=True)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dados = carregar_npz()
    classes = dados["classes"]

    with open(os.path.join(DIR_PROCESSED, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    checkpoint = torch.load(os.path.join(DIR_PROCESSED, "modelo_melhor.pt"), map_location=dispositivo)
    usar_velocidade = checkpoint.get("usar_velocidade", False)
    usar_atencao = checkpoint.get("usar_atencao", False)

    X_test = adicionar_features_velocidade(dados["X_test"]) if usar_velocidade else dados["X_test"]
    ds_teste = LibrasLandmarksDataset(X_test, dados["y_test"], scaler=scaler, fit_scaler=False)
    dl_teste = DataLoader(ds_teste, batch_size=16, shuffle=False)

    modelo = ClassificadorLSTM(
        n_features=checkpoint["n_features"], n_classes=len(classes), usar_atencao=usar_atencao
    )
    modelo.load_state_dict(checkpoint["state_dict"])
    modelo.to(dispositivo).eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in dl_teste:
            x = x.to(dispositivo)
            preds = modelo(x).argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(y.numpy().tolist())

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