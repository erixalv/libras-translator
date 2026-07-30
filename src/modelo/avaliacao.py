import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.modelo.arquiteturas import ClassificadorLSTM
from src.modelo.dataset import LibrasLandmarksDataset, carregar_dados_brutos, carregar_vocabulario

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR_PROCESSED = os.path.join(RAIZ, "data", "processed")
DIR_FIGURAS = os.path.join(RAIZ, "docs", "figuras")


def avaliar() -> None:
    os.makedirs(DIR_FIGURAS, exist_ok=True)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocabulario = carregar_vocabulario()
    sequencias, rotulos = carregar_dados_brutos()
    _, seq_val, _, rot_val = train_test_split(
        sequencias, rotulos, test_size=0.2, random_state=42, stratify=rotulos
    )

    with open(os.path.join(DIR_PROCESSED, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    ds_val = LibrasLandmarksDataset(seq_val, rot_val, vocabulario, scaler=scaler, fit_scaler=False)
    dl_val = DataLoader(ds_val, batch_size=16, shuffle=False)

    checkpoint = torch.load(os.path.join(DIR_PROCESSED, "modelo_melhor.pt"), map_location=dispositivo)
    modelo = ClassificadorLSTM(n_features=checkpoint["n_features"], n_classes=len(vocabulario))
    modelo.load_state_dict(checkpoint["state_dict"])
    modelo.to(dispositivo).eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in dl_val:
            x = x.to(dispositivo)
            preds = modelo(x).argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(y.numpy().tolist())

    print(classification_report(y_true, y_pred, target_names=vocabulario, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay(cm, display_labels=vocabulario).plot(ax=ax, xticks_rotation=90, colorbar=False)
    plt.tight_layout()
    caminho_fig = os.path.join(DIR_FIGURAS, "matriz_confusao.png")
    plt.savefig(caminho_fig, dpi=150)
    print(f"Matriz de confusao salva em: {caminho_fig}")


if __name__ == "__main__":
    avaliar()
