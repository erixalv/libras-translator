import os
import pickle
import random
import time

import numpy as np
import torch

from src.modelo.arquiteturas import ClassificadorLSTM
from src.modelo.dataset import carregar_vocabulario

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR_PROCESSED = os.path.join(RAIZ, "data", "processed")
CAMINHO_MODELO = os.path.join(DIR_PROCESSED, "modelo_melhor.pt")
CAMINHO_SCALER = os.path.join(DIR_PROCESSED, "scaler.pkl")

THRESHOLD_CONFIANCA = 0.6

_vocabulario = carregar_vocabulario()
_modelo = None
_scaler = None
_dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _carregar_modelo_real():
    global _modelo, _scaler
    if _modelo is not None:
        return
    checkpoint = torch.load(CAMINHO_MODELO, map_location=_dispositivo)
    # usa a ordem de classes salva no checkpoint (vinda do .npz), nao a
    # ordem do vocabulario.json -- e essa que bate com os indices do modelo
    global _vocabulario
    _vocabulario = checkpoint["vocabulario"]
    _modelo = ClassificadorLSTM(n_features=checkpoint["n_features"], n_classes=len(checkpoint["vocabulario"]))
    _modelo.load_state_dict(checkpoint["state_dict"])
    _modelo.to(_dispositivo).eval()
    with open(CAMINHO_SCALER, "rb") as f:
        _scaler = pickle.load(f)


def predict(sequencia: np.ndarray) -> dict:
    timestamp_ms = int(time.time() * 1000)
    modelo_existe = os.path.exists(CAMINHO_MODELO) and os.path.exists(CAMINHO_SCALER)

    if not modelo_existe:
        gloss = random.choice(_vocabulario)
        confidence = round(random.uniform(0.55, 0.95), 2)
        return {"gloss": gloss, "confidence": confidence, "timestamp_ms": timestamp_ms}

    _carregar_modelo_real()
    x = _scaler.transform(sequencia)
    x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(_dispositivo)

    with torch.no_grad():
        logits = _modelo(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        indice = int(torch.argmax(probs).item())
        confidence = float(probs[indice].item())

    if confidence < THRESHOLD_CONFIANCA:
        return {"gloss": "NENHUM", "confidence": round(confidence, 4), "timestamp_ms": timestamp_ms}

    return {"gloss": _vocabulario[indice], "confidence": round(confidence, 4), "timestamp_ms": timestamp_ms}


if __name__ == "__main__":
    seq_falsa = np.random.randn(30, 260).astype(np.float32)
    print(predict(seq_falsa))