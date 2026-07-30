import json
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

THRESHOLD_CONFIANCA = 0.6  # travado na Secao 3.5 do CONTRATOS.md

_vocabulario = carregar_vocabulario()
_modelo = None
_scaler = None
_dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _carregar_modelo_real():
    global _modelo, _scaler
    if _modelo is not None:
        return
    checkpoint = torch.load(CAMINHO_MODELO, map_location=_dispositivo)
    _modelo = ClassificadorLSTM(n_features=checkpoint["n_features"], n_classes=len(checkpoint["vocabulario"]))
    _modelo.load_state_dict(checkpoint["state_dict"])
    _modelo.to(_dispositivo).eval()
    with open(CAMINHO_SCALER, "rb") as f:
        _scaler = pickle.load(f)


def predict(sequencia: np.ndarray) -> dict:
    """
    sequencia: array (30, N_FEATURES), saida do Contrato B
    retorna: dict no formato do Contrato C -> {"gloss": str, "confidence": float, "timestamp_ms": int}
    """
    timestamp_ms = int(time.time() * 1000)

    modelo_existe = os.path.exists(CAMINHO_MODELO) and os.path.exists(CAMINHO_SCALER)

    if not modelo_existe:
        # modo mock -- destrava P4 e P5 antes do treino terminar
        gloss = random.choice(_vocabulario)
        confidence = round(random.uniform(0.55, 0.95), 2)
        return {"gloss": gloss, "confidence": confidence, "timestamp_ms": timestamp_ms}

    _carregar_modelo_real()
    x = _scaler.transform(sequencia)
    x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(_dispositivo)  # (1, 30, n_features)

    with torch.no_grad():
        logits = _modelo(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        indice = int(torch.argmax(probs).item())
        confidence = float(probs[indice].item())

    if confidence < THRESHOLD_CONFIANCA:
        return {"gloss": "NENHUM", "confidence": round(confidence, 4), "timestamp_ms": timestamp_ms}

    return {"gloss": _vocabulario[indice], "confidence": round(confidence, 4), "timestamp_ms": timestamp_ms}


if __name__ == "__main__":
    # teste rapido manual
    seq_falsa = np.random.randn(30, 258).astype(np.float32)
    print(predict(seq_falsa))
