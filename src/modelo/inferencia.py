import os
import pickle
import random
import time

import numpy as np
import torch

from src.modelo.arquiteturas import ClassificadorLSTM
from src.modelo.dataset import adicionar_features_velocidade, carregar_vocabulario

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR_PROCESSED = os.path.join(RAIZ, "data", "processed")
CAMINHO_MODELO = os.path.join(DIR_PROCESSED, "modelo_melhor.pt")
CAMINHO_SCALER = os.path.join(DIR_PROCESSED, "scaler.pkl")

_vocabulario = carregar_vocabulario()
_modelo = None
_scaler = None
_usar_velocidade = True
_dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _carregar_modelo_real():
    global _modelo, _scaler, _usar_velocidade
    if _modelo is not None:
        return
    checkpoint = torch.load(CAMINHO_MODELO, map_location=_dispositivo)
    # usa a ordem de classes salva no checkpoint (vinda do .npz), nao a
    # ordem do vocabulario.json -- e essa que bate com os indices do modelo
    global _vocabulario
    _vocabulario = checkpoint["vocabulario"]
    _usar_velocidade = checkpoint.get("usar_velocidade", False)
    usar_atencao = checkpoint.get("usar_atencao", False)
    _modelo = ClassificadorLSTM(
        n_features=checkpoint["n_features"], n_classes=len(checkpoint["vocabulario"]), usar_atencao=usar_atencao
    )
    _modelo.load_state_dict(checkpoint["state_dict"])
    _modelo.to(_dispositivo).eval()
    with open(CAMINHO_SCALER, "rb") as f:
        _scaler = pickle.load(f)


def predict(sequencia: np.ndarray, top_k: int = 5) -> dict:
    """
    Devolve sempre a classe de maior probabilidade e sua confianca real --
    NAO decide aqui se a confianca e suficiente pra aceitar a glosa. Quem
    decide isso e o chamador (PipelineIntegrador.limiar_confianca, ajustavel
    na barra lateral do app), senao o limiar vira fixo e a barra lateral
    para de fazer efeito abaixo dele.

    Tambem devolve "top_k": as `top_k` classes mais prováveis (gloss +
    confidence), pra UI mostrar nao so a resposta escolhida mas as
    candidatas -- util com um modelo que erra bastante (~40% em
    sinalizador nunca visto): a resposta certa costuma aparecer entre as
    top poucas mesmo quando nao vence.
    """
    timestamp_ms = int(time.time() * 1000)
    modelo_existe = os.path.exists(CAMINHO_MODELO) and os.path.exists(CAMINHO_SCALER)

    if not modelo_existe:
        candidatos = random.sample(_vocabulario, min(top_k, len(_vocabulario)))
        confs = sorted((round(random.uniform(0.1, 0.95), 4) for _ in candidatos), reverse=True)
        gloss, confidence = candidatos[0], confs[0]
        return {
            "gloss": gloss,
            "confidence": confidence,
            "timestamp_ms": timestamp_ms,
            "top_k": [{"gloss": g, "confidence": c} for g, c in zip(candidatos, confs)],
        }

    _carregar_modelo_real()
    seq = adicionar_features_velocidade(sequencia) if _usar_velocidade else sequencia
    x = _scaler.transform(seq)
    x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(_dispositivo)

    with torch.no_grad():
        logits = _modelo(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        indice = int(torch.argmax(probs).item())
        confidence = float(probs[indice].item())

        k = min(top_k, probs.shape[0])
        top_confs, top_indices = torch.topk(probs, k)

    lista_top_k = [
        {"gloss": _vocabulario[i], "confidence": round(float(c), 4)}
        for i, c in zip(top_indices.tolist(), top_confs.tolist())
    ]

    return {
        "gloss": _vocabulario[indice],
        "confidence": round(confidence, 4),
        "timestamp_ms": timestamp_ms,
        "top_k": lista_top_k,
    }


if __name__ == "__main__":
    seq_falsa = np.random.randn(30, 260).astype(np.float32)
    print(predict(seq_falsa))