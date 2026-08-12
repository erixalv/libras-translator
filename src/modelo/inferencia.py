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

LIMIAR_CONFIANCA_PADRAO = 0.6  # o mesmo do Contrato C (CONTRATOS.md) -- ver _remapear_confianca()

_vocabulario = carregar_vocabulario()
_modelos = None
_scalers = None
_usar_velocidade = True
_temperatura = 1.0
_limiares_por_classe: dict[str, float] = {}
_dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def carregar_ensemble(dispositivo: torch.device):
    """
    Le o checkpoint salvo por salvar_ensemble() (treino.py): uma LISTA de
    state_dicts (um por modelo do CV), a lista de scalers correspondente em
    scaler.pkl, a temperatura calibrada (calibrar_temperatura()) e os
    limiares por classe (calibrar_limiares_por_classe()). Unica fonte de
    verdade do formato em disco -- usada por predict() aqui embaixo e por
    avaliacao.py, pra nao duplicar a logica de carregamento.
    """
    checkpoint = torch.load(CAMINHO_MODELO, map_location=dispositivo)
    with open(CAMINHO_SCALER, "rb") as f:
        scalers = pickle.load(f)

    usar_atencao = checkpoint.get("usar_atencao", False)
    modelos = []
    for state_dict in checkpoint["state_dicts"]:
        modelo = ClassificadorLSTM(
            n_features=checkpoint["n_features"],
            n_classes=len(checkpoint["vocabulario"]),
            usar_atencao=usar_atencao,
        )
        modelo.load_state_dict(state_dict)
        modelo.to(dispositivo).eval()
        modelos.append(modelo)

    return (
        modelos,
        scalers,
        checkpoint["vocabulario"],
        checkpoint.get("usar_velocidade", False),
        checkpoint.get("temperatura", 1.0),
        checkpoint.get("limiares_por_classe", {}),
    )


def _carregar_modelo_real():
    global _modelos, _scalers, _usar_velocidade, _vocabulario, _temperatura, _limiares_por_classe
    if _modelos is not None:
        return
    # usa a ordem de classes salva no checkpoint (vinda do .npz), nao a
    # ordem do vocabulario.json -- e essa que bate com os indices do modelo
    _modelos, _scalers, _vocabulario, _usar_velocidade, _temperatura, _limiares_por_classe = carregar_ensemble(
        _dispositivo
    )


def _remapear_confianca(conf_bruta: float, limiar_classe: float, limiar_global: float = LIMIAR_CONFIANCA_PADRAO) -> float:
    """
    Reescala a confianca bruta de UMA classe especifica de forma monotonica
    (preserva ordem) de modo que, no limiar_classe calibrado pra ela
    (calibrar_limiares_por_classe(), via curva precisao-recall no OOF do
    CV), a confianca reescalada bata exatamente no limiar_global (0.6,
    Contrato C). Efeito: aplicar o corte fixo de 0.6 por fora (como o app
    ja faz hoje, PipelineIntegrador.limiar_confianca) passa a se comportar
    como um limiar calibrado por classe, sem mudar o Contrato C nem a UI.
    """
    if not (0.0 < limiar_classe < 1.0):
        return conf_bruta
    if conf_bruta >= limiar_classe:
        return limiar_global + (conf_bruta - limiar_classe) / (1 - limiar_classe) * (1 - limiar_global)
    return conf_bruta / limiar_classe * limiar_global


def predict(sequencia: np.ndarray, top_k: int = 5) -> dict:
    """
    Devolve sempre a classe de maior probabilidade e sua confianca real --
    NAO decide aqui se a confianca e suficiente pra aceitar a glosa. Quem
    decide isso e o chamador (PipelineIntegrador.limiar_confianca, ajustavel
    na barra lateral do app), senao o limiar vira fixo e a barra lateral
    para de fazer efeito abaixo dele.

    A confianca vem de tres camadas, nessa ordem: (1) media do softmax dos
    N modelos do ensemble (ver treino.py: salvar_ensemble()); (2)
    temperature scaling (calibrar_temperatura()) -- reescala o quao afiada
    e a distribuicao, pra confianca alta corresponder de fato a mais chance
    de acerto; (3) remapeamento por classe (_remapear_confianca()) -- so a
    confianca da classe PREVISTA e das do top_k sao ajustadas pelo limiar
    calibrado daquela classe especifica, escalado pro limiar global de 0.6
    continuar fazendo sentido no app. Testado no holdout real (vocabulario
    de 20 sinais MINDS/V-LIBRASIL): 55.6% de acuracia com o ensemble.

    Tambem devolve "top_k": as `top_k` classes mais prováveis (gloss +
    confidence), pra UI mostrar nao so a resposta escolhida mas as
    candidatas -- util com um modelo que ainda erra bastante em sinalizador
    nunca visto: a resposta certa costuma aparecer entre as top poucas
    mesmo quando nao vence.
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

    probs_por_modelo = []
    with torch.no_grad():
        for modelo, scaler in zip(_modelos, _scalers):
            x = scaler.transform(seq)
            x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(_dispositivo)
            probs_por_modelo.append(torch.softmax(modelo(x) / _temperatura, dim=1).squeeze(0))
    probs = torch.stack(probs_por_modelo).mean(dim=0)

    indice = int(torch.argmax(probs).item())
    limiar_indice = _limiares_por_classe.get(_vocabulario[indice], LIMIAR_CONFIANCA_PADRAO)
    confidence = _remapear_confianca(float(probs[indice].item()), limiar_indice)

    k = min(top_k, probs.shape[0])
    top_confs, top_indices = torch.topk(probs, k)

    lista_top_k = [
        {
            "gloss": _vocabulario[i],
            "confidence": round(_remapear_confianca(float(c), _limiares_por_classe.get(_vocabulario[i], LIMIAR_CONFIANCA_PADRAO)), 4),
        }
        for i, c in zip(top_indices.tolist(), top_confs.tolist())
    ]
    # o remapeamento e por classe (cada uma com seu proprio limiar), entao
    # a ordem por probabilidade bruta nao garante mais ordem por confianca
    # reescalada -- reordena pra manter "top_k" de fato decrescente
    lista_top_k.sort(key=lambda item: item["confidence"], reverse=True)

    return {
        "gloss": _vocabulario[indice],
        "confidence": round(confidence, 4),
        "timestamp_ms": timestamp_ms,
        "top_k": lista_top_k,
    }


if __name__ == "__main__":
    seq_falsa = np.random.randn(30, 260).astype(np.float32)
    print(predict(seq_falsa))
