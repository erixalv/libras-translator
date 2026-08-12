"""
Permite o desenvolvimento e teste da interface Streamlit e pipeline de integração
sem depender dos módulos de Deep Learning ou NLP já treinados.
"""

import json
import os
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np


VOCABULARIO_PADRAO = [
    "EU", "VOCE", "ELE_ELA", "OI", "TCHAU", "OBRIGADO", "POR_FAVOR",
    "DESCULPA", "SIM", "NAO", "NOME", "QUERER", "PRECISAR", "AJUDA",
    "AGUA", "COMIDA", "BANHEIRO", "CASA", "TRABALHO", "ESTUDAR",
    "GOSTAR", "BOM", "RUIM", "GRANDE", "PEQUENO", "HOJE", "AMANHA"
]


def carregar_vocabulario_mock() -> List[str]:
    """
    Carrega a lista de sinais de vocabulario.json na raiz do projeto.
    Se o arquivo não existir ou for inválido, retorna VOCABULARIO_PADRAO.
    """
    caminho_raiz = Path(__file__).resolve().parents[3]
    caminho_vocab = caminho_raiz / "vocabulario.json"

    if caminho_vocab.exists():
        try:
            with open(caminho_vocab, "r", encoding="utf-8") as f:
                dados = json.load(f)
                if isinstance(dados, dict) and "sinais" in dados and isinstance(dados["sinais"], list) and dados["sinais"]:
                    return [str(s) for s in dados["sinais"]]
        except Exception:
            pass

    return list(VOCABULARIO_PADRAO)


# Cache do vocabulário em memória
_VOCABULARIO_MOCK = carregar_vocabulario_mock()


def _top_k_mock(gloss_vencedor: str, confidence_vencedor: float, top_k: int) -> List[Dict[str, Any]]:
    resto = [g for g in _VOCABULARIO_MOCK if g != gloss_vencedor]
    candidatos = random.sample(resto, min(top_k - 1, len(resto)))
    confs = sorted((round(random.uniform(0.05, confidence_vencedor), 4) for _ in candidatos), reverse=True)
    lista = [{"gloss": gloss_vencedor, "confidence": round(confidence_vencedor, 4)}]
    lista += [{"gloss": g, "confidence": c} for g, c in zip(candidatos, confs)]
    return lista


def predict(sequencia: Optional[np.ndarray] = None, top_k: int = 5) -> Dict[str, Any]:
    """
    Args:
        sequencia: numpy array de shape (30, N_FEATURES). Se None ou de formato arbitrário,
                   o stub gera predição mock válida.

    Returns:
        Dict no formato {"gloss": str, "confidence": float, "timestamp_ms": int, "top_k": [...]}
    """
    timestamp_ms = int(time.time() * 1000)

    # 15% de chance de retornar "NENHUM" para simular ruído/intervalo entre sinais
    if random.random() < 0.15:
        confidence = round(random.uniform(0.10, 0.58), 2)
        return {
            "gloss": "NENHUM",
            "confidence": confidence,
            "timestamp_ms": timestamp_ms,
            "top_k": _top_k_mock(random.choice(_VOCABULARIO_MOCK), confidence, top_k),
        }

    gloss = random.choice(_VOCABULARIO_MOCK)
    confidence = round(random.uniform(0.65, 0.99), 2)

    return {
        "gloss": gloss,
        "confidence": confidence,
        "timestamp_ms": timestamp_ms,
        "top_k": _top_k_mock(gloss, confidence, top_k),
    }


# Mapeamentos pré-definidos de glosas para frases em Português
MAPA_FRASES_MOCK: Dict[tuple, str] = {
    ("EU", "QUERER", "AGUA"): "Eu quero água.",
    ("EU", "QUERER", "COMIDA"): "Eu quero comida.",
    ("EU", "PRECISAR", "AJUDA"): "Eu preciso de ajuda.",
    ("EU", "GOSTAR", "ESTUDAR"): "Eu gosto de estudar.",
    ("VOCE", "PRECISAR", "AJUDA"): "Você precisa de ajuda?",
    ("QUAL", "SEU", "NOME"): "Qual é o seu nome?",
    ("MEU", "NOME"): "Meu nome é...",
    ("OI", "TCHAU"): "Oi, tchau!",
    ("BOM", "DIA"): "Bom dia!",
    ("BOA", "NOITE"): "Boa noite!",
    ("OBRIGADO"): "Muito obrigado.",
    ("DESCULPA"): "Peço desculpas.",
    ("SIM"): "Sim, correto.",
    ("NAO"): "Não, obrigado.",
}


def glosas_para_frase(glosas: List[str]) -> Dict[str, Any]:
    """

    Args:
        glosas: Lista de strings com as glosas acumuladas.

    Returns:
        Dict no formato {"glosas_recebidas": List[str], "frase": str}
    """
    if not glosas:
        return {
            "glosas_recebidas": [],
            "frase": ""
        }

    # Limpeza / normalização das glosas
    glosas_limpas = [str(g).strip().upper() for g in glosas if str(g).strip()]

    if not glosas_limpas:
        return {
            "glosas_recebidas": glosas,
            "frase": ""
        }

    chave_tuple = tuple(glosas_limpas)

    # 1. Procura combinação exata no mapa
    if chave_tuple in MAPA_FRASES_MOCK:
        frase = MAPA_FRASES_MOCK[chave_tuple]
    else:
        # 2. Regra heurística genérica de fallback
        # Exemplo: ["EU", "GOSTAR", "AGUA"] -> "Eu gostar agua." -> Formatado bonitinho
        palavras_formatadas = []
        for i, g in enumerate(glosas_limpas):
            val = g.lower()
            if i == 0:
                val = val.capitalize()
            palavras_formatadas.append(val)

        frase = " ".join(palavras_formatadas)
        if not frase.endswith((".", "!", "?")):
            frase += "."

    return {
        "glosas_recebidas": list(glosas),
        "frase": frase
    }
