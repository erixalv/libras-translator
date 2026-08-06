"""
Módulo de visualização e desenho de legenda (Overlay OpenCV).
Desenha caixas semitransparentes com as glosas detectadas e a frase traduzida.
"""

import unicodedata
from typing import List, Union, Optional
import cv2
import numpy as np


def _remover_acentos(texto: str) -> str:
    """
    Remove caracteres acentuados para compatibilidade com cv2.putText.
    O OpenCV FONT_HERSHEY_SIMPLEX não suporta caracteres UTF-8 estendidos.
    """
    if not texto:
        return ""
    nfkd = unicodedata.normalize('NFKD', texto)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])


def desenhar_overlay(
    frame: np.ndarray,
    frase: str = "",
    glosas: Optional[Union[List[str], str]] = None,
    glosa_atual: Optional[str] = None,
    confianca: Optional[float] = None,
    altura_barra_pct: float = 0.22,
    alpha: float = 0.65
) -> np.ndarray:
    """
    Desenha uma caixa semitransparente na parte inferior do frame com:
      - Glosas ativas/acumuladas (ex.: "Glosas: EU... QUERER... AGUA")
      - Frase em Português traduzida (ex.: "Frase: Eu quero água.")

    Args:
        frame: Array NumPy de formato (H, W, 3), dtype uint8 (BGR OpenCV).
        frase: Texto em português traduzido.
        glosas: Lista de glosas acumuladas no buffer ou string formatada.
        glosa_atual: Glosa pontual detectada no frame atual (opcional).
        confianca: Confiança da detecção atual entre 0.0 e 1.0 (opcional).
        altura_barra_pct: Percentual da altura do frame ocupado pela legenda inferior.
        alpha: Nível de opacidade da barra (0.0 transparente a 1.0 opaco).

    Returns:
        Novo ndarray do frame com a legenda sobreposta (mantém mesmo shape e dtype).
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
        raise ValueError("Frame deve ser um np.ndarray com 3 dimensões (H, W, C)")

    # Cria cópia para não alterar o array original em vigor
    frame_saida = frame.copy()
    h, w, c = frame_saida.shape

    # 1. Ajuste de parâmetros com base na resolução do frame (proporcional)
    altura_barra = int(h * altura_barra_pct)
    y_inicio = h - altura_barra

    # 2. Desenho da faixa semitransparente preta
    overlay = frame_saida.copy()
    cv2.rectangle(overlay, (0, y_inicio), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, alpha, frame_saida, 1 - alpha, 0, frame_saida)

    # Linha divisória superior fina em azul/ciano
    cv2.line(frame_saida, (0, y_inicio), (w, y_inicio), (235, 160, 50), 2)

    # 3. Formatação dos textos
    # Glosas acumuladas
    if isinstance(glosas, list):
        texto_glosas = " ".join([str(g).upper() for g in glosas if str(g).strip()])
    elif isinstance(glosas, str):
        texto_glosas = glosas
    else:
        texto_glosas = ""

    if glosa_atual and glosa_atual != "NENHUM":
        conf_str = f" ({int(confianca * 100)}%)" if confianca is not None else ""
        texto_glosa_inst = f"Atual: {glosa_atual}{conf_str}"
    else:
        texto_glosa_inst = "Atual: ---"

    texto_frase = f"Frase: {frase}" if frase else "Frase: ..."

    # 4. Renderização do texto no OpenCV
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    escala = max(0.45, w / 1100.0)
    espessura = max(1, int(escala * 2))

    # Posições Y das linhas de texto
    y_linha1 = y_inicio + int(altura_barra * 0.32)
    y_linha2 = y_inicio + int(altura_barra * 0.75)

    # Linha 1: Glosa atual e Glosas no buffer
    str_linha1 = f"{texto_glosa_inst}"
    if texto_glosas:
        str_linha1 += f"  |  Buffer: {texto_glosas}"

    str_linha1_ascii = _remover_acentos(str_linha1)
    cv2.putText(
        frame_saida,
        str_linha1_ascii,
        (15, y_linha1),
        fonte,
        escala * 0.85,
        (200, 230, 255),
        espessura,
        cv2.LINE_AA
    )

    # Linha 2: Frase traduzida final
    str_linha2_ascii = _remover_acentos(texto_frase)
    cv2.putText(
        frame_saida,
        str_linha2_ascii,
        (15, y_linha2),
        fonte,
        escala * 1.0,
        (255, 255, 255),
        espessura + 1,
        cv2.LINE_AA
    )

    return frame_saida
