"""
Extracao de landmarks com MediaPipe Holistic. Implementa o Contrato B.

    extrair_landmarks(frame)              -> dict (1 frame)
    construir_sequencia(lista_de_dicts)   -> np.ndarray (30, N_FEATURES)

ATENCAO ESPELHAMENTO: o Holistic rotula left/right pela anatomia da pessoa,
nao pelo lado da imagem. Mas se a Pessoa 5 aplicar cv2.flip() na webcam para
o efeito "espelho" antes de chamar extrair_landmarks(), as maos trocam em
relacao ao dataset e o modelo quebra. Combinar no kickoff: o flip so pode
acontecer DEPOIS da extracao, na hora de desenhar o overlay.
"""

from typing import Optional

import numpy as np

from .normalizacao import (
    normalizar_registro,
    preencher_lacunas,
    reamostrar,
    registro_para_vetor,
    ultima_janela,
)

_holistic = None


def _obter_holistic():
    """Instancia unica e preguicosa — o Holistic custa ~1s para carregar."""
    global _holistic
    if _holistic is None:
        import mediapipe as mp

        _holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,        # 2 e mais preciso e bem mais lento
            smooth_landmarks=True,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _holistic


def _reiniciar_holistic():
    """
    Descarta a instancia atual do Holistic.

    smooth_landmarks=True mantem um filtro temporal com estado entre frames —
    otimo dentro de UM video/sessao continua, mas contamina o inicio do
    proximo video se a mesma instancia for reaproveitada (suaviza os
    primeiros frames em direcao ao ultimo estado do video anterior). Chamar
    isso entre arquivos de video distintos.
    """
    global _holistic
    if _holistic is not None:
        _holistic.close()
        _holistic = None


def _pose_para_lista(lm) -> list:
    return [[p.x, p.y, p.z, p.visibility] for p in lm.landmark]


def _mao_para_lista(lm) -> Optional[list]:
    if lm is None:
        return None
    return [[p.x, p.y, p.z] for p in lm.landmark]


def extrair_landmarks(
    frame: np.ndarray, frame_id: int = 0, timestamp_ms: int = 0
) -> Optional[dict]:
    """
    Recebe 1 frame (480, 640, 3) BGR uint8 (Contrato A).
    Devolve 1 dict no formato do Contrato B, ja normalizado pelo centro dos
    ombros. Devolve None se nenhuma pose for detectada (sem pose nao ha
    referencia de normalizacao — o frame e descartado).
    """
    import cv2

    resultado = _obter_holistic().process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if resultado.pose_landmarks is None:
        return None

    bruto = {
        "frame_id": int(frame_id),
        "timestamp_ms": int(timestamp_ms),
        "pose": _pose_para_lista(resultado.pose_landmarks),
        "left_hand": _mao_para_lista(resultado.left_hand_landmarks),
        "right_hand": _mao_para_lista(resultado.right_hand_landmarks),
    }
    return normalizar_registro(bruto)


def construir_sequencia(
    landmarks_por_frame: list[dict],
    n_frames: int = 30,
    modo: str = "reamostrar",
) -> np.ndarray:
    """
    Lista de dicts (Contrato B) -> array (n_frames, N_FEATURES) float32.

    modo="reamostrar": comprime a sequencia inteira em n_frames. Use para
        sinais isolados (um video = um sinal) — treinamento.
    modo="ultima_janela": pega os ultimos n_frames. Use no fluxo ao vivo.
    """
    if not landmarks_por_frame:
        raise ValueError("landmarks_por_frame vazia")

    seq = np.stack([registro_para_vetor(r) for r in landmarks_por_frame])
    seq = preencher_lacunas(seq)

    if modo == "reamostrar":
        return reamostrar(seq, n_frames).astype(np.float32)
    if modo == "ultima_janela":
        return ultima_janela(seq, n_frames).astype(np.float32)
    raise ValueError(f"modo desconhecido: {modo}")


# --------------------------------------------------------------------------
# Utilitarios de dataset (nao fazem parte do contrato)
# --------------------------------------------------------------------------

def extrair_video(caminho: str, passo: int = 1) -> tuple[list[dict], dict]:
    """
    Extrai landmarks de um arquivo de video inteiro.
    Devolve (lista de registros do Contrato B, estatisticas de deteccao).

    passo > 1 subamostra frames — util para videos a 60 fps.
    """
    import cv2

    _reiniciar_holistic()  # cada video e uma sessao de suavizacao nova

    cap = cv2.VideoCapture(caminho)
    if not cap.isOpened():
        raise IOError(f"nao consegui abrir {caminho}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    registros, i = [], 0
    total, com_pose, com_mao = 0, 0, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % passo == 0:
            total += 1
            reg = extrair_landmarks(frame, frame_id=i, timestamp_ms=int(i / fps * 1000))
            if reg is not None:
                com_pose += 1
                if reg["left_hand"] is not None or reg["right_hand"] is not None:
                    com_mao += 1
                registros.append(reg)
        i += 1

    cap.release()

    stats = {
        "frames_lidos": total,
        "taxa_pose": com_pose / total if total else 0.0,
        "taxa_mao": com_mao / total if total else 0.0,   # criterio de aceite: > 0.90
    }
    return registros, stats
