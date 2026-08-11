"""
Extracao de landmarks com MediaPipe (Holistic ou Tasks API). Implementa o Contrato B.

    extrair_landmarks(frame)              -> dict (1 frame)
    extrair_landmarks_anotado(frame)      -> (dict, frame com pose/maos desenhados) -- so p/ debug visual ao vivo
    construir_sequencia(lista_de_dicts)   -> np.ndarray (30, N_FEATURES)

ATENCAO ESPELHAMENTO: o MediaPipe rotula left/right pela anatomia da pessoa,
nao pelo lado da imagem. Mas se a Pessoa 5 aplicar cv2.flip() na webcam para
o efeito "espelho" antes de chamar extrair_landmarks(), as maos trocam em
relacao ao dataset e o modelo quebra. Combinar no kickoff: o flip so pode
acontecer DEPOIS da extracao, na hora de desenhar o overlay.
"""

import os
import urllib.request
from typing import Optional, Tuple
import numpy as np

from .normalizacao import (
    normalizar_registro,
    preencher_lacunas,
    reamostrar,
    registro_para_vetor,
    ultima_janela,
)

_holistic = None
_tasks_pose = None
_tasks_hand = None
_modo_extracao = None  # "solutions" ou "tasks"


def _garantir_modelos_tasks() -> Tuple[str, str]:
    """Garante que os arquivos .task existam em data/processed/."""
    raiz = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dir_proc = os.path.join(raiz, "data", "processed")
    os.makedirs(dir_proc, exist_ok=True)

    pose_path = os.path.join(dir_proc, "pose_landmarker.task")
    hand_path = os.path.join(dir_proc, "hand_landmarker.task")

    if not os.path.exists(pose_path):
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        urllib.request.urlretrieve(url, pose_path)

    if not os.path.exists(hand_path):
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        urllib.request.urlretrieve(url, hand_path)

    return pose_path, hand_path


def _obter_extrator():
    """Retorna o modo de extracao ativo ('solutions' ou 'tasks')."""
    global _holistic, _tasks_pose, _tasks_hand, _modo_extracao

    if _modo_extracao is not None:
        return _modo_extracao

    import mediapipe as mp

    # Tenta 1: mp.solutions.holistic (Legacy API)
    solutions = getattr(mp, "solutions", None)
    if solutions is not None and hasattr(solutions, "holistic"):
        try:
            _holistic = solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                refine_face_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            _modo_extracao = "solutions"
            return _modo_extracao
        except Exception:
            _holistic = None

    # Tenta 2: mp.tasks.python.vision (Tasks API)
    try:
        from mediapipe.tasks.python import vision, BaseOptions
        pose_path, hand_path = _garantir_modelos_tasks()

        opts_pose = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_path),
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        opts_hand = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hand_path),
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _tasks_pose = vision.PoseLandmarker.create_from_options(opts_pose)
        _tasks_hand = vision.HandLandmarker.create_from_options(opts_hand)
        _modo_extracao = "tasks"
        return _modo_extracao
    except Exception as e:
        import logging
        logging.warning(f"Erro ao carregar MediaPipe (solutions ou tasks): {e}")
        return None


def _reiniciar_holistic():
    global _holistic, _tasks_pose, _tasks_hand, _modo_extracao
    if _holistic is not None:
        try:
            _holistic.close()
        except Exception:
            pass
        _holistic = None
    if _tasks_pose is not None:
        try:
            _tasks_pose.close()
        except Exception:
            pass
        _tasks_pose = None
    if _tasks_hand is not None:
        try:
            _tasks_hand.close()
        except Exception:
            pass
        _tasks_hand = None
    _modo_extracao = None


def _detectar_bruto(frame: np.ndarray) -> Optional[dict]:
    """
    Roda o Holistic/Tasks 1 vez e devolve pose/maos em coordenadas de imagem
    (fracao 0..1 da largura/altura do frame, ainda SEM a normalizacao pelo
    centro dos ombros). Usado tanto por extrair_landmarks() (Contrato B)
    quanto por extrair_landmarks_anotado() (desenha os pontos no frame) --
    assim os dois reaproveitam a MESMA inferencia, sem rodar o MediaPipe
    2x por frame.
    """
    import cv2
    import mediapipe as mp

    modo = _obter_extrator()
    if modo is None:
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if modo == "solutions":
        res = _holistic.process(frame_rgb)
        if res.pose_landmarks is None:
            return None
        return {
            "pose": [[p.x, p.y, p.z, p.visibility] for p in res.pose_landmarks.landmark],
            "left_hand": [[p.x, p.y, p.z] for p in res.left_hand_landmarks.landmark] if res.left_hand_landmarks else None,
            "right_hand": [[p.x, p.y, p.z] for p in res.right_hand_landmarks.landmark] if res.right_hand_landmarks else None,
        }

    elif modo == "tasks":
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        pose_res = _tasks_pose.detect(mp_img)
        hand_res = _tasks_hand.detect(mp_img)

        if not pose_res.pose_landmarks or len(pose_res.pose_landmarks) == 0:
            return None

        pose_lms = pose_res.pose_landmarks[0]
        pose_list = [[p.x, p.y, p.z, getattr(p, "visibility", 1.0)] for p in pose_lms]

        left_hand_list = None
        right_hand_list = None

        if hand_res.hand_landmarks and hand_res.handedness:
            for hand_lms, handedness in zip(hand_res.hand_landmarks, hand_res.handedness):
                label = handedness[0].category_name  # "Left" ou "Right"
                lms = [[p.x, p.y, p.z] for p in hand_lms]
                if label == "Left":
                    right_hand_list = lms
                elif label == "Right":
                    left_hand_list = lms

        return {"pose": pose_list, "left_hand": left_hand_list, "right_hand": right_hand_list}

    return None


def extrair_landmarks(
    frame: np.ndarray, frame_id: int = 0, timestamp_ms: int = 0
) -> Optional[dict]:
    """
    Recebe 1 frame (480, 640, 3) BGR uint8 (Contrato A).
    Devolve 1 dict no formato do Contrato B, ja normalizado pelo centro dos
    ombros. Devolve None se nenhuma pose for detectada.
    """
    bruto = _detectar_bruto(frame)
    if bruto is None:
        return None
    bruto["frame_id"] = int(frame_id)
    bruto["timestamp_ms"] = int(timestamp_ms)
    return normalizar_registro(bruto)


def extrair_landmarks_anotado(
    frame: np.ndarray, frame_id: int = 0, timestamp_ms: int = 0
) -> Tuple[Optional[dict], np.ndarray]:
    """
    Igual extrair_landmarks(), mas tambem devolve uma copia do frame com os
    pontos de pose/maos desenhados por cima -- so pra visualizacao ao vivo
    (debug de "esta pegando minha mao?"), nao faz parte do Contrato B e nao
    e usado no pipeline de treino/dataset.
    """
    bruto = _detectar_bruto(frame)
    if bruto is None:
        return None, frame

    frame_anotado = _desenhar_landmarks(frame, bruto["pose"], bruto["left_hand"], bruto["right_hand"])

    bruto["frame_id"] = int(frame_id)
    bruto["timestamp_ms"] = int(timestamp_ms)
    return normalizar_registro(bruto), frame_anotado


def _desenhar_landmarks(
    frame: np.ndarray,
    pose: Optional[list],
    mao_esq: Optional[list],
    mao_dir: Optional[list],
) -> np.ndarray:
    """
    Desenha pose (ciano) e maos (mao esquerda verde, mao direita vermelha)
    num frame BGR, usando as coordenadas de imagem 0..1 devolvidas por
    _detectar_bruto(). As conexoes (POSE_CONNECTIONS/HAND_CONNECTIONS) sao
    so a topologia dos pontos -- vale tanto pro backend "solutions" quanto
    "tasks", que usam a mesma numeracao de landmarks.
    """
    import cv2

    POSE_CONNS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)]
    HAND_CONNS = [(0, 1), (1, 5), (9, 13), (13, 17), (5, 9), (0, 17), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20)]

    frame_anotado = frame.copy()
    h, w = frame_anotado.shape[:2]

    def px(ponto):
        return int(ponto[0] * w), int(ponto[1] * h)

    def desenhar(pontos, conexoes, cor):
        if not pontos:
            return
        for a, b in conexoes:
            if a < len(pontos) and b < len(pontos):
                cv2.line(frame_anotado, px(pontos[a]), px(pontos[b]), cor, 1, cv2.LINE_AA)
        for p in pontos:
            cv2.circle(frame_anotado, px(p), 3, cor, -1, cv2.LINE_AA)

    desenhar(pose, POSE_CONNS, (255, 200, 0))
    desenhar(mao_esq, HAND_CONNS, (0, 220, 0))
    desenhar(mao_dir, HAND_CONNS, (0, 0, 255))

    return frame_anotado


def construir_sequencia(
    landmarks_por_frame: list[dict],
    n_frames: int = 30,
    modo: str = "reamostrar",
) -> np.ndarray:
    """
    Lista de dicts (Contrato B) -> array (n_frames, N_FEATURES) float32.

    modo="reamostrar": comprime a sequencia inteira em n_frames.
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


def extrair_video(caminho: str, passo: int = 1) -> tuple[list[dict], dict]:
    """
    Extrai landmarks de um arquivo de video inteiro.
    Devolve (lista de registros do Contrato B, estatisticas de deteccao).
    """
    import cv2

    _reiniciar_holistic()

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
        "taxa_mao": com_mao / total if total else 0.0,
    }
    return registros, stats
