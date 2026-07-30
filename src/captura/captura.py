from typing import Iterator
import cv2
import numpy as np
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def capturar_frames(fonte: str | int = 0) -> Iterator[np.ndarray]:
    """
    Captura frames de uma webcam ou arquivo de vídeo.
    fonte: 0 (webcam padrão) ou caminho de arquivo .mp4
    yield: frame (480, 640, 3) BGR uint8, um por vez
    """
    cap = cv2.VideoCapture(fonte)
    
    # Força a resolução estipulada no contrato
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a fonte de vídeo: {fonte}")

    prev_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Verifica e avisa sobre queda de FPS
            curr_time = time.time()
            fps_atual = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            
            if fps_atual < 15.0:
                logger.warning(f"FPS baixo detectado: {fps_atual:.1f}. O ideal é > 15 FPS.")
            
            # Garante que o shape é exatamente 480x640, caso a câmera ignore a configuração
            if frame.shape[:2] != (480, 640):
                frame = cv2.resize(frame, (640, 480))

            yield frame
    finally:
        cap.release()
