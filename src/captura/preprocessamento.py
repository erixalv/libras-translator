import cv2
import numpy as np

def aplicar_clahe(frame: np.ndarray) -> np.ndarray:
    """Aplica CLAHE no canal V do espaço HSV para normalizar iluminação."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # clipLimit=2.0 e tileGridSize=(8,8) exigidos pelo projeto
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v_clahe = clahe.apply(v)
    
    hsv_clahe = cv2.merge((h, s, v_clahe))
    return cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)

def reduzir_ruido(frame: np.ndarray) -> np.ndarray:
    """Aplica filtro Non-Local Means para reduzir ruído."""
    return cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

def preprocessar(frame: np.ndarray) -> np.ndarray:
    """Pipeline completo de pré-processamento clássico."""
    frame_suave = reduzir_ruido(frame)
    frame_iluminado = aplicar_clahe(frame_suave)
    return frame_iluminado
