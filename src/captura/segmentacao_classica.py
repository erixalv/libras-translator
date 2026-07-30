import cv2
import numpy as np

def segmentar_pele_ycbcr(frame: np.ndarray) -> np.ndarray:
    """Segmenta a pele no espaço YCbCr seguindo faixas fixas."""
    ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    
    # Faixas definidas no contrato (Cr em [135,180], Cb em [85,135])
    lower = np.array([0, 135, 85], dtype=np.uint8)
    upper = np.array([255, 180, 135], dtype=np.uint8)
    
    mascara = cv2.inRange(ycbcr, lower, upper)
    
    # Operações morfológicas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=1)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return mascara

def extrair_contornos(mascara: np.ndarray):
    """Encontra os contornos na máscara segmentada."""
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contornos
