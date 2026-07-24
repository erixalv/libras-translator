from src.modelo.inferencia import predict
import numpy as np


def test_predict_retorna_formato_esperado():
    seq = np.random.randn(30, 258).astype(np.float32)
    resultado = predict(seq)
    assert "gloss" in resultado
    assert "confidence" in resultado
    assert "timestamp_ms" in resultado
    assert 0.0 <= resultado["confidence"] <= 1.0
