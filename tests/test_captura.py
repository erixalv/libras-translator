import pytest
import numpy as np
from src.captura.captura import capturar_frames
from unittest.mock import patch, MagicMock

@patch('cv2.VideoCapture')
def test_capturar_frames(mock_vidcap):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    
    # Simular o retorno de um frame falso da câmera
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Lança duas vezes o frame pra simular que leu algo e depois acaba
    mock_cap.read.side_effect = [(True, fake_frame), (False, None)]
    
    mock_vidcap.return_value = mock_cap
    
    frames = list(capturar_frames('dummy'))
    
    assert len(frames) == 1
    assert frames[0].shape == (480, 640, 3)
    assert frames[0].dtype == np.uint8
