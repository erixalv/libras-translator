"""
Testes unitários para o módulo App:
  - Overlay (desenho de legenda OpenCV)
  - Mocks / Stubs (predict e glosas_para_frase)
  - PipelineIntegrador em modo mock
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Adiciona o diretório raiz ao path para importação limpa dos pacotes
RAIZ = str(Path(__file__).resolve().parents[1])
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

from src.app.overlay import desenhar_overlay, _remover_acentos
from src.app.mocks.stubs import predict as predict_mock, glosas_para_frase as glosas_para_frase_mock, carregar_vocabulario_mock
from src.app.pipeline_integrador import PipelineIntegrador


# ==============================================================================
# TESTES DE MOCKS / STUBS (Contratos C e D)
# ==============================================================================

def test_carregar_vocabulario_mock():
    """Garante que o vocabulário mock retorna uma lista não vazia de strings."""
    vocab = carregar_vocabulario_mock()
    assert isinstance(vocab, list)
    assert len(vocab) > 0
    assert all(isinstance(s, str) for s in vocab)


def test_predict_mock_formato_contrato_c():
    """Garante que o predict mock respeita rigorosamente a estrutura do Contrato C."""
    res = predict_mock()
    assert isinstance(res, dict)
    assert "gloss" in res
    assert "confidence" in res
    assert "timestamp_ms" in res
    assert isinstance(res["gloss"], str)
    assert isinstance(res["confidence"], float)
    assert isinstance(res["timestamp_ms"], int)
    assert 0.0 <= res["confidence"] <= 1.0


def test_glosas_para_frase_mock_formato_contrato_d():
    """Garante que o glosas_para_frase mock respeita a estrutura do Contrato D."""
    glosas_entrada = ["EU", "QUERER", "AGUA"]
    res = glosas_para_frase_mock(glosas_entrada)
    assert isinstance(res, dict)
    assert "glosas_recebidas" in res
    assert "frase" in res
    assert res["glosas_recebidas"] == glosas_entrada
    assert res["frase"] == "Eu quero água."


def test_glosas_para_frase_mock_fallback():
    """Testa o fallback para uma sequência não mapeada no dicionário."""
    glosas_entrada = ["HOJE", "ESTUDAR", "CASA"]
    res = glosas_para_frase_mock(glosas_entrada)
    assert isinstance(res, dict)
    assert res["glosas_recebidas"] == glosas_entrada
    assert res["frase"].startswith("Hoje")
    assert res["frase"].endswith(".")


# ==============================================================================
# TESTES DE OVERLAY (DESENHO DE LEGENDA OPENCV)
# ==============================================================================

def test_remover_acentos():
    """Testa a remoção de acentos para compatibilidade com cv2.putText."""
    assert _remover_acentos("Água, você, pão & atenção!") == "Agua, voce, pao & atencao!"
    assert _remover_acentos("") == ""


def test_desenhar_overlay_shape_e_dtype():
    """Garante que o overlay preserva o shape (480, 640, 3) e o dtype uint8."""
    frame_original = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_overlay = desenhar_overlay(
        frame=frame_original,
        frase="Eu quero água.",
        glosas=["EU", "QUERER", "AGUA"],
        glosa_atual="AGUA",
        confianca=0.88
    )

    assert isinstance(frame_overlay, np.ndarray)
    assert frame_overlay.shape == (480, 640, 3)
    assert frame_overlay.dtype == np.uint8
    # O frame alterado não deve ser idêntico ao frame totalmente preto
    assert not np.array_equal(frame_original, frame_overlay)


def test_desenhar_overlay_frame_invalido():
    """Garante que o overlay levanta ValueError se receber frame inválido."""
    with pytest.raises(ValueError):
        desenhar_overlay(frame=None)

    with pytest.raises(ValueError):
        desenhar_overlay(frame=np.zeros((100, 100), dtype=np.uint8))  # 2D ao invés de 3D


# ==============================================================================
# TESTES DO PIPELINE INTEGRADOR (MODO MOCK)
# ==============================================================================

def test_pipeline_integrador_inicializacao():
    """Testa a inicialização e o estado default do PipelineIntegrador."""
    pipeline = PipelineIntegrador(modo_mock=True, limiar_confianca=0.60)
    assert pipeline.modo_mock is True
    assert pipeline.limiar_confianca == 0.60
    assert pipeline.frase_atual == ""
    assert pipeline.glosa_atual == "NENHUM"
    assert len(pipeline.buffer_glosas) == 0


def test_pipeline_integrador_processar_frame():
    """Testa o processamento de um frame de vídeo em modo mock."""
    pipeline = PipelineIntegrador(modo_mock=True)
    frame_teste = np.zeros((480, 640, 3), dtype=np.uint8)

    frame_saida, status = pipeline.processar_frame(frame_teste)

    assert isinstance(frame_saida, np.ndarray)
    assert frame_saida.shape == (480, 640, 3)
    assert isinstance(status, dict)
    assert "glosa_atual" in status
    assert "confianca_atual" in status
    assert "buffer_glosas" in status
    assert "frase_atual" in status
    assert "modo_mock" in status
    assert status["modo_mock"] is True


def test_pipeline_integrador_reset():
    """Garante que o método reset limpa corretamente os buffers e estado."""
    pipeline = PipelineIntegrador(modo_mock=True)
    frame_teste = np.zeros((480, 640, 3), dtype=np.uint8)

    # Processa múltiplos frames para preencher buffers
    for _ in range(10):
        pipeline.processar_frame(frame_teste)

    pipeline.reset()

    assert len(pipeline.buffer_landmarks) == 0
    assert len(pipeline.buffer_glosas) == 0
    assert pipeline.glosa_atual == "NENHUM"
    assert pipeline.confianca_atual == 0.0
    assert pipeline.frase_atual == ""
    assert pipeline.frame_count == 0


def test_pipeline_integrador_acumulo_glosas_e_disparo():
    """Testa a inserção direta no buffer de glosas e o disparo da tradução."""
    pipeline = PipelineIntegrador(modo_mock=True, max_glosas_buffer=3)
    
    # Injeta glosas no buffer e simula disparo
    pipeline.buffer_glosas = ["EU", "QUERER", "AGUA"]
    pipeline._disparar_traducao_frase()

    assert pipeline.frase_atual == "Eu quero água."
    assert len(pipeline.buffer_glosas) == 0


def test_pipeline_integrador_alternar_modo_mock():
    """Garante que alternar modo_mock dinamicamente não gera AttributeError em processar_frame."""
    pipeline = PipelineIntegrador(modo_mock=True)
    frame_teste = np.zeros((480, 640, 3), dtype=np.uint8)

    # Tenta desativar o modo mock dinamicamente
    pipeline.modo_mock = False
    
    # O processamento de frame deve rodar sem exceção de atributo não encontrado
    frame_saida, status = pipeline.processar_frame(frame_teste)
    assert isinstance(frame_saida, np.ndarray)
    assert isinstance(status, dict)
