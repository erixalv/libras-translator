"""
Módulo Pipeline Integrador.
Integra os 4 módulos do sistema (Captura, Landmarks, Modelo DL e Linguagem),
segmentando sinais isolados do fluxo continuo (ver src/app/segmentador.py) e
gerenciando o acúmulo de glosas.
"""

import time
from collections import deque
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from src.app.overlay import desenhar_overlay
from src.app.mocks.stubs import predict as predict_mock, glosas_para_frase as glosas_para_frase_mock
from src.app.segmentador import SegmentadorSinal


class PipelineIntegrador:
    """
    Classe central de integração do Tradutor de Libras.

    Gerencia:
      - Modo Mock vs. Modo Real
      - Segmentação de sinais isolados por presença de mão (SegmentadorSinal,
        modo real) ou janela deslizante simulada (modo mock)
      - Lógica de acúmulo de glosas e gatilhos de tradução
      - Renderização de overlay no frame (OpenCV)
    """

    def __init__(
        self,
        modo_mock: bool = True,
        limiar_confianca: float = 0.60,
        tamanho_janela: int = 30,
        tempo_silencio_seg: float = 2.0,
        max_glosas_buffer: int = 5,
        mostrar_landmarks: bool = True,
    ):
        """
        Args:
            modo_mock: Se True, usa stubs simulados. Se False, usa módulos reais.
            limiar_confianca: Confiança mínima (0.0 a 1.0) para aceitar uma glosa.
            tamanho_janela: Quantidade de frames para a inferência DL (30).
            tempo_silencio_seg: Segundos sem nova glosa para disparar tradução.
            max_glosas_buffer: Máximo de glosas acumuladas antes de disparar tradução.
            mostrar_landmarks: Se True (modo real), desenha pose/mãos detectados no frame --
                debug visual pra conferir se a mão está sendo captada.
        """
        self._modo_mock = modo_mock
        self.limiar_confianca = limiar_confianca
        self.tamanho_janela = tamanho_janela
        self.tempo_silencio_seg = tempo_silencio_seg
        self.max_glosas_buffer = max_glosas_buffer
        self.mostrar_landmarks = mostrar_landmarks

        # Buffers internos
        self.buffer_landmarks = deque(maxlen=tamanho_janela)  # usado so no modo mock
        self.segmentador = SegmentadorSinal()  # usado no modo real -- ver src/app/segmentador.py
        self.buffer_glosas: List[str] = []

        # Estado atual
        self.glosa_atual: str = "NENHUM"
        self.confianca_atual: float = 0.0
        self.top_k_atual: List[Dict[str, Any]] = []
        self.frase_atual: str = ""

        # Timers
        self.tempo_ultima_glosa: float = time.time()
        self.frame_count: int = 0

        # Carrega módulos reais se _modo_mock == False
        self._carregar_modulos_reais()

    @property
    def modo_mock(self) -> bool:
        return self._modo_mock

    @modo_mock.setter
    def modo_mock(self, valor: bool) -> None:
        if self._modo_mock != valor:
            self._modo_mock = valor
            self._carregar_modulos_reais()

    def _carregar_modulos_reais(self) -> None:
        """Carrega dinamicamente os módulos reais se modo_mock=False."""
        if not self._modo_mock:
            try:
                from src.landmarks.extrator_mediapipe import (
                    extrair_landmarks,
                    extrair_landmarks_anotado,
                    construir_sequencia,
                    _obter_extrator,
                )
                from src.modelo.inferencia import predict as predict_real
                from src.linguagem.regras_gramaticais import glosas_para_frase as glosas_para_frase_real

                if _obter_extrator() is None:
                    raise RuntimeError("MediaPipe nao esta disponivel neste ambiente.")

                self._fn_extrair_landmarks = extrair_landmarks
                self._fn_extrair_landmarks_anotado = extrair_landmarks_anotado
                self._fn_construir_sequencia = construir_sequencia
                self._fn_predict = predict_real
                self._fn_glosas_para_frase = glosas_para_frase_real
            except (ImportError, Exception) as e:
                # Se faltar algum módulo real ou mediapipe holistic, faz fallback ou alerta
                print(f"[AVISO PipelineIntegrador] Erro ao carregar módulos reais: {e}. Revertendo para modo_mock=True.")
                self._modo_mock = True
                self._fn_predict = predict_mock
                self._fn_glosas_para_frase = glosas_para_frase_mock
        else:
            self._fn_predict = predict_mock
            self._fn_glosas_para_frase = glosas_para_frase_mock

    def reset(self) -> None:
        """Reseta todos os buffers e estados da aplicação."""
        self.buffer_landmarks.clear()
        self.segmentador.reset()
        self.buffer_glosas.clear()
        self.glosa_atual = "NENHUM"
        self.confianca_atual = 0.0
        self.top_k_atual = []
        self.frase_atual = ""
        self.tempo_ultima_glosa = time.time()
        self.frame_count = 0

    def processar_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processa um frame de vídeo (480, 640, 3):
          1. Extrai landmarks (ou simula no modo mock).
          2. Modo mock: acumula numa janela deslizante de 30 frames.
             Modo real: acumula no SegmentadorSinal, que isola o sinal
             completo pela presenca de mao (ver src/app/segmentador.py) --
             o modelo foi treinado com 1 video = 1 sinal inteiro reamostrado
             pra 30 frames, entao prever numa janela deslizante crua nao
             bate com isso.
          3. Executa o classificador DL quando ha um sinal pronto.
          4. Aplica as regras de acúmulo de glosas.
          5. Renderiza a legenda (overlay) no frame.

        Args:
            frame: Array NumPy BGR (480, 640, 3) uint8.

        Returns:
            Tuple (frame_com_overlay, dict_status)
        """
        self.frame_count += 1
        agora = time.time()
        timestamp_ms = int(agora * 1000)

        if not self.modo_mock and not hasattr(self, "_fn_extrair_landmarks"):
            self._carregar_modulos_reais()

        if self.modo_mock:
            # No modo mock, simula recebimento de landmarks a cada frame e
            # uma predicao a cada 3 frames
            reg_mock = {"frame_id": self.frame_count, "timestamp_ms": timestamp_ms}
            self.buffer_landmarks.append(reg_mock)
            frames_no_buffer = len(self.buffer_landmarks)
            estado_captura = "mock"

            if self.frame_count % 3 == 0:
                res_pred = self._fn_predict(None)
                self.glosa_atual = res_pred.get("gloss", "NENHUM")
                self.confianca_atual = float(res_pred.get("confidence", 0.0))
                self.top_k_atual = res_pred.get("top_k", [])
                self._atualizar_acumulo_glosas(agora)
        else:
            if self.mostrar_landmarks:
                reg, frame = self._fn_extrair_landmarks_anotado(
                    frame, frame_id=self.frame_count, timestamp_ms=timestamp_ms
                )
            else:
                reg = self._fn_extrair_landmarks(frame, frame_id=self.frame_count, timestamp_ms=timestamp_ms)
            frames_no_buffer = self.segmentador.n_frames_atual
            estado_captura = "gravando_sinal" if self.segmentador.gravando else "aguardando"

            if reg is not None:
                segmento = self.segmentador.processar(reg)
                frames_no_buffer = self.segmentador.n_frames_atual
                estado_captura = "gravando_sinal" if self.segmentador.gravando else "aguardando"

                if segmento is not None:
                    seq = self._fn_construir_sequencia(segmento, n_frames=self.tamanho_janela, modo="reamostrar")
                    res_pred = self._fn_predict(seq)
                    self.glosa_atual = res_pred.get("gloss", "NENHUM")
                    self.confianca_atual = float(res_pred.get("confidence", 0.0))
                    self.top_k_atual = res_pred.get("top_k", [])
                    self._atualizar_acumulo_glosas(agora)

        # 3. Checagem de tempo de silêncio 
        # Se passaram 2 segundos sem nova glosa e temos glosas no buffer, dispara tradução
        if self.buffer_glosas and (agora - self.tempo_ultima_glosa >= self.tempo_silencio_seg):
            self._disparar_traducao_frase()

        # 4. Renderização do Overlay
        frame_overlay = desenhar_overlay(
            frame=frame,
            frase=self.frase_atual,
            glosas=self.buffer_glosas,
            glosa_atual=self.glosa_atual,
            confianca=self.confianca_atual
        )

        status = {
            "glosa_atual": self.glosa_atual,
            "confianca_atual": self.confianca_atual,
            "top_k_atual": list(self.top_k_atual),
            "buffer_glosas": list(self.buffer_glosas),
            "frase_atual": self.frase_atual,
            "modo_mock": self.modo_mock,
            "frames_no_buffer": frames_no_buffer,
            "estado_captura": estado_captura,  # "mock" | "aguardando" | "gravando_sinal"
        }

        return frame_overlay, status

    def _atualizar_acumulo_glosas(self, momento_atual: float) -> None:
        """
        Aplica as regras para acúmulo de glosas:
          - Glosa != "NENHUM"
          - Confiança >= limiar
          - Glosa != última glosa aceita no buffer
        """
        if (
            self.glosa_atual != "NENHUM"
            and self.confianca_atual >= self.limiar_confianca
        ):
            # Evita repetição contínua do mesmo sinal
            if not self.buffer_glosas or self.buffer_glosas[-1] != self.glosa_atual:
                self.buffer_glosas.append(self.glosa_atual)
                self.tempo_ultima_glosa = momento_atual

                # Gatilho (a): atingiu limite de glosas no buffer
                if len(self.buffer_glosas) >= self.max_glosas_buffer:
                    self._disparar_traducao_frase()

    def _disparar_traducao_frase(self) -> None:
        """Chama glosas_para_frase() e limpa o buffer de glosas."""
        if not self.buffer_glosas:
            return

        res_frase = self._fn_glosas_para_frase(list(self.buffer_glosas))
        self.frase_atual = res_frase.get("frase", "")
        self.buffer_glosas.clear()
        self.tempo_ultima_glosa = time.time()
