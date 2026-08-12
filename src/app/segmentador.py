"""
Segmentacao de sinais isolados a partir do fluxo continuo de landmarks,
pro modo real do app ao vivo.

O modelo foi treinado com 1 video = 1 sinal completo, comprimido pra 30
frames (construir_sequencia(..., modo="reamostrar")). Prever direto numa
janela deslizante de frames crus (modo="ultima_janela") nao bate com isso --
a janela raramente contem exatamente 1 sinal do inicio ao fim (a maior parte
do tempo pega transicao entre sinais, mao em repouso ou so um pedaco do
gesto), entao a maioria das previsoes ao vivo sai ruidosa mesmo em sinais
bem representados no treino.

SegmentadorSinal resolve isso detectando INICIO e FIM de cada sinal pela
presenca de mao (mask_esq/mask_dir do Contrato B): a mao aparece -> comeca a
capturar; a mao some por `tempo_confirmacao_fim_seg` segundos seguidos -> o
sinal acabou, devolve os frames daquele sinal prontos pra passar em
construir_sequencia(..., modo="reamostrar") -- exatamente como no treino.

MARGEM: os videos originais de treino (V-LIBRASIL/MINDS/gravacao propria)
tem bastante "sobra" parada de mao antes/depois do gesto em si (~50-70% do
clipe, ver historico do projeto) -- reamostrar() comprime isso tudo junto
com o gesto real. Cortar bem rente ao gesto (sem sobra nenhuma) da uma
sequencia reamostrada com "formato" temporal diferente do que o modelo
aprendeu, e isso mostrou na pratica um viés forte pra certas classes
("Água"/"Ajudar" absorvendo previsoes de varias outras palavras). Por isso
o segmentador guarda uma margem de frames parados antes do inicio e depois
do fim do gesto, em vez de cortar exatamente na borda.
"""

from collections import deque
from typing import Optional


def _mao_detectada(reg: dict) -> bool:
    return reg.get("left_hand") is not None or reg.get("right_hand") is not None


class SegmentadorSinal:
    """
    Maquina de estados PARADO <-> GRAVANDO baseada em presenca de mao.

    processar(reg) devolve None enquanto o sinal ainda esta em andamento (ou
    ninguem esta sinalizando), e devolve list[dict] quando um sinal completo
    acabou de ser capturado.
    """

    def __init__(
        self,
        tempo_confirmacao_fim_seg: float = 0.4,
        min_frames_sinal: int = 8,
        max_tempo_segmento_seg: float = 4.0,
        margem_frames: int = 8,
    ):
        self.tempo_confirmacao_fim_seg = tempo_confirmacao_fim_seg
        self.min_frames_sinal = min_frames_sinal
        self.max_tempo_segmento_seg = max_tempo_segmento_seg
        self.margem_frames = margem_frames

        self._frames: list[dict] = []
        self._pre_buffer: deque = deque(maxlen=margem_frames)
        self._indice_ultima_deteccao: int = -1
        self._tempo_inicio: float = 0.0
        self._tempo_ultima_deteccao: float = 0.0

    @property
    def gravando(self) -> bool:
        return bool(self._frames)

    @property
    def n_frames_atual(self) -> int:
        return len(self._frames)

    def processar(self, reg: dict) -> Optional[list[dict]]:
        agora = reg["timestamp_ms"] / 1000.0
        mao_ok = _mao_detectada(reg)

        if not self._frames:
            if not mao_ok:
                # guarda uns frames parados recentes -- viram margem de
                # "antes do gesto" se um sinal comecar em seguida
                self._pre_buffer.append(reg)
                return None
            self._frames = list(self._pre_buffer) + [reg]
            self._pre_buffer.clear()
            self._indice_ultima_deteccao = len(self._frames) - 1
            self._tempo_inicio = agora
            self._tempo_ultima_deteccao = agora
            return None

        self._frames.append(reg)
        if mao_ok:
            self._indice_ultima_deteccao = len(self._frames) - 1
            self._tempo_ultima_deteccao = agora

        tempo_sem_mao = agora - self._tempo_ultima_deteccao
        tempo_total = agora - self._tempo_inicio
        if tempo_sem_mao >= self.tempo_confirmacao_fim_seg or tempo_total >= self.max_tempo_segmento_seg:
            return self._finalizar()

        return None

    def _finalizar(self) -> Optional[list[dict]]:
        # mantem ate `margem_frames` de sobra parada depois da ultima
        # deteccao real de mao (em vez de cortar bem na borda)
        limite = min(self._indice_ultima_deteccao + self.margem_frames, len(self._frames) - 1)
        segmento = self._frames[: limite + 1]
        self.reset()

        if len(segmento) >= self.min_frames_sinal:
            return segmento
        return None

    def reset(self) -> None:
        self._frames = []
        self._pre_buffer.clear()
        self._indice_ultima_deteccao = -1
