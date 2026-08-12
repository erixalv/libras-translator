"""
Classificador LSTM bidirecional para landmarks de mao (LIBRAS)
  - LSTM bidirecional, 2 camadas (arquitetura travada na Secao 3.5 do
    CONTRATOS.md), hidden_size=48 e dropout mais forte -- reduzidos em
    relacao ao hidden_size=128/dropout=0.3 originais depois de medir
    overfitting severo (~99% treino vs ~35-40% em sinalizador nunca visto,
    com apenas 12 sinalizadores distintos no dataset -- ver discussao do
    dia em que isso foi diagnosticado). Continua sendo uma LSTM bidirecional
    de 2 camadas, so com capacidade calibrada pro tamanho real do dataset.
  - Dropout adicional antes da camada linear de saida (regularizacao extra)
  - Linear(hidden_size*2, n_classes) na saida (bidirecional)
  - leitura da sequencia por ATENCAO (usar_atencao=True, padrao) em vez de
    so pegar o ultimo estado oculto: o ultimo estado joga fora a informacao
    dos outros 29 frames, forcando a LSTM a "resumir tudo" numa unica saida
    final. Atencao deixa o modelo aprender qual(is) frame(s) do gesto
    pesam mais pra cada sinal, sem sair da familia LSTM (nao muda o nucleo
    recorrente travado no CONTRATOS.md, so o jeito de ler a saida dele)
"""
import torch
import torch.nn as nn


class ClassificadorLSTM(nn.Module):
    def __init__(
        self,
        n_features: int = 260,
        hidden_size: int = 48,
        n_layers: int = 2,
        dropout: float = 0.4,
        dropout_saida: float = 0.4,
        n_classes: int = 10,
        usar_atencao: bool = True,
    ):
        super().__init__()
        self.usar_atencao = usar_atencao
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        if usar_atencao:
            self.atencao = nn.Linear(hidden_size * 2, 1)
        self.dropout_saida = nn.Dropout(dropout_saida)
        self.classificador = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        saida_lstm, (h_n, _) = self.lstm(x)  # saida_lstm: (batch, T, hidden*2)

        if self.usar_atencao:
            pesos = torch.softmax(self.atencao(saida_lstm).squeeze(-1), dim=1)  # (batch, T)
            representacao = torch.sum(saida_lstm * pesos.unsqueeze(-1), dim=1)  # (batch, hidden*2)
        else:
            ultima_fwd = h_n[-2]  # ultima camada, direcao forward
            ultima_bwd = h_n[-1]  # ultima camada, direcao backward
            representacao = torch.cat([ultima_fwd, ultima_bwd], dim=1)

        representacao = self.dropout_saida(representacao)
        return self.classificador(representacao)
