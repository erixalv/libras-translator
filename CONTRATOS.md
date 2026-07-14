
# CONTRATOS.md — Tradutor Automático de Libras → Português

### Documento de referência para o kickoff (Dia 1). Copiar para a raiz do repositório.

---

## 1. Estrutura de Pastas do Repositório

Cada pessoa tem **uma pasta própria** dentro de `src/`. Ninguém edita a pasta de outra pessoa — comunicação só via os contratos abaixo. `app/pipeline_integrador.py` (Pessoa 5) é o único arquivo que importa de todos os módulos, e só é tocado de verdade a partir da Semana 3.

```
libras-translator/
├── README.md
├── CONTRATOS.md                  # este arquivo
├── vocabulario.json              # lista fechada de sinais do MVP (Contrato 0)
├── requirements.txt              # dependências únicas do projeto inteiro
│
├── data/
│   ├── raw/                      # vídeos brutos (dataset público + gravações próprias)
│   ├── processed/                # landmarks já extraídos (.npz/.csv), saída da Pessoa 2
│   └── mocks/                    # arquivos de exemplo de CADA contrato (ver Seção 3)
│       ├── frame_exemplo.mp4
│       ├── landmarks_exemplo.npz
│       ├── predicao_exemplo.json
│       └── frase_exemplo.json
│
├── src/
│   ├── captura/                  # PESSOA 1
│   │   ├── captura.py
│   │   ├── preprocessamento.py
│   │   └── segmentacao_classica.py
│   │
│   ├── landmarks/                # PESSOA 2
│   │   ├── extrator_mediapipe.py
│   │   ├── normalizacao.py
│   │   └── dataset_builder.py
│   │
│   ├── modelo/                   # PESSOA 3
│   │   ├── dataset.py
│   │   ├── arquiteturas.py
│   │   ├── treino.py
│   │   ├── avaliacao.py
│   │   ├── baseline_classico.py
│   │   └── inferencia.py
│   │
│   ├── linguagem/                # PESSOA 4
│   │   ├── regras_gramaticais.py
│   │   ├── modelo_seq2seq.py     # opcional, se houver tempo
│   │   └── corpus_glosa_frase.json
│   │
│   └── app/                      # PESSOA 5
│       ├── main.py               # entrada Streamlit/Flask
│       ├── overlay.py
│       ├── pipeline_integrador.py
│       └── mocks/                # stubs que simulam os outros 4 módulos
│
├── notebooks/                    # exploração livre, 1 subpasta por pessoa
├── tests/                        # 1 arquivo de teste por módulo
├── scripts/
│   └── setup_ambiente.sh
└── docs/
    ├── relatorio/                # fonte LaTeX/Overleaf
    └── figuras/
```

**Regra de ouro**: se você precisa importar código de outra pasta que não seja via a função de contrato documentada abaixo, pare — isso é sinal de que o contrato está mal desenhado ou incompleto.

---

## 2. Contrato 0 — Vocabulário Fechado

Arquivo `vocabulario.json` na raiz, travado no kickoff e alterado só por consenso da equipe (mudar depois de todos começarem quebra os Contratos B, C e D ao mesmo tempo):

```json
{
  "versao": 1,
  "sinais": [
    "EU", "VOCE", "ELE_ELA", "OI", "TCHAU", "OBRIGADO", "POR_FAVOR",
    "DESCULPA", "SIM", "NAO", "NOME", "QUERER", "PRECISAR", "AJUDA",
    "AGUA", "COMIDA", "BANHEIRO", "CASA", "TRABALHO", "ESTUDAR",
    "GOSTAR", "BOM", "RUIM", "GRANDE", "PEQUENO", "HOJE", "AMANHA",
    "ONTEM", "DINHEIRO", "FAMILIA", "MAE", "PAI", "AMIGO", "ESCOLA",
    "PROFESSOR", "ALUNO", "LIVRO", "ESCREVER", "LER", "FALAR",
    "ENTENDER", "FELIZ", "TRISTE", "SAUDE", "MEDICO", "TELEFONE",
    "COMER", "BEBER", "DORMIR", "DESCULPA_2"
  ]
}
```

*(Ajustem a lista final considerando quais sinais têm exemplos suficientes na MINDS-Libras/V-LIBRASIL — melhor reduzir para ~30 sinais com bons dados do que ter 50 mal representados.)*

---

## 3. Contratos entre Módulos

### Contrato A — Frame de Vídeo (Pessoa 1 → Pessoa 2)

**Formato:** `numpy.ndarray`, shape `(480, 640, 3)`, dtype `uint8`, BGR (padrão OpenCV).

**Função de interface (assinatura fixa que Pessoa 1 deve implementar):**

```python
# src/captura/captura.py
from typing import Iterator
import numpy as np

def capturar_frames(fonte: str | int = 0) -> Iterator[np.ndarray]:
    """
    fonte: 0 (webcam padrão) ou caminho de arquivo .mp4
    yield: frame (480, 640, 3) BGR uint8, um por vez
    """
```

**Mock obrigatório (entregar Dia 1):** `data/mocks/frame_exemplo.mp4`, 5–10s, resolução 640×480, para todos testarem sem precisar de webcam.

---

### Contrato B — Sequência de Landmarks (Pessoa 2 → Pessoa 3)

**Formato por frame:**

```json
{
  "frame_id": 42,
  "timestamp_ms": 1400,
  "pose": [[x, y, z, visibility], "... 33 pontos"],
  "left_hand": [[x, y, z], "... 21 pontos, ou null se não detectada"],
  "right_hand": [[x, y, z], "... 21 pontos, ou null se não detectada"]
}
```

**Regras fechadas no kickoff:**

- Coordenadas normalizadas em relação ao centro dos ombros (não pixels absolutos) — invariância a posição/escala do sinalizador.
- Mão ausente → `null` explícito (não zeros — zero é uma posição válida e confunde o modelo).
- Sequência de treino = janela fixa de **30 frames**, stride de 15 (50% overlap).

**Função de interface:**

```python
# src/landmarks/extrator_mediapipe.py
import numpy as np

def extrair_landmarks(frame: np.ndarray) -> dict:
    """Recebe 1 frame (Contrato A), retorna 1 dict no formato do Contrato B."""

def construir_sequencia(landmarks_por_frame: list[dict]) -> np.ndarray:
    """Recebe lista de dicts, retorna array (30, N_FEATURES) pronto pro modelo."""
```

**Mock obrigatório:** `data/mocks/landmarks_exemplo.npz` com 1 sequência de 30 frames (pode ser gerada com dados aleatórios no formato certo — não precisa ser um sinal real ainda).

---

### Contrato C — Predição do Classificador (Pessoa 3 → Pessoas 4 e 5)

```json
{
  "gloss": "OBRIGADO",
  "confidence": 0.87,
  "timestamp_ms": 1400
}
```

**Regra:** `gloss` é sempre um item de `vocabulario.json` (Contrato 0) ou o valor especial `"NENHUM"` (quando confiança abaixo de um limiar, ex. 0.5 — evita legendas piscando com ruído).

**Função de interface:**

```python
# src/modelo/inferencia.py
import numpy as np

def predict(sequencia: np.ndarray) -> dict:
    """
    sequencia: array (30, N_FEATURES), saída do Contrato B
    retorna: dict no formato do Contrato C
    """
```

**Mock obrigatório (Dia 1, antes do modelo treinado existir):** versão "burra" de `predict()` que sorteia uma glosa aleatória de `vocabulario.json` com confiança fixa — assim Pessoas 4 e 5 já plugam contra algo real desde o início.

---

### Contrato D — Frase Final (Pessoa 4 → Pessoa 5)

```json
{
  "glosas_recebidas": ["EU", "QUERER", "AGUA"],
  "frase": "Eu quero água."
}
```

**Função de interface:**

```python
# src/linguagem/regras_gramaticais.py

def glosas_para_frase(glosas: list[str]) -> dict:
    """Recebe lista de glosas acumuladas (Contrato C), retorna dict do Contrato D."""
```

**Mock obrigatório:** `data/mocks/frase_exemplo.json`, com 3–5 exemplos de entrada/saída para Pessoa 5 testar o overlay sem depender do modelo de linguagem pronto.

---

## 4. O Que Cada Pessoa Entrega — Detalhado

### Pessoa 1 — Captura e PDI Clássico

| Arquivo                                 | Conteúdo                                                                                                                                                                                    |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/captura/captura.py`              | Implementa`capturar_frames()` (Contrato A)                                                                                                                                                 |
| `src/captura/preprocessamento.py`     | Normalização de iluminação, equalização de histograma, redução de ruído                                                                                                             |
| `src/captura/segmentacao_classica.py` | Baseline PDI puro: segmentação HSV/YCbCr de pele + contornos/morfologia —**não usa MediaPipe**, é o material de fundamentação teórica de PDI clássico exigido pela disciplina |
| `data/mocks/frame_exemplo.mp4`        | Vídeo de teste para todos                                                                                                                                                                   |
| `tests/test_captura.py`               | Testa shape/dtype do frame retornado                                                                                                                                                         |

Trabalha 100% isolado desde o Dia 1: só precisa de uma webcam ou de vídeos de exemplo baixados da internet.

### Pessoa 2 — Landmarks e Dataset

| Arquivo                                 | Conteúdo                                                                                                              |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `src/landmarks/extrator_mediapipe.py` | Implementa`extrair_landmarks()` e `construir_sequencia()` (Contrato B)                                             |
| `src/landmarks/normalizacao.py`       | Normalização espacial (relativa aos ombros) e tratamento de mãos ausentes                                           |
| `src/landmarks/dataset_builder.py`    | Baixa/organiza MINDS-Libras e/ou V-LIBRASIL, filtra pelos sinais de`vocabulario.json`, gera `data/processed/*.npz` |
| `data/mocks/landmarks_exemplo.npz`    | Sequência sintética de 30 frames no formato certo                                                                    |
| `tests/test_landmarks.py`             | Testa shape/formato da sequência                                                                                      |

Trabalha com vídeos próprios (não precisa dos vídeos reais da Pessoa 1) e com os datasets públicos — pode começar a extrair landmarks de vídeos baixados de exemplo desde o Dia 1.

### Pessoa 3 — Modelagem e Treinamento

| Arquivo                             | Conteúdo                                                                                                 |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `src/modelo/dataset.py`           | `Dataset`/`DataLoader` que lê os `.npz` de `data/processed/` (ou do mock, no início)            |
| `src/modelo/arquiteturas.py`      | Definição LSTM/GRU (e ST-GCN/Transformer se houver tempo)                                               |
| `src/modelo/treino.py`            | Loop de treino, checkpoints                                                                               |
| `src/modelo/avaliacao.py`         | Acurácia, matriz de confusão, F1 por classe                                                             |
| `src/modelo/baseline_classico.py` | SVM/Random Forest (scikit-learn) sobre features simplificadas — comparação no relatório               |
| `src/modelo/inferencia.py`        | Implementa`predict()` (Contrato C) — versão mock (aleatória) no Dia 1, versão real depois do treino |

Treina primeiro com dados sintéticos gerados por script próprio (mesmo formato do Contrato B), troca pelo dataset real da Pessoa 2 assim que ele existir, sem mudar uma linha do `treino.py`.

### Pessoa 4 — Linguística Computacional

| Arquivo                                   | Conteúdo                                                                                         |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `src/linguagem/regras_gramaticais.py`   | Implementa`glosas_para_frase()` (Contrato D) via regras (reordenação, artigos, preposições) |
| `src/linguagem/modelo_seq2seq.py`       | Opcional: fine-tuning leve de modelo pequeno (ex. T5) se houver tempo na Semana 3/4               |
| `src/linguagem/corpus_glosa_frase.json` | Pares glosa→frase construídos manualmente pela equipe para treinar/validar as regras            |
| `data/mocks/frase_exemplo.json`         | Exemplos de entrada/saída                                                                        |

Trabalha só com texto — testa digitando listas de glosas manualmente, nunca precisa rodar vídeo, MediaPipe ou o modelo de deep learning.

### Pessoa 5 — Interface, Integração e Documentação

| Arquivo                            | Conteúdo                                                                                         |
| ---------------------------------- | ------------------------------------------------------------------------------------------------- |
| `src/app/main.py`                | App Streamlit/Flask com vídeo ao vivo                                                            |
| `src/app/overlay.py`             | Desenho da legenda sobre o frame                                                                  |
| `src/app/mocks/`                 | Stubs de`predict()` e `glosas_para_frase()` para desenvolver a UI sem depender de ninguém    |
| `src/app/pipeline_integrador.py` | Só a partir da Semana 3: importa de verdade`captura`, `landmarks`, `modelo`, `linguagem` |
| `README.md`, `docs/`           | Documentação, organização do relatório final, vídeo de demonstração                       |

Constrói a aplicação inteira contra os próprios mocks desde a Semana 1; a integração real na Semana 3/4 é literalmente trocar os imports dos mocks pelos módulos de verdade, já que todos respeitam os mesmos contratos.

---

## 5. `requirements.txt` Sugerido

```
opencv-python==4.10.*
mediapipe==0.10.*
numpy
pandas
scikit-learn
torch                 # ou tensorflow — escolher UM no kickoff
streamlit
matplotlib
```

---

## 6. Critério de "Pronto" por Módulo (definir junto no kickoff)

| Módulo        | Critério objetivo de aceite                                                                                                        |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Captura (P1)   | `capturar_frames()` roda ≥25 FPS e passa em `test_captura.py`                                                                  |
| Landmarks (P2) | `extrair_landmarks()` detecta mãos em ≥90% dos frames de teste; dataset processado cobre todos os itens de `vocabulario.json` |
| Modelo (P3)    | `predict()` responde em <100ms; acurácia mínima combinada (ex. ≥70%) no conjunto de validação                                |
| Linguagem (P4) | `glosas_para_frase()` cobre 100% dos casos do `corpus_glosa_frase.json` sem erro                                                |
| App (P5)       | Pipeline ponta a ponta roda sem crash por ≥2 minutos contínuos com webcam real                                                    |
