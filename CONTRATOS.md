# CONTRATOS.md — Tradutor Automático de Libras → Português

### Documento de referência para o kickoff (Dia 1). Copiar para a raiz do repositório.

> **v2 — atualizado**: além dos formatos de dado entre módulos (Seção 3), este documento agora também trava as **decisões internas de arquitetura e treino do modelo de deep learning** (Seção 3.5), definidas pela Pessoa 3. Isso não é mais algo a discutir no kickoff — é ponto de partida. Framework: **PyTorch** (fechado).

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
│       ├── main.py               # entrada Streamlit (framework travado)
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

**Parâmetros de captura travados:**

- `cv2.VideoCapture`, resolução forçada via `CAP_PROP_FRAME_WIDTH=640` / `CAP_PROP_FRAME_HEIGHT=480`.
- **FPS alvo: 30**, com fallback aceitável até 15 (webcams mais fracas) — o sistema deve funcionar em ambos, já que a janela de 30 frames (Contrato B) é definida em quantidade de frames, não em tempo fixo. Se FPS cair muito abaixo de 15, registrar aviso no log, não travar a aplicação.

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

**Regras fechadas:**

- Coordenadas normalizadas em relação ao centro dos ombros (não pixels absolutos) — invariância a posição/escala do sinalizador.
- Mão ausente → `null` explícito neste dicionário bruto (facilita debug visual). **Atenção**: na hora de virar array pro modelo (`construir_sequencia()`), a Pessoa 3 substitui `null` por zeros **depois** de normalizar — ver Seção 3.5. Ou seja, Pessoa 2 nunca entrega zero aqui; quem decide o que fazer com o vazio é o consumidor (Pessoa 3), não a extração.
- Sequência de treino = janela fixa de **30 frames**, stride de 15 (50% overlap).
- **`N_FEATURES = 258`** (33 pontos de pose × 4 valores [x,y,z,visibility] + 21 pontos por mão × 3 valores [x,y,z] × 2 mãos). Este número é fixo em todo o projeto — se a Pessoa 2 decidir incluir/excluir algum ponto, avisar a Pessoa 3 imediatamente, pois muda a camada de entrada do modelo.
- **Parâmetros do MediaPipe travados**: `min_detection_confidence=0.5`, `min_tracking_confidence=0.5`, `model_complexity=1` (equilíbrio padrão entre velocidade e precisão — só subir pra `2` se sobrar tempo e a Pessoa 5 identificar que a taxa de FPS aguenta).

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

**Regra:** `gloss` é sempre um item de `vocabulario.json` (Contrato 0) ou o valor especial `"NENHUM"` (quando confiança abaixo de **0.6** — valor travado por P3, evita legendas piscando com ruído; pode ser recalibrado depois com dado real, mas o comportamento — existir um threshold e um valor `"NENHUM"` — é fixo).

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

## 3.5. Decisões Técnicas do Modelo (Travadas por P3 — não é mais assunto de reunião)

Estas decisões ficam **dentro** da "caixa-preta" da Pessoa 3 — ninguém mais no grupo precisa opinar ou aprovar, só saber que existem, porque duas delas vazam pro resto do time através dos contratos (marcadas abaixo com **[afeta outros módulos]**).

**Framework**

- **PyTorch** — fechado. **[afeta P5]**: o `requirements.txt` do projeto usa `torch`, não `tensorflow`; o `pipeline_integrador.py` carrega o modelo via `torch.load`.

**Arquitetura**

- LSTM **bidirecional**, 2 camadas, `hidden_size=128`, `dropout=0.3` entre camadas, seguida de `Linear(256, n_classes)` (256 = 128×2 por ser bidirecional).
- `n_classes` é lido dinamicamente do tamanho de `vocabulario.json` — muda automaticamente se a lista de sinais mudar, sem precisar editar código.

**Pré-processamento de features (dentro do módulo `modelo/`)**

- `StandardScaler` (scikit-learn) ajustado no conjunto de treino, salvo em `data/processed/scaler.pkl` junto com o checkpoint do modelo.
- **[afeta P5]**: na hora de integrar, a inferência ao vivo precisa aplicar o **mesmo** `scaler.pkl` antes de chamar `predict()` — isso já vem embutido dentro da própria função `predict()` em `inferencia.py`, então P5 não precisa se preocupar, só não pode chamar o modelo "por fora" da função de contrato.
- Preenchimento de mão ausente (`null` do Contrato B) → zero, aplicado **depois** da normalização, dentro de `construir_sequencia()`.

**Treino**

- Loss: `CrossEntropyLoss` · Otimizador: `Adam` (`lr=1e-3`, `weight_decay=1e-5`) · Batch size: 16
- Até 100 épocas, com *early stopping* (para se validação não melhorar por 10 épocas)
- Split 80/20 treino/validação, estratificado por classe
- Checkpoint do melhor modelo salvo em `data/processed/modelo_melhor.pt`

**Inferência**

- Threshold de confiança: **0.6** (ver Contrato C acima)
- Tempo de resposta alvo: <100ms por chamada de `predict()`

**Por que isso importa pro resto do grupo**: como essas decisões já estão fechadas, ninguém precisa esperar uma reunião pra P3 começar a treinar — e o restante do grupo só precisa saber os 2 pontos marcados **[afeta outros módulos]** acima. Tudo o mais é implementação interna do módulo `modelo/`.

---

## 3.6. Contrato C.5 — Lógica de Acúmulo de Glosas (P5, entre Contrato C e D)

**Isso estava em aberto e precisava ser decidido**: quando exatamente o sistema chama `glosas_para_frase()`? Se for a cada predição isolada, a "frase" nunca passa de 1 palavra. Regra travada:

- P5 mantém um buffer de glosas em memória (lista Python simples).
- A cada chamada de `predict()` com `gloss != "NENHUM"` e diferente da última glosa aceita (evita repetir a mesma glosa várias vezes por estar parada), adiciona ao buffer.
- **Gatilho de disparo**: chama `glosas_para_frase(buffer)` quando ocorrer **qualquer um** dos dois eventos:
  - (a) o buffer atingir **5 glosas**, ou
  - (b) se passarem **2 segundos consecutivos** sem nenhuma nova glosa aceita (silêncio = fim da frase).
- Após o disparo, o buffer é limpo e a legenda exibida é substituída pela `frase` retornada.
- Enquanto o buffer ainda está sendo montado (antes do disparo), a interface pode mostrar as glosas cruas acumuladas como feedback visual (ex.: "EU... QUERER..."), decisão de UX da Pessoa 5.

Isso fecha um buraco que existia entre os Contratos C e D — sem essa regra, a Pessoa 5 não saberia quando "fechar" uma frase.

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

| Arquivo                                 | Conteúdo                                                                                                                                                                                                                                                                                                                                                                                |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/captura/captura.py`              | Implementa`capturar_frames()` (Contrato A)                                                                                                                                                                                                                                                                                                                                             |
| `src/captura/preprocessamento.py`     | Normalização de iluminação via**CLAHE** (`cv2.createCLAHE`, `clipLimit=2.0`, `tileGridSize=(8,8)`) aplicado no canal V do HSV — melhor que equalização global pra vídeo com iluminação desigual; redução de ruído com `cv2.fastNlMeansDenoisingColored`                                                                                                       |
| `src/captura/segmentacao_classica.py` | Baseline PDI puro: segmentação de pele em**YCbCr** com faixa travada `Cr∈[135,180]`, `Cb∈[85,135]` (mais robusta a variação de iluminação que HSV puro) + operações morfológicas de abertura/fechamento (kernel 5×5) + detecção de contornos — **não usa MediaPipe**, é o material de fundamentação teórica de PDI clássico exigido pela disciplina |
| `data/mocks/frame_exemplo.mp4`        | Vídeo de teste para todos                                                                                                                                                                                                                                                                                                                                                               |
| `tests/test_captura.py`               | Testa shape/dtype do frame retornado                                                                                                                                                                                                                                                                                                                                                     |

Trabalha 100% isolado desde o Dia 1: só precisa de uma webcam ou de vídeos de exemplo baixados da internet.

### Pessoa 2 — Landmarks e Dataset

| Arquivo                                 | Conteúdo                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `src/landmarks/extrator_mediapipe.py` | Implementa`extrair_landmarks()` e `construir_sequencia()` (Contrato B)                                                                                                                                                                                                                                                                                                                                                                                   |
| `src/landmarks/normalizacao.py`       | Normalização espacial (relativa aos ombros) e tratamento de mãos ausentes                                                                                                                                                                                                                                                                                                                                                                                 |
| `src/landmarks/dataset_builder.py`    | Baixa/organiza**MINDS-Libras + V-LIBRASIL combinadas** (decisão travada: usar as duas, para maximizar cobertura de sinais e sinalizadores), checa cobertura real de cada sinal (**mínimo travado: 3 amostras/sinalizadores diferentes por sinal**, senão o sinal é descartado do vocabulário final), atualiza `vocabulario.json` com a lista definitiva, filtra e extrai os vídeos pelos sinais aprovados, gera `data/processed/*.npz` |
| `data/mocks/landmarks_exemplo.npz`    | Sequência sintética de 30 frames no formato certo                                                                                                                                                                                                                                                                                                                                                                                                          |
| `tests/test_landmarks.py`             | Testa shape/formato da sequência                                                                                                                                                                                                                                                                                                                                                                                                                            |

Trabalha com vídeos próprios (não precisa dos vídeos reais da Pessoa 1) e com os datasets públicos — pode começar a extrair landmarks de vídeos baixados de exemplo desde o Dia 1.

**Decisão travada sobre a MINDS-Libras**: rodar o `extrair_landmarks()` do zero sobre os vídeos brutos da base, **ignorando** os pontos pré-anotados que a base já disponibiliza. Dá mais trabalho, mas garante que o mesmo processo de extração usado no treino é usado depois no vídeo ao vivo (consistência treino/inferência é mais importante que economizar esse passo).

**Relação direta com a Pessoa 3**: o `data/processed/*.npz` gerado aqui **é** o dataset real que a Pessoa 3 usa pra treinar de verdade — não é uma etapa separada nem um bloqueio prévio. A Pessoa 3 já está treinando desde o Dia 1 com dado sintético e vocabulário provisório; assim que este entregável existir, ela só troca a fonte do `dataset.py`, sem esperar nenhuma reunião ou aprovação adicional.

### Pessoa 3 — Modelagem e Treinamento

Arquitetura, hiperparâmetros e framework já estão **travados na Seção 3.5** — não é mais decisão a tomar, é especificação a implementar. O trabalho agora é 100% de código.

| Arquivo                             | Conteúdo                                                                                                                                                           |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/modelo/dataset.py`           | `Dataset`/`DataLoader` (PyTorch) que lê os `.npz` de `data/processed/` (ou do gerador sintético, no início)                                              |
| `src/modelo/arquiteturas.py`      | LSTM bidirecional 2 camadas,`hidden_size=128`, conforme Seção 3.5 (não precisa testar outras arquiteturas agora — só depois de ter uma baseline funcionando) |
| `src/modelo/treino.py`            | Loop de treino com os hiperparâmetros fixados (Adam, lr=1e-3, batch 16, early stopping), salva`modelo_melhor.pt` e `scaler.pkl`                                |
| `src/modelo/avaliacao.py`         | Acurácia, matriz de confusão, F1 por classe                                                                                                                       |
| `src/modelo/baseline_classico.py` | SVM/Random Forest (scikit-learn) sobre features simplificadas — comparação no relatório                                                                         |
| `src/modelo/inferencia.py`        | Implementa`predict()` (Contrato C), aplicando `scaler.pkl` e o threshold 0.6 internamente — versão mock (aleatória) até o treino terminar                   |
| `scripts/gerar_mock_landmarks.py` | Gerador de sequências sintéticas`(30, 258)` com rótulo aleatório — é o que destrava o treino desde já, sem esperar a Pessoa 2                              |

**O que fazer agora que está tudo decidido**: implementar os 6 arquivos acima nessa ordem, rodar o pipeline inteiro ponta a ponta contra dado sintético, confirmar que treina sem erro e que `predict()` responde no formato certo. Quando a Pessoa 2 entregar `data/processed/*.npz` real, só trocar a fonte do `dataset.py` — nenhuma outra mudança de código.

### Pessoa 4 — Linguística Computacional

| Arquivo                                   | Conteúdo                                                                                         |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `src/linguagem/regras_gramaticais.py`   | Implementa`glosas_para_frase()` (Contrato D) via regras (reordenação, artigos, preposições) |
| `src/linguagem/modelo_seq2seq.py`       | Opcional: fine-tuning leve de modelo pequeno (ex. T5) se houver tempo na Semana 3/4               |
| `src/linguagem/corpus_glosa_frase.json` | Pares glosa→frase construídos manualmente pela equipe para treinar/validar as regras            |
| `data/mocks/frase_exemplo.json`         | Exemplos de entrada/saída                                                                        |

Trabalha só com texto — testa digitando listas de glosas manualmente, nunca precisa rodar vídeo, MediaPipe ou o modelo de deep learning.

**Decisão travada**: `regras_gramaticais.py` é o entregável obrigatório (baseado em regras de reordenação sujeito-verbo-objeto + inserção de artigos/preposições comuns) — é isso que vai pro MVP funcional. `modelo_seq2seq.py` é **stretch goal opcional**, só se sobrar tempo real na Semana 3/4; o projeto não depende dele para funcionar.

### Pessoa 5 — Interface, Integração e Documentação

**Decisão travada**: **Streamlit** (não Flask) — reduz drasticamente o tempo de dev de UI, e o componente `st.camera_input`/`streamlit-webrtc` já resolve boa parte da captura de vídeo ao vivo no navegador.

| Arquivo                            | Conteúdo                                                                                                                                                                                               |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/app/main.py`                | App Streamlit com vídeo ao vivo (via`streamlit-webrtc`)                                                                                                                                              |
| `src/app/overlay.py`             | Desenho da legenda: caixa semitransparente preta na parte inferior do frame, texto branco, fonte`cv2.FONT_HERSHEY_SIMPLEX`, tamanho proporcional à resolução — estilo "legenda de vídeo" padrão |
| `src/app/mocks/`                 | Stubs de`predict()` e `glosas_para_frase()` para desenvolver a UI sem depender de ninguém                                                                                                          |
| `src/app/pipeline_integrador.py` | Só a partir da Semana 3: importa de verdade`captura`, `landmarks`, `modelo`, `linguagem`; implementa a lógica de buffer do Contrato C.5 (Seção 3.6)                                         |
| `README.md`, `docs/`           | Documentação, organização do relatório final, vídeo de demonstração                                                                                                                             |

Constrói a aplicação inteira contra os próprios mocks desde a Semana 1; a integração real na Semana 3/4 é literalmente trocar os imports dos mocks pelos módulos de verdade, já que todos respeitam os mesmos contratos.

---

## 5. `requirements.txt` Sugerido

```
opencv-python==4.10.*
mediapipe==0.10.*
numpy
pandas
scikit-learn
torch                 # framework fechado (ver Seção 3.5) — não usar tensorflow
streamlit
streamlit-webrtc      # captura de vídeo ao vivo no navegador (ver Seção 4, Pessoa 5)
matplotlib
```

---

## 6. Critério de "Pronto" por Módulo

| Módulo        | Critério objetivo de aceite                                                                                                               |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Captura (P1)   | `capturar_frames()` roda ≥25 FPS e passa em `test_captura.py`                                                                         |
| Landmarks (P2) | `extrair_landmarks()` detecta mãos em ≥90% dos frames de teste; dataset processado cobre todos os itens de `vocabulario.json`        |
| Modelo (P3)    | `predict()` responde em <100ms (travado); acurácia mínima combinada (ex. ≥70%, a recalibrar com dado real) no conjunto de validação |
| Linguagem (P4) | `glosas_para_frase()` cobre 100% dos casos do `corpus_glosa_frase.json` sem erro                                                       |
| App (P5)       | Pipeline ponta a ponta roda sem crash por ≥2 minutos contínuos com webcam real                                                           |

---

## 7. Workflow de Git (Travado)

- Branch `main` protegida (sem push direto).
- Uma branch por pessoa: `feature/captura`, `feature/landmarks`, `feature/modelo`, `feature/linguagem`, `feature/app`.
- Merge via Pull Request; revisor sugerido = a pessoa "consumidora" do contrato daquele módulo (ex.: PR de `feature/landmarks` é revisado por quem está em `feature/modelo`, já que é ela quem consome o Contrato B) — assim a revisão já funciona como checagem informal do contrato.
- Commits pequenos e frequentes, mensagens descritivas (`feat:`, `fix:`, `docs:` como prefixo, padrão Conventional Commits).

---

## 8. O Que Ainda NÃO Está (e Não Pode Ser) Travado Aqui

Sendo honesto: tudo que era decisão técnica de implementação eu já fechei acima. Sobram só 4 pontos que dependem de informação que só a equipe ou o professor têm — nenhuma quantidade de especificação resolve isso de antemão:

1. **Resultado final do vocabulário** (`vocabulario.json`) — o *processo* de decisão já está travado (mínimo de 3 amostras/sinalizadores por sinal, ver Seção 4/Pessoa 2), mas o *resultado* (quais ~25–40 sinais sobrevivem ao corte) só existe depois que a Pessoa 2 rodar esse levantamento contra os dados reais das bases. Isso não bloqueia ninguém — todo mundo já trabalha com a lista provisória até lá.
2. **Gravar vídeos próprios como complemento**, caso o vocabulário final tenha sinais mal cobertos nas bases públicas — depende do resultado do item 1.
3. **Mapeamento de "Pessoa 1–5" para os integrantes reais do grupo** — depende de afinidade/interesse de cada um, só vocês sabem.
4. **Formato de entrega exigido pelo professor** (LaTeX vs. Word, número de páginas, template da disciplina) — precisa ser confirmado com quem passou o trabalho, não é algo que eu ou o código possam definir.

Tudo o mais neste documento — formatos de dado, arquitetura, hiperparâmetros, frameworks, lógica de buffer, estilo de overlay, workflow de Git — está travado e pronto pra implementar.
