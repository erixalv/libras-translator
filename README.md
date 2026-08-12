# 🤟 Tradutor Automático de Libras → Português

Sistema de tradução automática em tempo real de Língua Brasileira de Sinais (Libras) para Português, combinando Visão Computacional, Deep Learning e Processamento de Linguagem Natural (NLP) baseado em regras.

---

## 📐 Arquitetura do Sistema

O projeto é dividido em 5 módulos independentes baseados em contratos bem definidos (`CONTRATOS.md`):

1. **Captura (`src/captura/`)**: Leitura da webcam ou arquivos de vídeo (`OpenCV`), pré-processamento de iluminação (`CLAHE`) e segmentação de pele (`YCbCr`).
2. **Landmarks (`src/landmarks/`)**: Extração de 260 landmarks corporais e das mãos (`MediaPipe Holistic`), normalização centrada nos ombros e vetorização.
3. **Modelo DL (`src/modelo/`)**: Classificador temporal LSTM Bidirecional em `PyTorch` treinado em dataset final com 20 classes de sinais (MINDS-Libras).
4. **Linguagem (`src/linguagem/`)**: Módulo de NLP baseado em regras que converte sequências de glosas acumuladas em frases gramaticalmente corretas em português.
5. **App & Interface (`src/app/`)**: Interface Web interativa em `Streamlit` com HUD/Overlay OpenCV desenhado no vídeo ao vivo e pipeline integrador.

---

## 🛠️ Instalação e Requisitos

### Pré-requisitos

- Python 3.10+ (suporta Windows, Linux e macOS)
- Webcam para captura em tempo real (opcional: aceita arquivos `.mp4`)

### Instalação das Dependências

```bash
pip install -r requirements.txt
```

---

## 🚀 Como Executar

### 1. Executar a Aplicação Web (Streamlit)

```bash
streamlit run src/app/main.py
```

- A interface será aberta automaticamente no seu navegador (padrão: `http://localhost:8501`).
- Por padrão, a aplicação é iniciada em **Modo Real (IA ativa)** carregando o modelo `modelo_melhor.pt` e o `scaler.pkl`.
- É possível selecionar a fonte de vídeo (Webcam ou Vídeo de Exemplo) e alternar para o Modo Mock no painel lateral.

### 2. Executar a Suíte de Testes Unitários

```bash
pytest
```

---

## 📊 Estrutura de Pastas

```
libras-translator/
├── CONTRATOS.md               # Especificação técnica dos contratos entre módulos
├── vocabulario.json           # Lista oficial e origem dos 37 sinais do vocabulário
├── requirements.txt           # Dependências do projeto
├── data/
│   ├── mocks/                 # Arquivos de exemplo e mocks
│   └── processed/             # Dataset final (.npz), modelo PyTorch (.pt) e scaler (.pkl)
├── src/
│   ├── captura/               # Módulo de captura de vídeo e PDI clássico
│   ├── landmarks/             # Extração MediaPipe e normalização espacial
│   ├── modelo/                # Arquitetura LSTM, treino, avaliação e inferência
│   ├── linguagem/             # NLP baseado em regras e dicionário de concordância
│   └── app/                   # Interface Streamlit, overlay e pipeline integrador
└── tests/                     # Testes unitários do sistema
```
