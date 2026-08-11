"""
Loop de treino (v7)

Com so 12 sinalizadores distintos no dataset, uma unica divisao
treino/validacao e loteria: rodando o mesmo treino com o holdout de
validacao trocado, a acc final variou de 11% a 47% (ver historico do
projeto). Por isso o fluxo aqui e em duas etapas:

  1. validar_cruzado(): GroupKFold por sinalizador (~3 de fora por fold).
     Cada fold treina do zero e mede acc no holdout -- a media +- desvio
     entre folds e a estimativa CONFIAVEL de generalizacao pra reportar.
     Tambem devolve a mediana da "melhor epoca" entre os folds.
  2. treinar_epocas_fixas(): treino final de producao, com TODOS os 12
     sinalizadores de treino (sem separar validacao -- pouca gente pra
     desperdicar em early stopping), por esse numero fixo de epocas.

O conjunto de teste (X_test/y_test, holdout de sinalizador desconhecido,
nunca usado no CV nem no treino final) e avaliado UMA UNICA VEZ, no fim.

Outros pontos:
  - augmentation com espelhamento lateral (robustez a mao dominante),
    ruido e mascaramento de frames
  - capacidade do modelo (hidden_size) e regularizacao (dropout,
    weight_decay) calibradas pro tamanho real do dataset -- ver
    arquiteturas.py: com hidden_size=128/weight_decay=1e-5 originais o
    modelo memorizava o treino (~99%) e generalizava mal (~35-40% em
    sinalizador novo, 804k parametros para 12 pessoas)
  - balanceamento por AMOSTRA (WeightedRandomSampler), nao so por classe:
    dentro de classes que misturam V-LIBRASIL + gravacao propria (ex.:
    "Água" com 20 amostras de 4 pessoas vs so 2 do V-LIBRASIL), o
    gradiente era dominado 10:1 pela fonte majoritaria -- o modelo
    aprendia "o jeito da gravacao propria" e generalizava mal pro estilo
    V-LIBRASIL (~5% de acuracia nessa fonte, mesmo com deteccao de mao
    quase perfeita). _pesos_amostra_balanceados() da peso igual a cada
    SINALIZADOR INDIVIDUAL dentro da sua classe, nao so a cada classe
    (ver historico do projeto)
  - gradient clipping

Hiperparametros de loss/otimizador/batch continuam os da Secao 3.5 do
CONTRATOS.md (CrossEntropyLoss, Adam, batch=16, ate 100 epocas, paciencia
10 no CV); hidden_size e weight_decay foram ajustados depois de medir
overfitting severo com os valores originais (ver historico do projeto).

Uso:
    python -m src.modelo.treino
"""
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.modelo.arquiteturas import ClassificadorLSTM
from src.modelo.dataset import LibrasLandmarksDataset, carregar_npz, espelhar_sequencias

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR_SAIDA = os.path.join(RAIZ, "data", "processed")

LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16
MAX_EPOCAS = 100
PACIENCIA = 10
N_FOLDS_CV = 4  # com 12 sinalizadores, ~3 por fold
GRAD_CLIP_NORM = 5.0


def _pesos_amostra_balanceados(y: np.ndarray, sinalizadores: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Peso por amostra que balanceia DUAS coisas ao mesmo tempo:
      1. entre classes -- cada classe contribui igual no total (equivalente
         ao class_weight="balanced" que isso substitui)
      2. DENTRO de cada classe, entre sinalizadores individuais -- cada
         pessoa/sinalizador contribui igual dentro da sua classe

    Sem o item 2, uma classe com amostras de fontes muito desiguais (ex.:
    "Água" com 20 amostras de 4 pessoas da gravacao propria + so 2 do
    V-LIBRASIL) aprende quase so o "jeito" da fonte majoritaria. Foi isso
    que explicou o V-LIBRASIL saindo com ~5% de acuracia no teste mesmo
    tendo a deteccao de mao mais limpa de todo o dataset -- o gradiente
    dessas classes era dominado 10:1 pela gravacao propria (ver historico
    do projeto). Usado como pesos de um WeightedRandomSampler, nao como
    peso de classe no criterio -- os dois juntos duplicariam a correcao.
    """
    pesos = np.zeros(len(y), dtype=np.float64)
    for c in np.unique(y):
        mascara_classe = y == c
        sinalizadores_da_classe = sinalizadores[mascara_classe]
        sinalizadores_unicos = np.unique(sinalizadores_da_classe)
        n_sinalizadores = len(sinalizadores_unicos)
        for s in sinalizadores_unicos:
            mascara_s = mascara_classe & (sinalizadores == s)
            contagem = mascara_s.sum()
            pesos[mascara_s] = 1.0 / (n_classes * n_sinalizadores * contagem)
    return pesos


def _preparar_treino(X_train: np.ndarray, y_train: np.ndarray, sinalizadores_train: np.ndarray, n_classes: int):
    """Espelha (dobra) o treino e monta o Dataset + pesos por amostra (classe
    + sinalizador) -- comum tanto ao treino com early stopping quanto ao
    treino final de epocas fixas."""
    X_train_esp = espelhar_sequencias(X_train)
    X_train_total = np.concatenate([X_train, X_train_esp])
    y_train_total = np.concatenate([y_train, y_train])
    sinalizadores_total = np.concatenate([sinalizadores_train, sinalizadores_train])
    ds_treino = LibrasLandmarksDataset(X_train_total, y_train_total, fit_scaler=True, augment=True)
    pesos_amostra = _pesos_amostra_balanceados(y_train_total, sinalizadores_total, n_classes)
    return ds_treino, pesos_amostra


def _treinar_um_modelo(
    X_train,
    y_train,
    sinalizadores_train,
    X_val,
    y_val,
    n_classes: int,
    dispositivo,
    verbose: bool = True,
) -> tuple[float, int, "ClassificadorLSTM"]:
    """
    Treina 1 modelo do zero com early stopping em (X_val, y_val). Devolve
    (melhor_acc_val, melhor_epoca, modelo com os melhores pesos carregados).

    Usado dentro do CV (pra medir generalizacao) -- o treino final de
    producao NAO usa esta funcao, usa treinar_epocas_fixas (ver abaixo),
    porque um unico split treino/validacao com so 12 sinalizadores se
    mostrou instavel demais pra escolher o checkpoint final (ver historico).
    """
    ds_treino, pesos_amostra = _preparar_treino(X_train, y_train, sinalizadores_train, n_classes)
    ds_val = LibrasLandmarksDataset(X_val, y_val, scaler=ds_treino.scaler, fit_scaler=False, augment=False)

    sampler = WeightedRandomSampler(pesos_amostra, num_samples=len(pesos_amostra), replacement=True)
    dl_treino = DataLoader(ds_treino, batch_size=BATCH_SIZE, sampler=sampler)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)

    n_features = X_train.shape[-1]
    modelo = ClassificadorLSTM(n_features=n_features, n_classes=n_classes).to(dispositivo)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterio = nn.CrossEntropyLoss()  # balanceamento ja vem do sampler (classe + sinalizador)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador, mode="max", factor=0.5, patience=4)

    melhor_acc_val = 0.0
    melhor_epoca = 0
    melhor_state_dict = None
    epocas_sem_melhora = 0

    for epoca in range(1, MAX_EPOCAS + 1):
        modelo.train()
        perda_total = 0.0
        for x, y in dl_treino:
            x, y = x.to(dispositivo), y.to(dispositivo)
            otimizador.zero_grad()
            logits = modelo(x)
            perda = criterio(logits, y)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), GRAD_CLIP_NORM)
            otimizador.step()
            perda_total += perda.item() * x.size(0)
        perda_media = perda_total / len(ds_treino)

        acc_val = avaliar_rapido(modelo, dl_val, dispositivo)
        scheduler.step(acc_val)
        if verbose:
            lr_atual = otimizador.param_groups[0]["lr"]
            print(f"  Epoca {epoca:03d} | perda_treino={perda_media:.4f} | acc_val={acc_val:.4f} | lr={lr_atual:.2e}")

        if acc_val > melhor_acc_val:
            melhor_acc_val = acc_val
            melhor_epoca = epoca
            epocas_sem_melhora = 0
            melhor_state_dict = {k: v.clone() for k, v in modelo.state_dict().items()}
        else:
            epocas_sem_melhora += 1
            if epocas_sem_melhora >= PACIENCIA:
                if verbose:
                    print(f"  Early stopping na epoca {epoca} (sem melhora ha {PACIENCIA} epocas).")
                break

    modelo.load_state_dict(melhor_state_dict)
    return melhor_acc_val, melhor_epoca, modelo


def validar_cruzado(X_train_full, y_train_full, sinalizadores_train_full, n_classes, dispositivo) -> int | None:
    """
    GroupKFold por sinalizador: cada fold treina do zero com ~3 sinalizadores
    de fora, e mede a acc nesse holdout. A media +- desvio entre folds e uma
    estimativa muito mais estavel de generalizacao do que uma unica divisao
    treino/validacao -- com so 12 pessoas no dataset, uma unica divisao
    depende demais de sorte de quem caiu no holdout.

    Devolve a mediana da "melhor epoca" entre os folds -- usada como numero
    fixo de epocas pro treino final com 100% do treino (ver treinar_epocas_fixas).
    """
    if sinalizadores_train_full is None:
        print("AVISO: sem sinalizadores_train -- pulando validacao cruzada.")
        return None

    grupos = np.asarray(sinalizadores_train_full)
    n_grupos = len(set(grupos.tolist()))
    n_folds = min(N_FOLDS_CV, n_grupos)

    print(f"\n=== Validacao cruzada por sinalizador ({n_folds} folds, {n_grupos} sinalizadores) ===")
    gkf = GroupKFold(n_splits=n_folds)
    accs, epocas = [], []
    for i, (idx_treino, idx_val) in enumerate(gkf.split(X_train_full, y_train_full, groups=grupos), start=1):
        sinalizadores_fold_val = sorted(set(grupos[idx_val].tolist()))
        acc_val, melhor_epoca, _ = _treinar_um_modelo(
            X_train_full[idx_treino],
            y_train_full[idx_treino],
            grupos[idx_treino],
            X_train_full[idx_val],
            y_train_full[idx_val],
            n_classes,
            dispositivo,
            verbose=False,
        )
        print(f"  fold {i}/{n_folds} (holdout={sinalizadores_fold_val}): acc_val={acc_val:.4f} (melhor epoca={melhor_epoca})")
        accs.append(acc_val)
        epocas.append(melhor_epoca)

    print(f"Validacao cruzada: {np.mean(accs):.4f} +- {np.std(accs):.4f}  (folds: {[round(a, 4) for a in accs]})")
    epoca_final = int(np.median(epocas))
    print(f"Mediana da melhor epoca entre os folds: {epoca_final} (epocas por fold: {epocas})")
    return epoca_final


def treinar_epocas_fixas(
    X_train_full,
    y_train_full,
    sinalizadores_train_full,
    n_classes,
    dispositivo,
    n_epocas: int,
    salvar_em: str,
    classes: list[str],
) -> "ClassificadorLSTM":
    """
    Treino final de producao: usa TODOS os sinalizadores de treino (sem
    separar validacao) por um numero FIXO de epocas -- vindo da mediana do
    CV. Sem validacao pra fazer early stopping por dois motivos: (1) toda
    pessoa disponivel deveria ir pro treino, ja que sao so 12 no total, e
    (2) o CV ja mostrou que escolher o checkpoint via um unico split
    pequeno de validacao e instavel demais pra confiar (ver historico).
    """
    ds_treino, pesos_amostra = _preparar_treino(X_train_full, y_train_full, sinalizadores_train_full, n_classes)
    sampler = WeightedRandomSampler(pesos_amostra, num_samples=len(pesos_amostra), replacement=True)
    dl_treino = DataLoader(ds_treino, batch_size=BATCH_SIZE, sampler=sampler)

    n_features = X_train_full.shape[-1]
    modelo = ClassificadorLSTM(n_features=n_features, n_classes=n_classes).to(dispositivo)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterio = nn.CrossEntropyLoss()  # balanceamento ja vem do sampler (classe + sinalizador)

    modelo.train()
    for epoca in range(1, n_epocas + 1):
        perda_total = 0.0
        for x, y in dl_treino:
            x, y = x.to(dispositivo), y.to(dispositivo)
            otimizador.zero_grad()
            logits = modelo(x)
            perda = criterio(logits, y)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), GRAD_CLIP_NORM)
            otimizador.step()
            perda_total += perda.item() * x.size(0)
        perda_media = perda_total / len(ds_treino)
        print(f"  Epoca {epoca:03d}/{n_epocas} | perda_treino={perda_media:.4f}")

    torch.save(
        {"state_dict": modelo.state_dict(), "n_features": n_features, "vocabulario": classes},
        os.path.join(salvar_em, "modelo_melhor.pt"),
    )
    with open(os.path.join(salvar_em, "scaler.pkl"), "wb") as f:
        pickle.dump(ds_treino.scaler, f)

    return modelo


def treinar() -> None:
    os.makedirs(DIR_SAIDA, exist_ok=True)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {dispositivo}")

    dados = carregar_npz()
    classes = dados["classes"]
    print(f"Classes carregadas: {len(classes)} -> {classes}")

    X_train_full, y_train_full = dados["X_train"], dados["y_train"]
    sinalizadores_train_full = dados.get("sinalizadores_train")

    epoca_final = validar_cruzado(X_train_full, y_train_full, sinalizadores_train_full, len(classes), dispositivo)
    if epoca_final is None:
        epoca_final = MAX_EPOCAS // 2  # fallback sem sinalizador pra guiar o CV

    print(f"\n=== Treino final (producao, 100% do treino, {epoca_final} epocas) ===")
    print(f"Treino: {len(y_train_full)} (x2 com espelhamento) | Teste (holdout): {len(dados['y_test'])}")

    if sinalizadores_train_full is None:
        # sem sinalizador pra balancear por fonte -- trata como 1 fonte so
        # (o balanceamento por sinalizador vira um no-op, so sobra o de classe)
        sinalizadores_train_full = np.full(len(y_train_full), "desconhecido")

    modelo = treinar_epocas_fixas(
        X_train_full, y_train_full, sinalizadores_train_full, len(classes), dispositivo, epoca_final, DIR_SAIDA, classes
    )

    with open(os.path.join(DIR_SAIDA, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    # teste final -- usado UMA UNICA VEZ, so para reportar o numero real
    ds_teste = LibrasLandmarksDataset(dados["X_test"], dados["y_test"], scaler=scaler, fit_scaler=False, augment=False)
    dl_teste = DataLoader(ds_teste, batch_size=BATCH_SIZE, shuffle=False)
    acc_teste = avaliar_rapido(modelo, dl_teste, dispositivo)
    print(f"Acuracia final no TESTE (holdout de sinalizador desconhecido) = {acc_teste:.4f}")


def avaliar_rapido(modelo, dl, dispositivo) -> float:
    modelo.eval()
    corretos, total = 0, 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(dispositivo), y.to(dispositivo)
            preds = modelo(x).argmax(dim=1)
            corretos += (preds == y).sum().item()
            total += y.size(0)
    return corretos / total if total > 0 else 0.0


if __name__ == "__main__":
    treinar()
