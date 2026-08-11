"""
Loop de treino (v8)

Com so 12 sinalizadores distintos no dataset, uma unica divisao
treino/validacao e loteria: rodando o mesmo treino com o holdout de
validacao trocado, a acc final variou de 11% a 47% (ver historico do
projeto). Por isso o fluxo aqui e em duas etapas:

  1. validar_cruzado(devolver_modelos=True): GroupKFold por sinalizador
     (~3 de fora por fold). Cada fold treina do zero e mede acc no holdout
     -- a media +- desvio entre folds e a estimativa CONFIAVEL de
     generalizacao pra reportar. Os 4 modelos treinados (cada um viu um
     recorte diferente de sinalizadores) sao mantidos em vez de
     descartados.
  2. salvar_ensemble(): esses 4 modelos + scalers SAO a producao -- em vez
     de treinar um 5o modelo do zero com 100% do treino (como era antes,
     ver historico do projeto), a predicao final usa a MEDIA do softmax
     dos 4 (ver avaliar_ensemble()). Testado no holdout real: 48.4% de
     acuracia com o ensemble vs 40.8% do modelo unico anterior, mesma
     arquitetura -- ganho de graca, sem dado novo, so parar de jogar fora
     os modelos que o CV ja treina. Elimina tambem a necessidade de
     escolher um numero fixo de epocas via mediana do CV: cada modelo do
     ensemble ja usa seu proprio checkpoint de early stopping.

O conjunto de teste (X_test/y_test, holdout de sinalizador desconhecido,
nunca usado no CV) e avaliado UMA UNICA VEZ, no fim, com avaliar_ensemble().

Outros pontos:
  - augmentation com espelhamento lateral (robustez a mao dominante),
    ruido e mascaramento de frames
  - capacidade do modelo (hidden_size) e regularizacao (dropout,
    weight_decay) calibradas pro tamanho real do dataset -- ver
    arquiteturas.py: com hidden_size=128/weight_decay=1e-5 originais o
    modelo memorizava o treino (~99%) e generalizava mal (~35-40% em
    sinalizador novo, 804k parametros para 12 pessoas)
  - balanceamento por CLASSE (class_weight balanceado, via CrossEntropyLoss).
    Tentamos balancear por SINALIZADOR INDIVIDUAL tambem (pra corrigir
    classes que misturam 20 amostras de gravacao propria com so 2 do
    V-LIBRASIL) via WeightedRandomSampler, mas empiricamente NAO ajudou o
    V-LIBRASIL (ficou identico, 4.5%) e piorou a gravacao propria (20% ->
    9%) -- reponderar so redistribui atencao entre dado que ja existe, nao
    cria dado novo, e com so 2 amostras de V-LIBRASIL por classe nao tem
    "atencao extra" que resolva isso. Revertido (ver historico do projeto).
  - features de velocidade (adicionar_features_velocidade em dataset.py,
    liga/desliga com USAR_VELOCIDADE) e leitura por atencao em vez de
    ultimo estado oculto (ClassificadorLSTM(usar_atencao=...), ver
    arquiteturas.py) -- duas mudancas independentes, testadas via CV antes
    de virar padrao de producao
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
from torch.utils.data import DataLoader

from src.modelo.arquiteturas import ClassificadorLSTM
from src.modelo.dataset import (
    LibrasLandmarksDataset,
    adicionar_features_velocidade,
    carregar_npz,
    espelhar_sequencias,
)

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR_SAIDA = os.path.join(RAIZ, "data", "processed")

LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16
MAX_EPOCAS = 100
PACIENCIA = 10
N_FOLDS_CV = 4  # com 12 sinalizadores, ~3 por fold
GRAD_CLIP_NORM = 5.0

# defaults de producao -- mude aqui pra ligar/desligar as features novas
USAR_VELOCIDADE = True
USAR_ATENCAO = True


def _pesos_classe_balanceados(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """
    Peso balanceado por classe (n_amostras / (n_classes * contagem)), igual
    ao class_weight="balanced" do sklearn, mas robusto a classes ausentes
    neste fold/split: em vez de quebrar (compute_class_weight so aceita
    classes com pelo menos 1 amostra), da peso neutro (1.0) pra quem nao
    apareceu -- nao ha gradiente pra essa classe de qualquer forma.
    """
    contagem = np.bincount(y, minlength=n_classes).astype(np.float64)
    pesos = np.where(contagem > 0, contagem.sum() / (n_classes * np.maximum(contagem, 1)), 1.0)
    return torch.tensor(pesos, dtype=torch.float32)


def _preparar_treino(X_train: np.ndarray, y_train: np.ndarray, n_classes: int, usar_velocidade: bool):
    """Espelha (dobra) o treino, adiciona features de velocidade (opcional) e
    monta o Dataset + pesos de classe -- comum tanto ao treino com early
    stopping quanto ao treino final de epocas fixas."""
    X_train_esp = espelhar_sequencias(X_train)
    X_train_total = np.concatenate([X_train, X_train_esp])
    y_train_total = np.concatenate([y_train, y_train])
    if usar_velocidade:
        X_train_total = adicionar_features_velocidade(X_train_total)
    ds_treino = LibrasLandmarksDataset(X_train_total, y_train_total, fit_scaler=True, augment=True)
    pesos_classe = _pesos_classe_balanceados(y_train_total, n_classes)
    return ds_treino, pesos_classe


def _treinar_um_modelo(
    X_train,
    y_train,
    X_val,
    y_val,
    n_classes: int,
    dispositivo,
    verbose: bool = True,
    usar_velocidade: bool = USAR_VELOCIDADE,
    usar_atencao: bool = USAR_ATENCAO,
) -> tuple[float, int, "ClassificadorLSTM", object]:
    """
    Treina 1 modelo do zero com early stopping em (X_val, y_val). Devolve
    (melhor_acc_val, melhor_epoca, modelo com os melhores pesos carregados, scaler).

    Usada dentro do CV (pra medir generalizacao) -- os proprios modelos
    treinados aqui, um por fold, viram a producao (ensemble, ver
    salvar_ensemble() e avaliar_ensemble() em treinar()).
    """
    ds_treino, pesos_classe = _preparar_treino(X_train, y_train, n_classes, usar_velocidade)
    X_val_prep = adicionar_features_velocidade(X_val) if usar_velocidade else X_val
    ds_val = LibrasLandmarksDataset(X_val_prep, y_val, scaler=ds_treino.scaler, fit_scaler=False, augment=False)

    dl_treino = DataLoader(ds_treino, batch_size=BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)

    n_features = ds_treino.X.shape[-1]
    modelo = ClassificadorLSTM(n_features=n_features, n_classes=n_classes, usar_atencao=usar_atencao).to(dispositivo)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterio = nn.CrossEntropyLoss(weight=pesos_classe.to(dispositivo))
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
    return melhor_acc_val, melhor_epoca, modelo, ds_treino.scaler


def validar_cruzado(
    X_train_full,
    y_train_full,
    sinalizadores_train_full,
    n_classes,
    dispositivo,
    usar_velocidade: bool = USAR_VELOCIDADE,
    usar_atencao: bool = USAR_ATENCAO,
    devolver_modelos: bool = False,
):
    """
    GroupKFold por sinalizador: cada fold treina do zero com ~3 sinalizadores
    de fora, e mede a acc nesse holdout. A media +- desvio entre folds e uma
    estimativa muito mais estavel de generalizacao do que uma unica divisao
    treino/validacao -- com so 12 pessoas no dataset, uma unica divisao
    depende demais de sorte de quem caiu no holdout.

    Devolve (epoca_final, modelos_e_scalers) -- epoca_final e so a mediana
    da "melhor epoca" entre os folds, reportada pra referencia (nao ha mais
    um treino final separado que a use, ver docstring do modulo);
    modelos_e_scalers e None a menos que devolver_modelos=True, caso em que
    e a lista [(modelo, scaler), ...] dos N modelos do CV, prontos pra
    ensemble (ver avaliar_ensemble() e salvar_ensemble()).
    """
    if sinalizadores_train_full is None:
        print("AVISO: sem sinalizadores_train -- pulando validacao cruzada.")
        return None, None

    grupos = np.asarray(sinalizadores_train_full)
    n_grupos = len(set(grupos.tolist()))
    n_folds = min(N_FOLDS_CV, n_grupos)

    print(f"\n=== Validacao cruzada por sinalizador ({n_folds} folds, {n_grupos} sinalizadores, "
          f"velocidade={usar_velocidade}, atencao={usar_atencao}) ===")
    gkf = GroupKFold(n_splits=n_folds)
    accs, epocas, modelos_e_scalers = [], [], []
    for i, (idx_treino, idx_val) in enumerate(gkf.split(X_train_full, y_train_full, groups=grupos), start=1):
        sinalizadores_fold_val = sorted(set(grupos[idx_val].tolist()))
        acc_val, melhor_epoca, modelo, scaler = _treinar_um_modelo(
            X_train_full[idx_treino],
            y_train_full[idx_treino],
            X_train_full[idx_val],
            y_train_full[idx_val],
            n_classes,
            dispositivo,
            verbose=False,
            usar_velocidade=usar_velocidade,
            usar_atencao=usar_atencao,
        )
        print(f"  fold {i}/{n_folds} (holdout={sinalizadores_fold_val}): acc_val={acc_val:.4f} (melhor epoca={melhor_epoca})")
        accs.append(acc_val)
        epocas.append(melhor_epoca)
        if devolver_modelos:
            modelos_e_scalers.append((modelo, scaler))

    print(f"Validacao cruzada: {np.mean(accs):.4f} +- {np.std(accs):.4f}  (folds: {[round(a, 4) for a in accs]})")
    epoca_final = int(np.median(epocas))
    print(f"Mediana da melhor epoca entre os folds: {epoca_final} (epocas por fold: {epocas})")
    return epoca_final, (modelos_e_scalers if devolver_modelos else None)


def avaliar_ensemble(
    modelos_e_scalers: list,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dispositivo,
    usar_velocidade: bool = USAR_VELOCIDADE,
) -> float:
    """
    Media do softmax dos modelos do CV (cada um viu ~9 de 12 sinalizadores,
    combinacoes diferentes) -- reduz a variancia de um unico modelo, sem
    precisar de dado novo: e so aproveitar o que o CV ja treinou.
    """
    X_prep = adicionar_features_velocidade(X_test) if usar_velocidade else X_test
    probs_por_modelo = []
    for modelo, scaler in modelos_e_scalers:
        ds = LibrasLandmarksDataset(X_prep, y_test, scaler=scaler, fit_scaler=False, augment=False)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        modelo.eval()
        probs_modelo = []
        with torch.no_grad():
            for x, _ in dl:
                x = x.to(dispositivo)
                probs_modelo.append(torch.softmax(modelo(x), dim=1).cpu())
        probs_por_modelo.append(torch.cat(probs_modelo))

    probs_media = torch.stack(probs_por_modelo).mean(dim=0)
    preds = probs_media.argmax(dim=1).numpy()
    return float((preds == y_test).mean())


def salvar_ensemble(
    modelos_e_scalers: list,
    salvar_em: str,
    classes: list[str],
    usar_velocidade: bool = USAR_VELOCIDADE,
    usar_atencao: bool = USAR_ATENCAO,
) -> None:
    """
    Salva os N modelos do CV (e seus scalers) como o artefato de producao
    -- um unico arquivo com uma LISTA de state_dicts em vez de um
    state_dict so. inferencia.py e avaliacao.py carregam essa lista e
    fazem a media do softmax dos N modelos (ver avaliar_ensemble()).

    Formato interno, nao afeta o Contrato C (predict() continua devolvendo
    o mesmo dict) -- so muda o que tem dentro de modelo_melhor.pt/scaler.pkl.
    """
    n_features = modelos_e_scalers[0][0].lstm.input_size
    torch.save(
        {
            "state_dicts": [modelo.state_dict() for modelo, _ in modelos_e_scalers],
            "n_features": n_features,
            "vocabulario": classes,
            "usar_velocidade": usar_velocidade,
            "usar_atencao": usar_atencao,
        },
        os.path.join(salvar_em, "modelo_melhor.pt"),
    )
    with open(os.path.join(salvar_em, "scaler.pkl"), "wb") as f:
        pickle.dump([scaler for _, scaler in modelos_e_scalers], f)


def treinar() -> None:
    os.makedirs(DIR_SAIDA, exist_ok=True)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {dispositivo}")

    dados = carregar_npz()
    classes = dados["classes"]
    print(f"Classes carregadas: {len(classes)} -> {classes}")

    X_train_full, y_train_full = dados["X_train"], dados["y_train"]
    sinalizadores_train_full = dados.get("sinalizadores_train")

    _, modelos_e_scalers = validar_cruzado(
        X_train_full, y_train_full, sinalizadores_train_full, len(classes), dispositivo, devolver_modelos=True
    )
    if not modelos_e_scalers:
        raise RuntimeError("CV nao devolveu modelos (sem sinalizadores_train?) -- nao ha ensemble pra salvar.")

    print(f"\n=== Producao: ensemble dos {len(modelos_e_scalers)} modelos do CV ===")
    print(f"Treino: {len(y_train_full)} (x2 com espelhamento, por fold) | Teste (holdout): {len(dados['y_test'])}")

    salvar_ensemble(modelos_e_scalers, DIR_SAIDA, classes)

    # teste final -- usado UMA UNICA VEZ, so para reportar o numero real
    acc_teste = avaliar_ensemble(modelos_e_scalers, dados["X_test"], dados["y_test"], dispositivo)
    print(f"Acuracia final do ENSEMBLE no TESTE (holdout de sinalizador desconhecido) = {acc_teste:.4f}")


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
