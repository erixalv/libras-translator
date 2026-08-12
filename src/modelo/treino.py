"""
Loop de treino

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
N_FOLDS_CV = 8  # leave-one-signer-out: com so 8 sinalizadores de treino (vocabulario
                # reduzido a MINDS/V-LIBRASIL), cada fold deixa so 1 de fora -- maximiza
                # dado por fold (7/8) e da 8 modelos no ensemble em vez de 4 (mais
                # diversidade, ver discussao do "efeito ima" em sequencias ambiguas)
GRAD_CLIP_NORM = 5.0
LIMIAR_CONFIANCA_PADRAO = 0.6  # mesmo valor travado no Contrato C (CONTRATOS.md); usado
                                # como fallback do limiar por classe quando faltar dado OOF

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


def _logits_validacao(modelo, scaler, X_val, y_val, dispositivo, usar_velocidade):
    """Logits crus (pre-softmax) do modelo no seu proprio holdout do fold --
    usados pra calibracao (temperature scaling / limiar por classe), nunca
    pra decidir a predicao em si."""
    X_prep = adicionar_features_velocidade(X_val) if usar_velocidade else X_val
    ds = LibrasLandmarksDataset(X_prep, y_val, scaler=scaler, fit_scaler=False, augment=False)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    modelo.eval()
    logits_todos, y_todos = [], []
    with torch.no_grad():
        for x, y in dl:
            logits_todos.append(modelo(x.to(dispositivo)).cpu())
            y_todos.append(y)
    return torch.cat(logits_todos), torch.cat(y_todos)


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

    Devolve (epoca_final, modelos_e_scalers, oof) -- epoca_final e so a
    mediana da "melhor epoca" entre os folds, reportada pra referencia (nao
    ha mais um treino final separado que a use, ver docstring do modulo);
    modelos_e_scalers e None a menos que devolver_modelos=True, caso em que
    e a lista [(modelo, scaler), ...] dos N modelos do CV, prontos pra
    ensemble (ver avaliar_ensemble() e salvar_ensemble()); oof e None a
    menos que devolver_modelos=True, caso em que e (logits_oof, y_oof) --
    os logits de CADA amostra de treino, previstos pelo modelo do fold em
    que ela ficou de fora (sem vazamento: nenhum modelo ve a propria
    amostra que esta prevendo). Usado pra calibrar temperatura e limiar por
    classe (ver calibrar_temperatura() e calibrar_limiares_por_classe()).
    """
    if sinalizadores_train_full is None:
        print("AVISO: sem sinalizadores_train -- pulando validacao cruzada.")
        return None, None, None

    grupos = np.asarray(sinalizadores_train_full)
    n_grupos = len(set(grupos.tolist()))
    n_folds = min(N_FOLDS_CV, n_grupos)

    print(f"\n=== Validacao cruzada por sinalizador ({n_folds} folds, {n_grupos} sinalizadores, "
          f"velocidade={usar_velocidade}, atencao={usar_atencao}) ===")
    gkf = GroupKFold(n_splits=n_folds)
    accs, epocas, modelos_e_scalers = [], [], []
    logits_oof_lista, y_oof_lista = [], []
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
            logits_fold, y_fold = _logits_validacao(
                modelo, scaler, X_train_full[idx_val], y_train_full[idx_val], dispositivo, usar_velocidade
            )
            logits_oof_lista.append(logits_fold)
            y_oof_lista.append(y_fold)

    print(f"Validacao cruzada: {np.mean(accs):.4f} +- {np.std(accs):.4f}  (folds: {[round(a, 4) for a in accs]})")
    epoca_final = int(np.median(epocas))
    print(f"Mediana da melhor epoca entre os folds: {epoca_final} (epocas por fold: {epocas})")

    oof = (torch.cat(logits_oof_lista), torch.cat(y_oof_lista)) if devolver_modelos else None
    return epoca_final, (modelos_e_scalers if devolver_modelos else None), oof


def avaliar_ensemble(
    modelos_e_scalers: list,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dispositivo,
    temperatura: float = 1.0,
    usar_velocidade: bool = USAR_VELOCIDADE,
) -> float:
    """
    Media do softmax (calibrado por temperatura, ver calibrar_temperatura())
    dos modelos do CV (cada um viu N-1 dos N sinalizadores, combinacoes
    diferentes) -- reduz a variancia de um unico modelo, sem precisar de
    dado novo: e so aproveitar o que o CV ja treinou.
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
                probs_modelo.append(torch.softmax(modelo(x) / temperatura, dim=1).cpu())
        probs_por_modelo.append(torch.cat(probs_modelo))

    probs_media = torch.stack(probs_por_modelo).mean(dim=0)
    preds = probs_media.argmax(dim=1).numpy()
    return float((preds == y_test).mean())


def calibrar_temperatura(logits_oof: torch.Tensor, y_oof: torch.Tensor) -> float:
    """
    Temperature scaling (Guo et al. 2017): ajusta 1 escalar T que minimiza
    o NLL de softmax(logits/T) nos logits OOF do CV. So reescala o quao
    "afiada" e a distribuicao de cada modelo -- nao muda o argmax de
    ninguem, so faz a confianca refletir melhor a chance real de acerto.

    Motivacao: medimos que uma sequencia TODA ZERO (sem gesto nenhum)
    tirava 0.32 de confianca do ensemble numa classe especifica -- ou
    seja, o modelo pode estar confiante e errado, o que faz o limiar de
    confianca do Contrato C (0.6) filtrar menos do que deveria.
    """
    log_T = torch.zeros(1, requires_grad=True)
    otimizador = torch.optim.LBFGS([log_T], lr=0.01, max_iter=200)
    criterio = nn.CrossEntropyLoss()

    def _fechar():
        otimizador.zero_grad()
        T = log_T.exp()
        perda = criterio(logits_oof / T, y_oof)
        perda.backward()
        return perda

    otimizador.step(_fechar)
    return float(log_T.exp().item())


def calibrar_limiares_por_classe(
    probs_oof: torch.Tensor,
    y_oof: torch.Tensor,
    classes: list[str],
    limiar_padrao: float = LIMIAR_CONFIANCA_PADRAO,
    min_amostras_positivas: int = 5,
) -> dict[str, float]:
    """
    Um limiar de confianca global unico (0.6 pra todo mundo, Contrato C) e
    grosseiro: classes faceis (ex. Esquina, Vacina) e dificeis (ex. Bala,
    Conhecer) tem calibracao bem diferente. Aqui, pra cada classe, acha o
    ponto de corte que maximiza F1 (curva precisao-recall, one-vs-rest) nos
    dados OOF do CV -- classes com poucas amostras positivas no OOF
    (< min_amostras_positivas) caem no limiar_padrao, porque nao ha dado
    suficiente pra calibrar algo especifico com confianca.

    O resultado NAO muda o Contrato C (predict() continua com 1 numero de
    confianca e o app continua com 1 limiar global na barra lateral) --
    predict() usa esses limiares por classe so pra REESCALAR a confianca
    devolvida, de um jeito que aplicar o limiar global de 0.6 por fora
    tenha o mesmo efeito que aplicar o limiar certo de cada classe por
    dentro (ver _remapear_confianca() em inferencia.py).
    """
    from sklearn.metrics import precision_recall_curve

    y_np = y_oof.numpy()
    probs_np = probs_oof.numpy()
    limiares = {}
    for i, nome in enumerate(classes):
        y_bin = (y_np == i).astype(int)
        if y_bin.sum() < min_amostras_positivas:
            limiares[nome] = limiar_padrao
            continue
        precisao, recall, cortes = precision_recall_curve(y_bin, probs_np[:, i])
        f1 = np.divide(
            2 * precisao * recall, precisao + recall, out=np.zeros_like(precisao), where=(precisao + recall) > 0
        )
        if len(cortes) == 0:
            limiares[nome] = limiar_padrao
            continue
        melhor_idx = int(np.argmax(f1[:-1]))  # f1 tem 1 elemento a mais que cortes
        limiares[nome] = float(np.clip(cortes[melhor_idx], 0.35, 0.85))
    return limiares


def salvar_ensemble(
    modelos_e_scalers: list,
    salvar_em: str,
    classes: list[str],
    temperatura: float = 1.0,
    limiares_por_classe: dict[str, float] | None = None,
    usar_velocidade: bool = USAR_VELOCIDADE,
    usar_atencao: bool = USAR_ATENCAO,
) -> None:
    """
    Salva os N modelos do CV (e seus scalers) como o artefato de producao
    -- um unico arquivo com uma LISTA de state_dicts em vez de um
    state_dict so, mais a temperatura (calibrar_temperatura()) e os
    limiares por classe (calibrar_limiares_por_classe()). inferencia.py e
    avaliacao.py carregam isso e fazem a media do softmax calibrado dos N
    modelos (ver avaliar_ensemble()).

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
            "temperatura": temperatura,
            "limiares_por_classe": limiares_por_classe or {},
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

    _, modelos_e_scalers, oof = validar_cruzado(
        X_train_full, y_train_full, sinalizadores_train_full, len(classes), dispositivo, devolver_modelos=True
    )
    if not modelos_e_scalers:
        raise RuntimeError("CV nao devolveu modelos (sem sinalizadores_train?) -- nao ha ensemble pra salvar.")

    print(f"\n=== Producao: ensemble dos {len(modelos_e_scalers)} modelos do CV ===")
    print(f"Treino: {len(y_train_full)} (x2 com espelhamento, por fold) | Teste (holdout): {len(dados['y_test'])}")

    logits_oof, y_oof = oof
    temperatura = calibrar_temperatura(logits_oof, y_oof)
    probs_oof = torch.softmax(logits_oof / temperatura, dim=1)
    limiares_por_classe = calibrar_limiares_por_classe(probs_oof, y_oof, classes)
    print(f"Temperatura calibrada (OOF do CV): {temperatura:.3f}")
    print("Limiares por classe (OOF do CV, fallback=padrao quando pouco dado):")
    for nome, limiar in limiares_por_classe.items():
        print(f"  {nome:12s} {limiar:.3f}")

    salvar_ensemble(modelos_e_scalers, DIR_SAIDA, classes, temperatura=temperatura, limiares_por_classe=limiares_por_classe)

    # teste final -- usado UMA UNICA VEZ, so para reportar o numero real
    acc_teste = avaliar_ensemble(modelos_e_scalers, dados["X_test"], dados["y_test"], dispositivo, temperatura=temperatura)
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
