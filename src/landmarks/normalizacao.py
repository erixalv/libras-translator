"""
Normalização espacial e vetorização dos landmarks.

Layout do vetor de features (N_FEATURES = 260), ordem FIXA — combinar com a Pessoa 3:

    [  0 : 132]  pose        33 pontos x (x, y, z, visibility)
    [132 : 195]  mao_esq     21 pontos x (x, y, z)
    [195 : 258]  mao_dir     21 pontos x (x, y, z)
    [258]        mask_esq    1.0 se a mao esquerda foi detectada, 0.0 caso contrario
    [259]        mask_dir    idem para a direita

Mao ausente NAO vira zeros: as coordenadas recebem SENTINELA e a mask vai a 0.0.
Zero e uma posicao valida no espaco normalizado (e o centro dos ombros), entao
usar zero confundiria o modelo.
"""

import numpy as np

N_POSE = 33
N_MAO = 21

OFF_POSE = 0
OFF_MAO_ESQ = OFF_POSE + N_POSE * 4      # 132
OFF_MAO_DIR = OFF_MAO_ESQ + N_MAO * 3    # 195
OFF_MASK = OFF_MAO_DIR + N_MAO * 3       # 258
N_FEATURES = OFF_MASK + 2                # 260

# Indices MediaPipe Pose
OMBRO_ESQ, OMBRO_DIR = 11, 12
QUADRIL_ESQ, QUADRIL_DIR = 23, 24

# Pares (esquerda, direita) dos 33 pontos de pose do MediaPipe -- usado pelo
# espelhamento lateral. Indice 0 (nariz) fica de fora por nao ter par.
PARES_POSE_ESQ_DIR = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
    (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]

SENTINELA = -5.0   # fora de qualquer faixa plausivel apos normalizacao
EPS = 1e-6


# --------------------------------------------------------------------------
# Normalizacao espacial
# --------------------------------------------------------------------------

def _centro_e_escala(pose: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Origem = ponto medio dos ombros. Escala = distancia entre os ombros.

    Isso da invariancia a posicao do sinalizador no quadro e a distancia dele
    da camera. Fallback para a distancia ombro-quadril se os ombros estiverem
    quase sobrepostos (sinalizador de perfil).
    """
    o_esq = pose[OMBRO_ESQ, :3]
    o_dir = pose[OMBRO_DIR, :3]
    centro = (o_esq + o_dir) / 2.0

    escala = float(np.linalg.norm(o_esq[:2] - o_dir[:2]))
    if escala < 1e-3:
        quadril = (pose[QUADRIL_ESQ, :3] + pose[QUADRIL_DIR, :3]) / 2.0
        escala = float(np.linalg.norm(centro[:2] - quadril[:2]))
    if escala < EPS:
        escala = 1.0
    return centro, escala


def normalizar_registro(reg: dict) -> dict:
    """
    Recebe 1 dict no formato bruto (coordenadas MediaPipe, 0..1)
    e devolve o mesmo dict com as coordenadas normalizadas em relacao ao
    centro dos ombros. Maos ausentes continuam None.
    """
    pose = np.asarray(reg["pose"], dtype=np.float32)
    if pose.shape != (N_POSE, 4):
        raise ValueError(f"pose deve ser (33, 4), veio {pose.shape}")

    centro, escala = _centro_e_escala(pose)

    pose_n = pose.copy()
    pose_n[:, :3] = (pose[:, :3] - centro) / escala

    saida = {
        "frame_id": reg["frame_id"],
        "timestamp_ms": reg["timestamp_ms"],
        "pose": pose_n.tolist(),
        "left_hand": None,
        "right_hand": None,
    }

    for chave in ("left_hand", "right_hand"):
        mao = reg.get(chave)
        if mao is None:
            continue
        mao = np.asarray(mao, dtype=np.float32)
        if mao.shape != (N_MAO, 3):
            raise ValueError(f"{chave} deve ser (21, 3), veio {mao.shape}")
        saida[chave] = ((mao - centro) / escala).tolist()

    return saida


# --------------------------------------------------------------------------
# Vetorizacao
# --------------------------------------------------------------------------

def registro_para_vetor(reg: dict) -> np.ndarray:
    """dict (ja normalizado) -> vetor (N_FEATURES,) float32."""
    v = np.full(N_FEATURES, SENTINELA, dtype=np.float32)

    pose = np.asarray(reg["pose"], dtype=np.float32)
    v[OFF_POSE:OFF_MAO_ESQ] = pose.reshape(-1)

    for chave, off, i_mask in (
        ("left_hand", OFF_MAO_ESQ, OFF_MASK),
        ("right_hand", OFF_MAO_DIR, OFF_MASK + 1),
    ):
        mao = reg.get(chave)
        if mao is None:
            v[i_mask] = 0.0
            continue
        v[off:off + N_MAO * 3] = np.asarray(mao, dtype=np.float32).reshape(-1)
        v[i_mask] = 1.0

    return v


def espelhar_vetor(v: np.ndarray) -> np.ndarray:
    """
    Espelha lateralmente um vetor de features (N_FEATURES):
    inverte o eixo x, troca os pares esquerda/direita da pose e troca os
    blocos de mao esquerda/direita (landmarks + mask).

    Usado como data augmentation para tornar o modelo robusto a
    sinalizadores destros/canhotos que usam a mao dominante oposta em sinais
    de uma mao so -- e, de quebra, dobra a variedade efetiva de treino.

    ATENCAO: assume que o sinal nao muda de significado quando espelhado
    (valido pra maioria do vocabulario, mas nao necessariamente pra sinais
    direcionais/espaciais especificos -- conferir o vocabulario antes de usar).

    So flipa coordenadas x de blocos que estao de fato marcados como
    detectados (mask == 1.0); blocos com SENTINELA sao deixados como estao,
    senao a sentinela deixaria de ser um valor sentinela consistente.
    """
    v = v.copy()

    mask_esq_ativa = v[OFF_MASK] == 1.0
    mask_dir_ativa = v[OFF_MASK + 1] == 1.0

    pose = v[OFF_POSE:OFF_MAO_ESQ].reshape(N_POSE, 4).copy()
    pose[:, 0] = -pose[:, 0]
    for esq, dir_ in PARES_POSE_ESQ_DIR:
        pose[[esq, dir_]] = pose[[dir_, esq]]
    v[OFF_POSE:OFF_MAO_ESQ] = pose.reshape(-1)

    mao_esq = v[OFF_MAO_ESQ:OFF_MAO_DIR].reshape(N_MAO, 3).copy()
    mao_dir = v[OFF_MAO_DIR:OFF_MASK].reshape(N_MAO, 3).copy()
    if mask_esq_ativa:
        mao_esq[:, 0] = -mao_esq[:, 0]
    if mask_dir_ativa:
        mao_dir[:, 0] = -mao_dir[:, 0]

    # os blocos trocam de lugar (o que era mao esquerda vira mao direita)
    v[OFF_MAO_ESQ:OFF_MAO_DIR] = mao_dir.reshape(-1)
    v[OFF_MAO_DIR:OFF_MASK] = mao_esq.reshape(-1)
    v[OFF_MASK] = 1.0 if mask_dir_ativa else 0.0
    v[OFF_MASK + 1] = 1.0 if mask_esq_ativa else 0.0

    return v


def preencher_lacunas(seq: np.ndarray, max_lacuna: int = 5) -> np.ndarray:
    """
    Falha de deteccao curta (MediaPipe perde a mao por 2-3 frames em movimento
    rapido) e ruido, nao ausencia real. Repete o ultimo frame valido da mao
    enquanto a lacuna for <= max_lacuna. A mask continua 0.0 nesses frames, entao
    o modelo sabe que o valor foi imputado.

    seq: (T, N_FEATURES). Devolve copia.
    """
    seq = seq.copy()
    T = seq.shape[0]

    for off, i_mask in ((OFF_MAO_ESQ, OFF_MASK), (OFF_MAO_DIR, OFF_MASK + 1)):
        fatia = slice(off, off + N_MAO * 3)
        t = 0
        while t < T:
            if seq[t, i_mask] == 1.0:
                t += 1
                continue
            ini = t
            while t < T and seq[t, i_mask] == 0.0:
                t += 1
            fim = t  # primeiro valido apos a lacuna
            if fim - ini > max_lacuna:
                continue  # lacuna longa: mantem SENTINELA
            if ini > 0:
                seq[ini:fim, fatia] = seq[ini - 1, fatia]
            elif fim < T:
                seq[ini:fim, fatia] = seq[fim, fatia]

    return seq


# --------------------------------------------------------------------------
# Eixo temporal
# --------------------------------------------------------------------------

def reamostrar(seq: np.ndarray, n_frames: int = 30) -> np.ndarray:
    """
    Reamostra uma sequencia de T frames para exatamente n_frames, por indices
    uniformes. Usar no dataset de sinais ISOLADOS (MINDS/V-LIBRASIL): cada
    video ja e um sinal completo, entao vale comprimir o sinal inteiro na
    janela em vez de fatiar.
    """
    T = seq.shape[0]
    if T == 0:
        raise ValueError("sequencia vazia")
    idx = np.linspace(0, T - 1, n_frames)
    return seq[np.round(idx).astype(int)]


def ultima_janela(seq: np.ndarray, tamanho: int = 30) -> np.ndarray:
    """
    Pega os ultimos `tamanho` frames de uma sequencia. Usar no fluxo ao vivo:
    a cada frame novo que chega no buffer, recorta so a ponta mais
    recente para prever, sem esperar um sinal "terminar". Se o buffer ainda
    tem menos frames que `tamanho`, reamostra o que existe.
    """
    T = seq.shape[0]
    if T < tamanho:
        return reamostrar(seq, tamanho)
    return seq[-tamanho:]