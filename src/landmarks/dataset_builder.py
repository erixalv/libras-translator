"""
CLI que varre data/raw/, extrai landmarks de cada video e monta o dataset
final (N, 30, 260) para a Pessoa 3, com split treino/teste por sinalizador.

Convencao de pastas esperada:

    data/raw/<DATASET>/<SINAL>/<arquivo>_<sinalizador>.mp4

    ex: data/raw/MINDS/AGUA/sinal03_signer07.mp4
        -> sinal   = "AGUA"       (nome da pasta)
        -> arquivo = "sinal03"
        -> sinalizador = "signer07"  (ultimo trecho do nome, apos "_")

Uso:
    python -m src.landmarks.dataset_builder \\
        --raiz data/raw \\
        --saida data/processed/dataset.npz \\
        --teste-sinalizadores signer07,signer12
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .extrator_mediapipe import construir_sequencia, extrair_video


def _parse_nome(caminho: Path) -> tuple[str, str] | None:
    """(arquivo, sinalizador) a partir de '<arquivo>_<sinalizador>.mp4'."""
    if "_" not in caminho.stem:
        return None
    arquivo, _, sinalizador = caminho.stem.rpartition("_")
    if not arquivo or not sinalizador:
        return None
    return arquivo, sinalizador


def _coletar_videos(raiz: Path) -> list[tuple[Path, str]]:
    """Varre <raiz>/<DATASET>/<SINAL>/*.mp4 -> lista de (caminho, sinal)."""
    videos = []
    for caminho in sorted(raiz.glob("*/*/*.mp4")):
        sinal = caminho.parent.name
        videos.append((caminho, sinal))
    return videos


def _processar_video(
    caminho: Path, passo: int, n_frames: int
) -> tuple[np.ndarray | None, dict, str]:
    """
    Extrai e constroi a sequencia de 1 video.
    Devolve (sequencia ou None, stats, motivo_descarte). motivo_descarte e
    "" quando a sequencia foi construida com sucesso.
    """
    try:
        registros, stats = extrair_video(str(caminho), passo=passo)
    except IOError:
        return None, {}, "erro ao abrir o arquivo"

    if not registros:
        return None, stats, "pose nao detectada em nenhum frame"

    seq = construir_sequencia(registros, n_frames=n_frames, modo="reamostrar")
    return seq, stats, ""


def _caminho_cache(cache_dir: Path, sinal: str, arquivo: str, sinalizador: str) -> Path:
    return cache_dir / sinal / f"{arquivo}_{sinalizador}.npz"


def _processar_com_cache(
    caminho: Path,
    sinal: str,
    arquivo: str,
    sinalizador: str,
    cache_dir: Path | None,
    passo: int,
    n_frames: int,
) -> tuple[np.ndarray | None, float, str, bool]:
    """
    Igual a _processar_video, mas reaproveita um resultado ja calculado em
    cache_dir se existir. Devolve (sequencia ou None, taxa_mao, motivo, veio_do_cache).

    Isso e o que permite rodar o dataset aos poucos: cada video processado
    grava o resultado no cache imediatamente, entao interromper o script e
    rodar de novo depois pula direto para os videos que ainda faltam, sem
    reprocessar os que ja foram feitos.
    """
    caminho_cache = (
        _caminho_cache(cache_dir, sinal, arquivo, sinalizador) if cache_dir else None
    )

    if caminho_cache is not None and caminho_cache.exists():
        dados = np.load(caminho_cache, allow_pickle=False)
        if "seq" in dados:
            return dados["seq"], float(dados["taxa_mao"]), "", True
        return None, 0.0, str(dados["motivo"]), True

    seq, stats, motivo = _processar_video(caminho, passo, n_frames)
    taxa_mao = float(stats.get("taxa_mao", 0.0))

    if caminho_cache is not None:
        caminho_cache.parent.mkdir(parents=True, exist_ok=True)
        if motivo:
            np.savez_compressed(caminho_cache, motivo=np.array(motivo))
        else:
            np.savez_compressed(caminho_cache, seq=seq, taxa_mao=np.array(taxa_mao))

    return seq, taxa_mao, motivo, False


def _pendentes(
    videos: list[tuple[Path, str]], cache_dir: Path | None, limite: int | None
) -> list[tuple[Path, str, str, str]]:
    """
    Videos com nome valido que ainda nao estao no cache, ate `limite`. Usado
    tanto para pre-aquecer o cache em paralelo quanto (implicitamente, via o
    mesmo `limite`) para o loop serial de agregacao parar no mesmo ponto.
    """
    pendentes = []
    for caminho, sinal in videos:
        nome = _parse_nome(caminho)
        if nome is None:
            continue
        arquivo, sinalizador = nome
        if cache_dir is not None and _caminho_cache(
            cache_dir, sinal, arquivo, sinalizador
        ).exists():
            continue
        pendentes.append((caminho, sinal, arquivo, sinalizador))
        if limite is not None and len(pendentes) >= limite:
            break
    return pendentes


def _worker_processar(args: tuple[str, str, str, str, str, int, int]) -> None:
    """
    Roda em um processo filho: processa 1 video e grava no cache. Nao devolve
    a sequencia — quem le o resultado e o loop serial de agregacao depois,
    direto do cache (ja rapido, sem MediaPipe de novo).
    """
    caminho, sinal, arquivo, sinalizador, cache_dir, passo, n_frames = args

    import cv2

    cv2.setNumThreads(1)  # evita N processos disputando threads internas do OpenCV

    _processar_com_cache(
        Path(caminho), sinal, arquivo, sinalizador, Path(cache_dir), passo, n_frames
    )


def _processar_em_paralelo(
    tarefas: list[tuple[Path, str, str, str]],
    cache_dir: Path,
    passo: int,
    n_frames: int,
    n_processos: int,
) -> int:
    """Pre-aquece o cache rodando `n_processos` videos por vez. Devolve quantos processou."""
    if not tarefas:
        return 0

    args = [
        (str(caminho), sinal, arquivo, sinalizador, str(cache_dir), passo, n_frames)
        for caminho, sinal, arquivo, sinalizador in tarefas
    ]

    concluidos = 0
    with ProcessPoolExecutor(max_workers=n_processos) as executor:
        for _ in executor.map(_worker_processar, args):
            concluidos += 1
            if concluidos % 25 == 0 or concluidos == len(args):
                print(f"  [paralelo] {concluidos}/{len(args)} videos", flush=True)
    return concluidos


def montar_dataset(
    raiz: Path,
    teste_sinalizadores: set[str],
    passo: int = 1,
    n_frames: int = 30,
    cache_dir: Path | None = None,
    limite: int | None = None,
    paralelo: int = 1,
) -> dict:
    """
    Roda a extracao em todos os videos de <raiz> e devolve um dict pronto
    para np.savez_compressed, mais um relatorio para impressao.

    cache_dir: se informado, cada video processado e salvo ali e reaproveitado
        em execucoes futuras — permite rodar em varias sessoes/aos poucos.
    limite: numero maximo de videos NOVOS (fora do cache) a processar nesta
        execucao. Videos que ja estao no cache nao contam pro limite. Use
        para avancar o dataset em lotes controlados em vez de uma vez so.
    paralelo: numero de processos para pre-aquecer o cache em paralelo antes
        do loop de agregacao (que continua serial, mas fica rapido porque so
        le do cache). Precisa de cache_dir; 1 = comportamento serial normal.
    """
    videos = _coletar_videos(raiz)
    if not videos:
        raise FileNotFoundError(
            f"nenhum .mp4 encontrado em {raiz}/<DATASET>/<SINAL>/*.mp4"
        )

    processados_em_paralelo = 0
    if paralelo > 1:
        if cache_dir is None:
            raise ValueError("--paralelo precisa de cache (nao use --sem-cache junto)")
        tarefas = _pendentes(videos, cache_dir, limite)
        processados_em_paralelo = _processar_em_paralelo(
            tarefas, cache_dir, passo, n_frames, paralelo
        )

    classes = sorted({sinal for _, sinal in videos})
    indice_classe = {sinal: i for i, sinal in enumerate(classes)}

    treino_X, treino_y, treino_sinalizador = [], [], []
    teste_X, teste_y, teste_sinalizador = [], [], []

    descartes = Counter()
    taxas_mao = []
    contagem_classe = Counter()  # (sinal, "treino"/"teste") -> quantidade
    processados_nesta_execucao = 0
    do_cache = 0
    restantes = 0
    # o limite conta o total de videos NOVOS nesta execucao — o que o
    # paralelo ja processou tambem consome o limite, senao o loop serial
    # abaixo continua processando videos de verdade ate atingir o limite
    # sozinho, dobrando o trabalho.
    novos_total = processados_em_paralelo

    for caminho, sinal in videos:
        nome = _parse_nome(caminho)
        if nome is None:
            descartes["nome de arquivo fora da convencao <arquivo>_<sinalizador>"] += 1
            continue
        arquivo, sinalizador = nome

        if limite is not None and novos_total >= limite:
            veio_do_cache = (
                cache_dir is not None
                and _caminho_cache(cache_dir, sinal, arquivo, sinalizador).exists()
            )
            if not veio_do_cache:
                restantes += 1
                continue

        seq, taxa_mao, motivo, veio_do_cache = _processar_com_cache(
            caminho, sinal, arquivo, sinalizador, cache_dir, passo, n_frames
        )
        if veio_do_cache:
            do_cache += 1
        else:
            processados_nesta_execucao += 1
            novos_total += 1

        if motivo:
            descartes[motivo] += 1
            continue

        taxas_mao.append(taxa_mao)

        destino_X, destino_y, destino_sinalizador, rotulo_split = (
            (teste_X, teste_y, teste_sinalizador, "teste")
            if sinalizador in teste_sinalizadores
            else (treino_X, treino_y, treino_sinalizador, "treino")
        )
        destino_X.append(seq)
        destino_y.append(indice_classe[sinal])
        destino_sinalizador.append(sinalizador)
        contagem_classe[(sinal, rotulo_split)] += 1

    if not treino_X:
        raise RuntimeError("nenhuma amostra de treino sobrou — revisar data/raw/")

    dados = {
        "X_train": np.stack(treino_X).astype(np.float32),
        "y_train": np.array(treino_y, dtype=np.int64),
        "sinalizadores_train": np.array(treino_sinalizador),
        "X_test": (
            np.stack(teste_X).astype(np.float32)
            if teste_X
            else np.empty((0, n_frames, treino_X[0].shape[1]), dtype=np.float32)
        ),
        "y_test": np.array(teste_y, dtype=np.int64),
        "sinalizadores_test": np.array(teste_sinalizador),
        "classes": np.array(classes),
    }

    relatorio = {
        "classes": classes,
        "contagem_classe": contagem_classe,
        "descartes": descartes,
        "total_videos": len(videos),
        "total_mantidos": len(treino_X) + len(teste_X),
        "taxa_mao_media": float(np.mean(taxas_mao)) if taxas_mao else 0.0,
        "taxa_mao_abaixo_090": sum(1 for t in taxas_mao if t < 0.90),
        "processados_nesta_execucao": processados_nesta_execucao,
        "processados_em_paralelo": processados_em_paralelo,
        "vindos_do_cache": do_cache,
        "restantes_por_limite": restantes,
    }
    return dados, relatorio


def imprimir_relatorio(relatorio: dict) -> None:
    print(f"\nvideos encontrados: {relatorio['total_videos']}")
    print(f"amostras mantidas:  {relatorio['total_mantidos']}")
    total_novos = relatorio["processados_nesta_execucao"] + relatorio["processados_em_paralelo"]
    print(f"processados agora (fora do cache): {total_novos}")
    if relatorio["processados_em_paralelo"]:
        print(f"  (dos quais em paralelo: {relatorio['processados_em_paralelo']})")
    print(f"vindos do cache (ja existiam antes desta execucao ou foram pre-aquecidos em paralelo): {relatorio['vindos_do_cache']}")
    if relatorio["restantes_por_limite"]:
        print(
            f"ainda faltam (--limite atingido):  {relatorio['restantes_por_limite']}"
            "  — rode o mesmo comando de novo pra continuar"
        )

    print("\npor classe (treino / teste):")
    for sinal in relatorio["classes"]:
        n_treino = relatorio["contagem_classe"].get((sinal, "treino"), 0)
        n_teste = relatorio["contagem_classe"].get((sinal, "teste"), 0)
        print(f"  {sinal:20s} {n_treino:4d} / {n_teste:4d}")

    if relatorio["descartes"]:
        print("\ndescartados:")
        for motivo, qtd in relatorio["descartes"].most_common():
            print(f"  {qtd:4d}  {motivo}")
    else:
        print("\nnenhum video descartado")

    print(f"\ntaxa_mao media: {relatorio['taxa_mao_media']:.3f}")
    print(f"videos com taxa_mao < 0.90: {relatorio['taxa_mao_abaixo_090']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raiz", type=Path, default=Path("data/raw"))
    ap.add_argument("--saida", type=Path, default=Path("data/processed/dataset.npz"))
    ap.add_argument(
        "--teste-sinalizadores",
        type=str,
        default="",
        help="ids de sinalizador separados por virgula, ex: signer07,signer12",
    )
    ap.add_argument("--passo", type=int, default=1)
    ap.add_argument("--n-frames", type=int, default=30)
    ap.add_argument(
        "--cache",
        type=Path,
        default=Path("data/processed/cache"),
        help="onde guardar o resultado de cada video ja processado",
    )
    ap.add_argument(
        "--sem-cache", action="store_true", help="ignora e nao grava cache"
    )
    ap.add_argument(
        "--limite",
        type=int,
        default=None,
        help="processa no maximo N videos novos nesta execucao (o resto fica "
        "pro proximo run, retomando do cache)",
    )
    ap.add_argument(
        "--paralelo",
        type=int,
        default=1,
        help="numero de processos para pre-aquecer o cache em paralelo (precisa de cache)",
    )
    args = ap.parse_args()

    teste_sinalizadores = {
        s.strip() for s in args.teste_sinalizadores.split(",") if s.strip()
    }

    dados, relatorio = montar_dataset(
        raiz=args.raiz,
        teste_sinalizadores=teste_sinalizadores,
        passo=args.passo,
        n_frames=args.n_frames,
        cache_dir=None if args.sem_cache else args.cache,
        limite=args.limite,
        paralelo=args.paralelo,
    )

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.saida, **dados)
    print(f"dataset salvo em {args.saida}")

    imprimir_relatorio(relatorio)


if __name__ == "__main__":
    main()
