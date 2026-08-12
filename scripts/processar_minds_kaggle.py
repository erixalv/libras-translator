"""
Processa o MINDS-Libras direto do .zip baixado (scripts/baixar_minds_raw.py),
sem nunca extrair tudo de uma vez: pra cada video dentro do zip, extrai SO
ele pra um arquivo temporario, roda o pipeline de landmarks,
salva o vetor (30, 260) resultante num cache proprio, e apaga o video
temporario antes de passar pro proximo.

Pico de disco extra durante a execucao: so o tamanho de 1 video por vez
(~30-55MB), nao o dataset inteiro extraido.

Convencao de nome dentro do zip: "<NN><Sinal>Sinalizador<SS>-<R>.mp4"
    ex: "01AcontecerSinalizador01-1.mp4"
        -> sinal = "Acontecer", sinalizador = "Sinalizador01", repeticao = "1"

Uso:
    python scripts/processar_minds_kaggle.py
    python scripts/processar_minds_kaggle.py --limite 50
"""

import argparse
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.landmarks.extrator_mediapipe import construir_sequencia, extrair_video

ARQUIVO_ZIP = Path.home() / ".cache/kagglehub/datasets/j0aopsantos/minds-libras/3.archive"
CACHE_DIR = Path("data/processed/cache_minds")

PADRAO_NOME = re.compile(r"^\d+(?P<sinal>.+?)(?P<sinalizador>Sinalizador\d+)-(?P<repeticao>\d+)\.mp4$")


def _parse_nome_minds(nome_no_zip: str) -> tuple[str, str, str] | None:
    m = PADRAO_NOME.match(nome_no_zip)
    if not m:
        return None
    return m["sinal"], m["sinalizador"], m["repeticao"]


def _caminho_cache(sinal: str, arquivo: str, sinalizador: str) -> Path:
    return CACHE_DIR / sinal / f"{arquivo}_{sinalizador}.npz"


def processar(limite: int | None = None, n_frames: int = 30) -> None:
    with zipfile.ZipFile(ARQUIVO_ZIP) as z:
        membros = z.namelist()
        print(f"{len(membros)} videos no zip")

        processados = 0
        pulados_cache = 0
        pulados_nome = 0
        descartes = 0

        for nome in membros:
            parsed = _parse_nome_minds(nome)
            if parsed is None:
                pulados_nome += 1
                continue
            sinal, sinalizador, repeticao = parsed
            arquivo = f"rep{repeticao}"

            caminho_cache = _caminho_cache(sinal, arquivo, sinalizador)
            if caminho_cache.exists():
                pulados_cache += 1
                continue

            if limite is not None and processados >= limite:
                continue

            caminho_cache.parent.mkdir(parents=True, exist_ok=True)

            with tempfile.TemporaryDirectory() as tmp:
                caminho_tmp = Path(tmp) / "video.mp4"
                with z.open(nome) as origem, open(caminho_tmp, "wb") as destino:
                    destino.write(origem.read())

                try:
                    registros, stats = extrair_video(str(caminho_tmp))
                except Exception as e:  # noqa: BLE001
                    print(f"  ERRO em {nome}: {e}")
                    descartes += 1
                    continue
                # caminho_tmp e apagado automaticamente ao sair do "with" (TemporaryDirectory)

            if not registros:
                np.savez_compressed(caminho_cache, motivo=np.array("pose nao detectada"))
                descartes += 1
                continue

            seq = construir_sequencia(registros, n_frames=n_frames, modo="reamostrar")
            np.savez_compressed(
                caminho_cache, seq=seq.astype(np.float32), taxa_mao=np.array(stats["taxa_mao"])
            )
            processados += 1

            if processados % 20 == 0:
                print(f"  {processados} processados nesta execucao...")

    print()
    print(f"processados agora: {processados}")
    print(f"pulados (ja no cache): {pulados_cache}")
    print(f"pulados (nome fora do padrao): {pulados_nome}")
    print(f"descartes (sem pose / erro): {descartes}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limite", type=int, default=None)
    ap.add_argument("--n-frames", type=int, default=30)
    args = ap.parse_args()
    processar(limite=args.limite, n_frames=args.n_frames)
