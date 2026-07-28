"""
Organiza os videos brutos do V-LIBRASIL na convencao que dataset_builder.py
espera: data/raw/<DATASET>/<SINAL>/<arquivo>_<sinalizador>.mp4

Le "videos UFPE (V-LIBRASIL)/annotations.csv" (colunas video_id, video_name,
class, user_id, ...) e cria, para cada video, um link simbolico (ou copia,
com --copiar) em:

    data/raw/VLIBRASIL/<class>/<video_id sem extensao>_<user_id>.mp4

video_id vira o "arquivo" porque ja e um identificador unico e sem acento/
espaco/parenteses, evitando repetir o nome do sinal (que ja e o nome da
pasta) num nome de arquivo com caracteres estranhos.

Videos listados em error.csv (download quebrado: 0x0, fps 0) sao pulados.

Uso:
    python scripts/organizar_vlibrasil.py
    python scripts/organizar_vlibrasil.py --copiar
"""

import argparse
import csv
import shutil
from pathlib import Path


_ILEGAIS = '"*:<>?\\|'


def _canonico(nome: str) -> str:
    """
    Forma comparavel de um nome de arquivo: remove pontuacao ilegal em
    Windows e qualquer caractere de Area de Uso Privado do Unicode
    (U+E000-U+F8FF). Alguns downloads desse dataset trocam caracteres
    ilegais por substitutos nessa faixa (ex.: "?" vira "" no disco,
    visualmente identico a "?" no terminal mas um codepoint diferente) —
    comparar string literal falha silenciosamente nesses casos.
    """
    return "".join(
        c for c in nome if c not in _ILEGAIS and not (0xE000 <= ord(c) <= 0xF8FF)
    )


def _ler_ids_com_erro(caminho_error_csv: Path) -> set[str]:
    if not caminho_error_csv.exists():
        return set()
    with caminho_error_csv.open(newline="", encoding="utf-8-sig") as f:
        return {linha["video_id"] for linha in csv.DictReader(f)}


def organizar(origem: Path, destino: Path, copiar: bool) -> dict:
    anotacoes = origem / "annotations.csv"
    pasta_videos = origem / "data"
    ids_com_erro = _ler_ids_com_erro(origem / "error.csv")
    indice_videos = {_canonico(p.name): p for p in pasta_videos.iterdir()}

    criados = pulados_erro = pulados_ausente = pulados_existente = 0

    with anotacoes.open(newline="", encoding="utf-8-sig") as f:
        for linha in csv.DictReader(f):
            video_id = linha["video_id"]
            if video_id in ids_com_erro:
                pulados_erro += 1
                continue

            origem_arquivo = indice_videos.get(_canonico(linha["video_name"]))
            if origem_arquivo is None:
                pulados_ausente += 1
                continue

            sinal = linha["class"]
            sinalizador = linha["user_id"]
            arquivo = Path(video_id).stem

            pasta_sinal = destino / sinal
            pasta_sinal.mkdir(parents=True, exist_ok=True)
            destino_arquivo = pasta_sinal / f"{arquivo}_{sinalizador}.mp4"

            if destino_arquivo.exists() or destino_arquivo.is_symlink():
                pulados_existente += 1
                continue

            if copiar:
                shutil.copy2(origem_arquivo, destino_arquivo)
            else:
                destino_arquivo.symlink_to(origem_arquivo.resolve())
            criados += 1

    return {
        "criados": criados,
        "pulados_erro": pulados_erro,
        "pulados_ausente": pulados_ausente,
        "pulados_existente": pulados_existente,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--origem", type=Path, default=Path("videos UFPE (V-LIBRASIL)"))
    ap.add_argument("--destino", type=Path, default=Path("data/raw/VLIBRASIL"))
    ap.add_argument(
        "--copiar", action="store_true", help="copia os arquivos em vez de linkar"
    )
    args = ap.parse_args()

    resultado = organizar(args.origem, args.destino, args.copiar)

    print(f"links/copias criados:      {resultado['criados']}")
    print(f"pulados (error.csv):       {resultado['pulados_erro']}")
    print(f"pulados (arquivo ausente): {resultado['pulados_ausente']}")
    print(f"pulados (ja existia):      {resultado['pulados_existente']}")


if __name__ == "__main__":
    main()
