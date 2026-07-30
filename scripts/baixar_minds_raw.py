"""
Baixa SO o arquivo .zip bruto do MINDS-Libras no Kaggle, sem a extracao
automatica que kagglehub.dataset_download() faz por padrao.

Motivo: extrair tudo de uma vez precisaria do espaco do .zip (44.5GB) MAIS o
espaco dos arquivos extraidos ao mesmo tempo — nao cabe no disco disponivel.
Este script para logo apos o download, e o processamento video-a-video
(extrair 1, processar, apagar, proximo) fica pra outro script.

Usa funcoes internas do kagglehub (nao expostas na API publica) porque
dataset_download() nao tem opcao de pular a extracao.

Precisa da variavel de ambiente KAGGLE_API_TOKEN com o token da sua conta
(https://www.kaggle.com/settings/api). Nunca commitar o token no codigo.
"""

import os
import sys

from kagglehub.cache import Cache
from kagglehub.clients import build_kaggle_client, download_file
from kagglehub.config import set_kaggle_api_token
from kagglehub.exceptions import handle_call
from kagglehub.handle import parse_dataset_handle
from kagglehub.http_resolver import _build_dataset_download_request, _get_current_version

TOKEN = os.environ.get("KAGGLE_API_TOKEN")
if not TOKEN:
    sys.exit("defina a variavel de ambiente KAGGLE_API_TOKEN antes de rodar este script")

set_kaggle_api_token(TOKEN)

h = parse_dataset_handle("j0aopsantos/minds-libras")

with build_kaggle_client() as api_client:
    if not h.is_versioned():
        h = h.with_version(_get_current_version(api_client, h))

    cache = Cache()
    archive_path = cache.get_archive_path(h)
    os.makedirs(os.path.dirname(archive_path), exist_ok=True)
    print("Baixando para:", archive_path)

    r = _build_dataset_download_request(h, None)
    response = handle_call(lambda: api_client.datasets.dataset_api_client.download_dataset(r), h)
    download_file(response, archive_path, h)

print("Download concluido (sem extrair):", archive_path)
