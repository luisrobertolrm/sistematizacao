"""Versionamento de artefatos por execucao (treinamento e avaliacao).

Cada treinamento gera um run_id unico e grava modelo + metadados + EDA em
modelos/treinamentos/<run_id>/. Cada avaliacao grava em
modelos/avaliacoes/<eval_id>/. O MANIFESTO.json aponta qual run esta em
producao. Tudo sob DIR_MODELOS -> bind mount -> host/git.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sistematizacao.carregamento_inicial import (
    ARQUIVO_METADADOS,
    ARQUIVO_MODELO,
    DIR_MODELOS,
)

if TYPE_CHECKING:
    from pathlib import Path

DIR_TREINAMENTOS = DIR_MODELOS / "treinamentos"
DIR_AVALIACOES = DIR_MODELOS / "avaliacoes"
ARQUIVO_MANIFESTO = DIR_MODELOS / "MANIFESTO.json"


def _carimbo() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def novo_run_id() -> str:
    """Ex.: run_20260820T143022Z_a1b2c3d4 (ordenavel + unico)."""
    return f"run_{_carimbo()}_{uuid.uuid4().hex[:8]}"


def novo_eval_id() -> str:
    """Mesmo formato do run_id, com prefixo eval_."""
    return f"eval_{_carimbo()}_{uuid.uuid4().hex[:8]}"


def dir_treinamento(run_id: str) -> Path:
    """Cria (se preciso) e devolve modelos/treinamentos/<run_id>/ com subpasta eda/."""
    destino = DIR_TREINAMENTOS / run_id
    (destino / "eda").mkdir(parents=True, exist_ok=True)
    return destino


def dir_avaliacao(eval_id: str) -> Path:
    """Cria (se preciso) e devolve modelos/avaliacoes/<eval_id>/."""
    destino = DIR_AVALIACOES / eval_id
    destino.mkdir(parents=True, exist_ok=True)
    return destino


def escrever_json_atomico(caminho: Path, dados: dict[str, Any]) -> None:
    """Grava JSON via arquivo temporario + os.replace (leitor nunca ve meia escrita)."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    tmp = caminho.with_name(caminho.name + ".tmp")
    tmp.write_text(json.dumps(dados, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, caminho)  # noqa: PTH105


def ler_manifesto() -> dict[str, Any]:
    """Le o MANIFESTO.json; devolve o esqueleto vazio se ainda nao existe."""
    if not ARQUIVO_MANIFESTO.exists():
        return {"producao": None, "treinamentos": [], "avaliacoes": []}
    return json.loads(ARQUIVO_MANIFESTO.read_text(encoding="utf-8"))


def registrar_treinamento(run_id: str, info: dict[str, Any]) -> None:
    """Acrescenta o run ao historico de treinamentos do manifesto."""
    manifesto = ler_manifesto()
    manifesto["treinamentos"].append({"run_id": run_id, **info})
    escrever_json_atomico(ARQUIVO_MANIFESTO, manifesto)


def registrar_avaliacao(eval_id: str, info: dict[str, Any]) -> None:
    """Acrescenta a avaliacao ao historico do manifesto."""
    manifesto = ler_manifesto()
    manifesto["avaliacoes"].append({"eval_id": eval_id, **info})
    escrever_json_atomico(ARQUIVO_MANIFESTO, manifesto)


def promover(run_id: str) -> None:
    """Copia o modelo do run para os caminhos de PRODUCAO (atomico) e atualiza o manifesto.

    Rollback = chamar promover() com um run_id anterior.
    """
    origem = DIR_TREINAMENTOS / run_id
    joblib_run = origem / "modelo_churn.joblib"
    meta_run = origem / "metadados.json"
    for arquivo in (joblib_run, meta_run):
        if not arquivo.exists():
            msg = f"Run incompleto, falta {arquivo.name}: {arquivo}"
            raise FileNotFoundError(msg)

    for src, dst in ((joblib_run, ARQUIVO_MODELO), (meta_run, ARQUIVO_METADADOS)):
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)  # noqa: PTH105

    manifesto = ler_manifesto()
    manifesto["producao"] = run_id
    escrever_json_atomico(ARQUIVO_MANIFESTO, manifesto)
    print(f"Producao aponta agora para: {run_id}")
