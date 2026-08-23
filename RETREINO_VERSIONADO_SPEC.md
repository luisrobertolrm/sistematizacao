# Sistematização — Re-treino versionado por execução (spec para Claude Code)

> Documento de implementação. Cole no Claude Code e peça para aplicar no repositório
> `C:\Git\Sistematizacao`. **Estado atual do repo: NENHUMA das mudanças abaixo está
> aplicada** — tentativas anteriores de edição via MCP falharam por timeout, então
> aplique tudo do zero. Onde este doc diz "código que já projetamos", significa que o
> código foi desenhado nas conversas anteriores mas ainda não está no repo.

---

## 1. Objetivo

Validar (com dados fake, mas em arquitetura idêntica à de produção) o ciclo:

1. A API recebe/gera dados novos e os acumula.
2. Uma chamada de API re-treina o modelo usando o **mesmo método** do treino da etapa 7,
   sobre `dataset original + dados novos`.
3. Cada treino vira um **artefato versionado no host** (sob git): modelo + metadados + EDA.
4. Um modelo é escolhido como **produção** (automático, se liberado; ou via **aprovação
   humana**), e a API passa a servi-lo **sem reiniciar o container**.

O rótulo `Churn` dos dados fake sai do próprio modelo (circular). **Isso é irrelevante aqui**:
o que se valida é a arquitetura, não a qualidade do modelo. Em produção, o passo 1 vira
ingestão da fonte real (banco/warehouse), sem persistir CSV.

---

## 2. Decisões de arquitetura e o porquê de cada uma

| Decisão | Porquê |
|---|---|
| **Um `run_id` por treino** = `run_<timestampUTC>_<uuid8>` (ex.: `run_20260820T143022Z_a1b2c3d4`) | Timestamp torna o id **ordenável** e legível; o uuid curto garante **unicidade** mesmo em execuções no mesmo segundo ou em paralelo. |
| **Pasta por execução**: `treinamentos/<run_id>/` e `avaliacoes/<eval_id>/` | Mantém o artefato ligado ao dado que o gerou: auditoria, comparação de métricas entre runs, e **rollback** (voltar produção para um run anterior). EDA por run reflete o dataset daquele treino. |
| **Produção em caminho FIXO** (`modelos/modelo_churn.joblib`) | A API não muda a forma de carregar. Promover é só copiar o run → caminho fixo + reload. Rollback = promover outro `run_id`. |
| **Escrita atômica** (`.tmp` + `os.replace`) | Um `/prever` concorrente nunca lê um `.joblib` pela metade durante a troca. `os.replace` é atômico no mesmo filesystem. |
| **`MANIFESTO.json`** aponta a produção + histórico | Fonte única de verdade sobre qual run está no ar e o que já foi treinado/avaliado. |
| **`modelos/` como bind mount** para a pasta do repo (não named volume) | Só um bind mount cai numa pasta real do repo → visível ao **git no host**. Named volume vive na área do Docker, fora do git. |
| **Commit roda no host**, fora do container | Credencial de git **nunca** entra no container. Quem dispara o treino (host/systemd) commita depois. |
| **Treino síncrono no endpoint (demo)** | Simples para apresentar. Em produção, mover para tarefa em background — Optuna (40 trials) + CV bloqueia o worker HTTP por minutos. |
| **EDA gerada a partir do dado do run** | A EDA precisa descrever o dataset que efetivamente treinou aquele modelo (original + novos), não o dataset base. |

---

## 3. Layout de pastas (tudo sob `DIR_MODELOS`, bind-montado no host/git)

```
modelos/
├── treinamentos/
│   └── run_20260820T143022Z_a1b2c3d4/
│       ├── modelo_churn.joblib        # binário do run
│       ├── metadados.json             # run_id, params, métricas, treinado_em, n_amostras, novos usados
│       └── eda/
│           ├── 01_distribuicao_alvo.png
│           ├── 02_numericas_por_churn.png
│           └── 03_categoricas_por_churn.png
├── avaliacoes/
│   └── eval_20260820T150000Z_ff00 aa11/
│       └── observacao_lotes.json      # + qual run_id foi avaliado
├── modelo_churn.joblib                # PRODUÇÃO (o que a API carrega) — cópia do run promovido
├── modelo_churn.json                  # metadados de produção
└── MANIFESTO.json                     # { producao, treinamentos[], avaliacoes[] }
```

---

## 4. Alterações no Python

### 4.1 Novo módulo `sistematizacao/runs.py` (gestão de execuções)

```python
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
from pathlib import Path
from typing import Any

from sistematizacao.carregamento_inicial import (
    ARQUIVO_METADADOS,
    ARQUIVO_MODELO,
    DIR_MODELOS,
)

DIR_TREINAMENTOS = DIR_MODELOS / "treinamentos"
DIR_AVALIACOES = DIR_MODELOS / "avaliacoes"
ARQUIVO_MANIFESTO = DIR_MODELOS / "MANIFESTO.json"


def _carimbo() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def novo_run_id() -> str:
    """Ex.: run_20260820T143022Z_a1b2c3d4 (ordenavel + unico)."""
    return f"run_{_carimbo()}_{uuid.uuid4().hex[:8]}"


def novo_eval_id() -> str:
    return f"eval_{_carimbo()}_{uuid.uuid4().hex[:8]}"


def dir_treinamento(run_id: str) -> Path:
    """Cria (se preciso) e devolve modelos/treinamentos/<run_id>/ com subpasta eda/."""
    destino = DIR_TREINAMENTOS / run_id
    (destino / "eda").mkdir(parents=True, exist_ok=True)
    return destino


def dir_avaliacao(eval_id: str) -> Path:
    destino = DIR_AVALIACOES / eval_id
    destino.mkdir(parents=True, exist_ok=True)
    return destino


def escrever_json_atomico(caminho: Path, dados: dict[str, Any]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    tmp = caminho.with_name(caminho.name + ".tmp")
    tmp.write_text(json.dumps(dados, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, caminho)


def ler_manifesto() -> dict[str, Any]:
    if not ARQUIVO_MANIFESTO.exists():
        return {"producao": None, "treinamentos": [], "avaliacoes": []}
    return json.loads(ARQUIVO_MANIFESTO.read_text(encoding="utf-8"))


def registrar_treinamento(run_id: str, info: dict[str, Any]) -> None:
    manifesto = ler_manifesto()
    manifesto["treinamentos"].append({"run_id": run_id, **info})
    escrever_json_atomico(ARQUIVO_MANIFESTO, manifesto)


def registrar_avaliacao(eval_id: str, info: dict[str, Any]) -> None:
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
    if not joblib_run.exists():
        raise FileNotFoundError(f"Run sem binario: {joblib_run}")

    for src, dst in ((joblib_run, ARQUIVO_MODELO), (meta_run, ARQUIVO_METADADOS)):
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)

    manifesto = ler_manifesto()
    manifesto["producao"] = run_id
    escrever_json_atomico(ARQUIVO_MANIFESTO, manifesto)
```

### 4.2 `carregamento_inicial.py` — juntar original + novos

Adicione `carregar_novos()` logo após `carregar_bruto()`:

```python
def carregar_novos() -> list[pd.DataFrame]:
    """Le os CSVs de dados/novos/ (alimentados via POST /dados/fake).

    Mesmas 21 colunas do original -> concatenaveis antes da limpeza. Em
    producao, esta funcao seria trocada pela leitura da fonte real (banco).
    """
    dir_novos = DIR_DADOS / "novos"
    if not dir_novos.exists():
        return []
    return [pd.read_csv(c) for c in sorted(dir_novos.glob("*.csv"))]
```

Troque a assinatura e o início de `preparar_tudo`:

```python
def preparar_tudo(incluir_novos: bool = False) -> Dados:
    """Pipeline completo de preparacao.

    incluir_novos=True concatena dados/novos/ ao dataset original antes da
    limpeza (usado no re-treino via API). Default False mantem EDA, observacao
    e treino manual inalterados.
    """
    bruto = carregar_bruto()
    if incluir_novos:
        extras = carregar_novos()
        if extras:
            bruto = pd.concat([bruto, *extras], ignore_index=True)
            print(f"Incluidos {len(extras)} arquivo(s) de dados/novos -> {len(bruto)} linhas")
    df = aplicar_engenharia_atributos(limpar(bruto))
    X, y = separar_x_y(df)
    # ...resto igual (pre, split, cv, return Dados(...))
```

### 4.3 `eda.py` — permitir gerar a EDA numa pasta arbitrária

Parametrize `_salvar` e as três funções de plot com `dir_saida` (default = `DIR_RELATORIOS`,
mantendo o `main()` atual funcionando), e adicione `gerar_eda`:

```python
def _salvar(fig: plt.Figure, nome: str, dir_saida: Path = DIR_RELATORIOS) -> Path:
    dir_saida.mkdir(parents=True, exist_ok=True)
    destino = dir_saida / nome
    fig.tight_layout()
    fig.savefig(destino, dpi=120)
    plt.close(fig)
    print(f"  -> grafico salvo em {destino}")
    return destino


# nas assinaturas, receba dir_saida e repasse ao _salvar:
def distribuicao_alvo(df: pd.DataFrame, dir_saida: Path = DIR_RELATORIOS) -> None:
    # ...igual, mas: _salvar(fig, "01_distribuicao_alvo.png", dir_saida)
    ...

def numericas_por_churn(df: pd.DataFrame, dir_saida: Path = DIR_RELATORIOS) -> None:
    # ..._salvar(fig, "02_numericas_por_churn.png", dir_saida)
    ...

def categoricas_por_churn(df: pd.DataFrame, dir_saida: Path = DIR_RELATORIOS) -> None:
    # ..._salvar(fig, "03_categoricas_por_churn.png", dir_saida)
    ...


def gerar_eda(df: pd.DataFrame, dir_saida: Path) -> None:
    """Gera os 3 PNGs da EDA a partir de um df ja limpo, salvando em dir_saida."""
    distribuicao_alvo(df, dir_saida)
    numericas_por_churn(df, dir_saida)
    categoricas_por_churn(df, dir_saida)
```

> Observação: `gerar_eda` recebe o `df` do run (com original + novos). `dados.df` devolvido
> por `preparar_tudo` já serve — contém todas as colunas que a EDA usa.

### 4.4 `treinamento_avaliacao.py` — treino vira "publicar run"

Adicione `import os` (abaixo de `import json`) e os imports do `runs`:

```python
from sistematizacao.carregamento_inicial import DIR_DADOS  # junto aos imports existentes
from sistematizacao.eda import gerar_eda
from sistematizacao.runs import (
    dir_treinamento,
    escrever_json_atomico,
    novo_run_id,
    promover,
    registrar_treinamento,
)
```

**Remova o antigo `salvar()`** (que gravava direto em produção) e adicione `salvar_run` +
`treinar_publicar`. O modelo agora é gravado **dentro do run**; produção só é tocada por
`promover()`.

```python
def salvar_run(
    pipe: Pipeline,
    params: dict[str, Any],
    metricas: dict[str, float],
    run_id: str,
    destino: Path,
    n_amostras: int,
    incluiu_novos: bool,
    arquivos_novos: list[str],
) -> None:
    """Grava modelo + metadados DENTRO de treinamentos/<run_id>/ (atomico)."""
    destino.mkdir(parents=True, exist_ok=True)

    joblib_dst = destino / "modelo_churn.joblib"
    tmp = joblib_dst.with_name(joblib_dst.name + ".tmp")
    joblib.dump(pipe, tmp)
    os.replace(tmp, joblib_dst)

    escrever_json_atomico(
        destino / "metadados.json",
        {
            "run_id": run_id,
            "nome": NOME_MODELO,
            "params": params,
            "metricas_teste": metricas,
            "treinado_em": datetime.now(UTC).isoformat(),
            "n_amostras": n_amostras,
            "incluiu_novos": incluiu_novos,
            "arquivos_novos_usados": arquivos_novos,
            "colunas_entrada": list(pipe.feature_names_in_)
            if hasattr(pipe, "feature_names_in_")
            else [],
        },
    )
    print(f"\nRun salvo em: {destino}")


def treinar_publicar(
    incluir_novos: bool = True,
    promover_auto: bool = True,
    n_trials: int = N_TRIALS,
) -> dict[str, Any]:
    """Mesmo caminho do treino da etapa 7, mas versionado por run.

    Grava treinamentos/<run_id>/ (modelo + metadados + eda). Se promover_auto,
    copia o run para producao. Devolve run_id + metricas.
    """
    run_id = novo_run_id()
    destino = dir_treinamento(run_id)

    arquivos_novos: list[str] = []
    if incluir_novos:
        dir_novos = DIR_DADOS / "novos"
        if dir_novos.exists():
            arquivos_novos = [p.name for p in sorted(dir_novos.glob("*.csv"))]

    dados = preparar_tudo(incluir_novos=incluir_novos)

    gerar_eda(dados.df, destino / "eda")        # EDA do dataset deste run

    melhor = otimizar(dados, n_trials)
    pipe = treinar(dados, melhor.params)
    metricas = avaliar(pipe, dados.X_test, dados.y_test)
    coeficientes(pipe)

    salvar_run(
        pipe, melhor.params, metricas, run_id, destino,
        n_amostras=int(len(dados.X)),
        incluiu_novos=incluir_novos,
        arquivos_novos=arquivos_novos,
    )
    registrar_treinamento(
        run_id,
        {"treinado_em": datetime.now(UTC).isoformat(), "metricas": metricas,
         "n_amostras": int(len(dados.X))},
    )

    if promover_auto:
        promover(run_id)

    return {
        "run_id": run_id,
        "promovido": promover_auto,
        "n_amostras": int(len(dados.X)),
        "params": melhor.params,
        "metricas": metricas,
    }
```

Atualize `main()` para usar o novo caminho (o treino manual também gera um run e promove):

```python
def main() -> None:
    resultado = treinar_publicar(incluir_novos=False, promover_auto=True)
    print("\nrun:", resultado["run_id"], "| promovido:", resultado["promovido"])
```

### 4.5 `observacao.py` — avaliação vira run próprio

Importe `novo_eval_id, dir_avaliacao, registrar_avaliacao, ler_manifesto` de `runs` e troque
`salvar_historico` para gravar em `avaliacoes/<eval_id>/`, registrando qual modelo foi avaliado:

```python
def salvar_historico(historico: list[dict[str, float]]) -> None:
    from datetime import UTC, datetime
    from sistematizacao.runs import (
        dir_avaliacao, novo_eval_id, registrar_avaliacao, ler_manifesto,
    )

    eval_id = novo_eval_id()
    destino = dir_avaliacao(eval_id)
    (destino / "observacao_lotes.json").write_text(
        json.dumps(historico, indent=2), encoding="utf-8"
    )
    run_producao = ler_manifesto().get("producao")
    registrar_avaliacao(
        eval_id,
        {"avaliado_em": datetime.now(UTC).isoformat(),
         "run_avaliado": run_producao,
         "f1_macro_final": historico[-1]["f1_macro"] if historico else None},
    )
    print(f"Avaliacao salva em: {destino} (modelo em producao: {run_producao})")
```

### 4.6 `web.py` — endpoints de treino, promoção e histórico

Imports já presentes: `Annotated`, `Query`, `limpar_cache`, `carregar_modelo`. Adicione os
endpoints antes de `def main()`:

```python
@app.post("/modelo/treinar", summary="Re-treina (original + novos) e gera um run versionado")
def treinar_modelo(
    incluir_novos: Annotated[bool, Query(description="Concatena dados/novos ao treino")] = True,
    promover: Annotated[bool, Query(description="Promove o run p/ producao ao final")] = True,
) -> dict[str, object]:
    """SINCRONO: roda o treino da etapa 7, versiona o run e (opcional) promove.

    Bloqueia por alguns minutos (Optuna + CV). Em producao, mover para tarefa
    em background para nao segurar o worker HTTP.
    """
    from sistematizacao.treinamento_avaliacao import treinar_publicar  # noqa: PLC0415

    resultado = treinar_publicar(incluir_novos=incluir_novos, promover_auto=promover)
    if promover:
        limpar_cache()
        carregar_modelo()
    return {"status": "treinado", **resultado}


@app.post("/modelo/promover/{run_id}", summary="Promove um run para producao (aprovacao humana)")
def promover_modelo(run_id: str) -> dict[str, str]:
    from sistematizacao.runs import promover  # noqa: PLC0415

    try:
        promover(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    limpar_cache()
    carregar_modelo()
    return {"status": "promovido", "run_id": run_id}


@app.get("/modelo/runs", summary="Historico de treinamentos e avaliacoes")
def listar_runs() -> dict[str, object]:
    from sistematizacao.runs import ler_manifesto  # noqa: PLC0415

    return ler_manifesto()
```

> Opcional (revisão de EDA pela gestão): servir os PNGs com
> `GET /modelo/runs/{run_id}/eda/{arquivo}` via `fastapi.responses.FileResponse`,
> validando que `arquivo` não escapa da pasta do run.

---

## 5. Alterações no Docker

### 5.1 `docker-compose.yml` — bind mount de `modelos/`

O named volume não serve: o modelo precisa cair numa pasta real do repo para o git do host
enxergar. Adicione o bind mount (mantendo o volume dos CSVs):

```yaml
    volumes:
      - dados_novos:/app/dados/novos          # CSVs do /dados/fake (demo; fora do git)
      - ./python/modelos:/app/modelos         # bind -> pasta do repo (git): runs + producao + manifesto
```

> O bind mount sombreia o `COPY modelos ./modelos` em runtime. Como `./python/modelos/` já
> tem o binário seed versionado, são os mesmos arquivos — agora graváveis de volta ao host.
> O `COPY` continua servindo de fallback para rodar sem o mount.

### 5.2 `Dockerfile`

Nenhuma mudança obrigatória. O `COPY modelos ./modelos` e `ENV DIR_MODELOS=/app/modelos`
permanecem. As subpastas `treinamentos/`, `avaliacoes/` são criadas em runtime pelo código.

### 5.3 `.gitignore`

Queremos os binários **versionados**, então não ignore `modelos/`. Ignore só os temporários:

```gitignore
python/modelos/**/*.tmp
```

> Atenção prod: `.joblib` é blob binário; cada run incha o repositório. Para produção,
> migrar `modelos/` para **Git LFS** ou um artifact store. Para a apresentação, git puro serve.

### 5.4 Permissões

O container grava no bind mount como o uid do processo; os arquivos no host ficam com esse
uid. O `git commit` no host funciona mesmo assim (git não depende de dono). Se houver ruído
de permissão, alinhar o usuário do container ao do host (`user:` no compose).

---

## 6. Commit no host + agendamento

O container **não** commita (sem credencial de git dentro dele). Quem dispara o treino
commita depois. Script no host (`/srv/Sistematizacao/retreinar.sh`):

```bash
#!/usr/bin/env bash
set -euo pipefail
PORTA="${APP_PORT:-8005}"
REPO="/srv/Sistematizacao"     # ajuste ao caminho real do repo no servidor

# 1) treina (gera run) e promove automaticamente
curl -fsS -X POST "http://localhost:${PORTA}/modelo/treinar?incluir_novos=true&promover=true"

# 2) versiona os artefatos no git do host
cd "$REPO"
git add python/modelos
git commit -m "modelo: retrain $(date -u +%FT%TZ)" && git push

# 3) garante o reload a quente (o endpoint ja recarrega quando promover=true)
curl -fsS -X POST "http://localhost:${PORTA}/modelo/recarregar"
```

Agendamento via systemd timer (host):

```ini
# /etc/systemd/system/retreino.service
[Unit]
Description=Re-treino do modelo de churn
After=docker.service

[Service]
Type=oneshot
ExecStart=/srv/Sistematizacao/retreinar.sh
```

```ini
# /etc/systemd/system/retreino.timer
[Unit]
Description=Dispara o re-treino periodicamente

[Timer]
OnCalendar=weekly
Persistent=true

[Install]
WantedBy=timers.target
```

`sudo systemctl enable --now retreino.timer`

> Só habilite o modo automático se o professor liberar treino automático. Caso contrário,
> use o **modo com aprovação** (seção 7.2).

---

## 7. Fluxos

### 7.1 Automático (professor liberou)

1. `POST /dados/fake` (uma ou mais vezes) → acumula em `dados/novos/`.
2. `POST /modelo/treinar?incluir_novos=true&promover=true` → gera `run_id`, grava
   `treinamentos/<run_id>/` (modelo + metadados + eda), promove p/ produção, recarrega.
3. Host commita `python/modelos` no git.
4. `GET /modelo` mostra o `treinado_em` novo; `POST /prever` já usa o modelo novo.

### 7.2 Com aprovação humana (gate)

1. `POST /modelo/treinar?incluir_novos=true&promover=false` → gera o run **sem** promover.
2. Gestor revisa: `GET /modelo/runs` (métricas) + os PNGs em `treinamentos/<run_id>/eda/`.
3. Aprovou → `POST /modelo/promover/{run_id}` → copia p/ produção + reload.
4. Host commita `python/modelos`.

---

## 8. Checklist para o Claude Code

- [ ] Criar `sistematizacao/runs.py` (seção 4.1).
- [ ] `carregamento_inicial.py`: `carregar_novos()` + `preparar_tudo(incluir_novos=False)` (4.2).
- [ ] `eda.py`: parametrizar `_salvar` e plots com `dir_saida`; adicionar `gerar_eda` (4.3).
- [ ] `treinamento_avaliacao.py`: `import os`; remover `salvar()` antigo; adicionar
      `salvar_run` + `treinar_publicar`; atualizar `main()` (4.4).
- [ ] `observacao.py`: `salvar_historico` grava em `avaliacoes/<eval_id>/` + registra (4.5).
- [ ] `web.py`: endpoints `/modelo/treinar`, `/modelo/promover/{run_id}`, `/modelo/runs` (4.6).
- [ ] `docker-compose.yml`: bind mount `./python/modelos:/app/modelos` (5.1).
- [ ] `.gitignore`: `python/modelos/**/*.tmp` (5.3).
- [ ] Host: `retreinar.sh` + systemd timer (seção 6) — só no modo automático.
- [ ] Testar: `POST /dados/fake` → `POST /modelo/treinar` → conferir `treinamentos/<run_id>/`
      com modelo + metadados + eda, e `MANIFESTO.json` apontando produção.
