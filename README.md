# Sistematizacao — previsao de churn

Projeto da disciplina **Engenharia de Aprendizado de Maquina** (Trilha A, dataset A3).

O call center da Telecom Tal nao liga para toda a base: o modelo indica so os clientes com alta chance real de cancelar. A metrica operacional e a **precisao de Churn** (dos indicados, quantos de fato sairiam). As metas da rubrica A3 entram como piso:

- AUC-ROC >= 0,82
- Accuracy >= 0,80
- F1-Score >= 0,68 (tratado como **F1 macro**)

Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle/IBM). Apos limpeza: 7.032 clientes, ~26,6% de churn.

Modelo em producao (notebook `curso_2`): **Regressao Logistica** tunada, **limiar 0,545**. Random Forest e Gradient Boosting tiveram precisao maior na CV, mas falharam Accuracy e/ou F1 macro.

---

## O que tem neste repositorio

```
Sistematizacao/
├── docker-compose.yml          # API no servidor (porta APP_PORT, padrao 8005)
├── .env.production.example     # copie para .env no host (/DATA/sistematizacao/.env)
├── scripts/
│   ├── retreinar.sh            # job diario no HOST (nao roda no container)
│   ├── retreino.service        # systemd oneshot -> retreinar.sh
│   ├── retreino.timer          # dispara todo dia as 06:00
│   └── instalar-runner.sh      # self-hosted runner (opcional)
├── .github/workflows/
│   ├── ci.yml                  # ruff + smoke test + build da imagem (sem push)
│   └── deploy.yml              # rsync -> /DATA/sistematizacao + compose + timer
└── python/                     # codigo, Dockerfile, modelo versionado
    ├── src/sistematizacao/
    ├── modelos/                # producao + runs (bind mount no compose)
    ├── dados/                  # CSV Telco; dados/novos/ e gerado pela API
    └── postman/
```

Detalhe do pacote Python (Postman, dados fake, CLI): [`python/README.md`](python/README.md).

---

## Arquitetura

```
                    GitHub Actions (CI)
                    ruff + smoke + docker build
                              |
                    push em main -> deploy.yml (self-hosted)
                              |
                              v
              /DATA/sistematizacao  (rsync, preserva .env)
                              |
              docker compose up -d --build
                              |
              +---------------+----------------+
              | container sistematizacao-api   |
              | FastAPI :8000  (host :8005)    |
              | bind: python/modelos           |
              | volume: dados/novos            |
              +---------------+----------------+
                              ^
                              | curl localhost:8005
              +---------------+----------------+
              | HOST systemd                   |
              | retreino.timer  06:00 diario   |
              | retreino.service -> retreinar.sh
              | git commit/push python/modelos |
              +--------------------------------+
```

O systemd **nao** roda dentro do container. Credencial de git fica no host.

---

## Pipeline de ML (espelha o notebook)

| Modulo | Etapa | Funcao |
|---|---|---|
| `carregamento_inicial` | 1 e 3 | CSV, limpeza, `tenure` + `faixa_tenure`, split 80/20, ColumnTransformer |
| `eda` | 2 | graficos em `relatorios/` |
| `treinamento_avaliacao` | 4-7 | baseline, Optuna (Prec_churn), limiar, bundle `{pipe, threshold, modelo}` |
| `observacao` | 8 | alimenta o binario em lotes e acompanha metricas |
| `monitoramento` | — | drift, e-mail, retreino so se alarmar |
| `carregar_modelo` | — | le o bundle e prediz |
| `gerador_fake` | — | CSV sintetico (Faker) em `dados/novos/` |
| `web` | — | FastAPI |
| `runs` | — | `run_id`, promover, MANIFESTO.json |

Atributos removidos na EDA: `customerID`, `gender`, `TotalCharges`. **`tenure` permanece**.

Criterio Optuna: maior Precisao de Churn **entre** os que tem Acc >= 0,80 e AUC >= 0,82.

---

## Rodar local (sem Docker)

Requer **Python 3.12** e [uv](https://docs.astral.sh/uv/).

```bash
cd python
uv sync --all-extras

# CSV, nesta ordem: CHURN_CSV  ->  python/dados/*.csv  ->  kagglehub

uv run python -m sistematizacao.eda
uv run python -m sistematizacao.treinamento_avaliacao   # gera modelos/modelo_churn.joblib
uv run python -m sistematizacao.observacao
uv run python -m sistematizacao.web                     # http://127.0.0.1:8000/docs
```

Scripts equivalentes (depois do `uv sync`):

```bash
uv run sistematizacao-eda
uv run sistematizacao-treino
uv run sistematizacao-observacao
uv run sistematizacao-monitor
uv run sistematizacao-web
```

---

## API em producao

A API publica tem tres GETs de inspecao: `/health` confirma que o servico e o binario estao no ar; `/modelo` mostra o modelo atual (regressao logistica tunada e as metricas); `/modelo/runs` lista os treinamentos versionados e aponta qual run esta em producao.

### `/health`

[https://sistematizacao.myworklab.com.br/health](https://sistematizacao.myworklab.com.br/health)

Diz se a API esta no ar e se o arquivo do modelo existe no disco.

Resposta atual: `status: ok` e `modelo_disponivel: true`. Nao avalia qualidade do modelo — so confirma que o processo e o binario estao acessiveis.

### `/modelo`

[https://sistematizacao.myworklab.com.br/modelo](https://sistematizacao.myworklab.com.br/modelo)

Mostra **qual modelo esta em producao**: nome, hiperparametros, metricas do teste, data do treino e colunas usadas.

Hoje: **LogisticRegression (tunada)**, `run_id` `run_20260821T134624Z_94a0479d`, treinado em 21/08/2026, 7.042 amostras (base + um CSV fake). Metricas no hold-out: Acc ~0,79, AUC ~0,82, F1 macro ~0,70.

### `/modelo/runs`

[https://sistematizacao.myworklab.com.br/modelo/runs](https://sistematizacao.myworklab.com.br/modelo/runs)

E o **historico versionado** (`MANIFESTO.json`): qual `run_id` esta em producao e a lista de treinamentos (e avaliacoes, se houver). Serve para auditoria e rollback (`POST /modelo/promover/{run_id}`).

Hoje ha um treino na lista, o mesmo que esta em producao; `avaliacoes` esta vazia.

---

## API

Docs interativas: `http://<host>:<porta>/docs` (local **8000**; servidor em geral **8005**).

| Metodo | Rota | O que faz |
|---|---|---|
| GET | `/health` | processo no ar + se o binario existe |
| GET | `/modelo` | metadados do treino em producao |
| POST | `/prever` | um cliente; limiar padrao = o calibrado no treino |
| POST | `/prever/lote` | lista de clientes |
| POST | `/dados/fake` | gera CSV sintetico em `dados/novos/` |
| GET | `/dados/novos` | lista esses CSVs |
| POST | `/modelo/recarregar` | rele o `.joblib` sem reiniciar o container |
| POST | `/modelo/treinar` | retreina (original + novos); `promover=true` opcional |
| POST | `/modelo/monitorar` | job diario: drift, e-mail, retreino condicionado |
| POST | `/modelo/promover/{run_id}` | aponta producao para um run (rollback) |
| GET | `/modelo/runs` | MANIFESTO (producao + historico) |
| GET | `/modelo/runs/{run_id}/eda/{arquivo}` | PNG da EDA daquele run |

`tenure` vai cru; a API aplica `faixa_tenure` igual ao treino.

```bash
curl -X POST "http://127.0.0.1:8005/prever" -H "Content-Type: application/json" -d "{
  \"SeniorCitizen\": 0, \"Partner\": \"No\", \"Dependents\": \"No\", \"tenure\": 2,
  \"PhoneService\": \"Yes\", \"MultipleLines\": \"No\",
  \"InternetService\": \"Fiber optic\", \"OnlineSecurity\": \"No\",
  \"OnlineBackup\": \"No\", \"DeviceProtection\": \"No\", \"TechSupport\": \"No\",
  \"StreamingTV\": \"Yes\", \"StreamingMovies\": \"Yes\",
  \"Contract\": \"Month-to-month\", \"PaperlessBilling\": \"Yes\",
  \"PaymentMethod\": \"Electronic check\", \"MonthlyCharges\": 95.5
}"
```

Colecao Postman: `python/postman/`.

**Dados fake:** o rotulo `Churn` sai do proprio modelo. Serve para exercitar ingestao/retreino, **nao** para medir qualidade.

---

## Docker no servidor

```bash
cp .env.production.example .env   # ajuste APP_PORT e SMTP se quiser
docker compose up -d --build
```

- Container: `sistematizacao-api`, porta interna 8000, host `${APP_PORT:-8005}`
- `python/modelos` e bind mount (runs versionados visiveis no git do host)
- `dados/novos` e volume Docker (CSVs fake sobrevivem a redeploy)

---

## Monitoramento diario (systemd no host)

Isto **nao** roda dentro do container. O systemd e do Linux do servidor; o script usa `curl` na porta do host, o `.env` e `git` **fora** da imagem.

### Cadeia

```
06:00  retreino.timer
         |
      retreino.service   (oneshot, WorkingDirectory=/DATA/sistematizacao)
         |                EnvironmentFile=-/DATA/sistematizacao/.env
      scripts/retreinar.sh
         |
      POST /modelo/monitorar?retreinar=true     <-- quem decide
         |
      git add/commit/push python/modelos        <-- so se o retreino gerou artefato
         |
      POST /modelo/recarregar
```

O **timer** dispara todo dia as **06:00**. `Persistent=true`: se o servidor estava desligado nesse horario, o job roda na proxima ligacao.

O **service** so executa `/DATA/sistematizacao/scripts/retreinar.sh`.

### O que o `retreinar.sh` faz

1. `curl POST /modelo/monitorar?retreinar=true` (espera ate 1 h: Optuna e lento).
2. Se `python/modelos` mudou, commit + push no git do host; senao nao commita.
3. `POST /modelo/recarregar` — a API rele o binario sem `docker compose restart`.

`set -euo pipefail`: se a API estiver fora ou o curl falhar, o job para e nao mexe no git.

### O que `/modelo/monitorar` decide

Avalia o modelo de **producao** em `dados/novos/` (se houver CSV) ou no hold-out do Telco.

| Situacao | Resultado |
|---|---|
| Sem drift | nao treina; o modelo atual permanece |
| Drift | e-mail (se SMTP estiver no `.env`) -> retreina -> promove **so** se Acc >= 0,80 e AUC >= 0,82 |

Drift e qualquer um destes:

- Precisao de Churn < 0,65
- AUC < 0,80
- fracao prevista como churn longe de ~17% (faixa 12-22%)

**Nao** treina todo dia por rotina. Sem SMTP (`SMTP_HOST` / `EMAIL_TO`), o alarme so aparece no log:

```bash
journalctl -u retreino -n 80
```

### Quem instala o timer

O workflow `.github/workflows/deploy.yml`, **depois** do `docker compose up` e do `/health`: copia os unit files para `/etc/systemd/system/`, `daemon-reload` e `enable --now retreino.timer`.

Nao coloque esses `systemctl` no entrypoint da imagem.

Conferir no servidor:

```bash
systemctl list-timers retreino.timer
systemctl is-enabled retreino.timer
```

---

## Variaveis de ambiente (`.env` no host)

| Variavel | Uso |
|---|---|
| `APP_PORT` | porta publicada (padrao 8005) |
| `SMTP_HOST` | se vazio, nao envia e-mail |
| `SMTP_PORT` | padrao 587 |
| `SMTP_USER` / `SMTP_PASSWORD` / `SMTP_FROM` | autenticacao SMTP |
| `EMAIL_TO` | destinatarios, separados por virgula |
| `CHURN_CSV` | (local) forca o CSV de treino |
| `DIR_MODELOS` / `DIR_DADOS` | caminhos dentro do container (compose ja define) |

O rsync do deploy **nao** sobrescreve `.env` (`--exclude='.env'`).

---

## CI e deploy

- **CI** (GitHub-hosted): ruff, import dos modulos, `GET /health`, build da imagem **sem** publicar em registry. Quem constroi a imagem que roda e o servidor, no deploy.
- **Deploy** (self-hosted): `rsync` para `/DATA/sistematizacao/`, `docker compose up -d --build`, espera `/health`, instala o timer systemd.

---

## Versionamento do modelo

Cada treino vira `python/modelos/treinamentos/<run_id>/` (joblib + metadados + EDA). Producao e o caminho fixo `modelo_churn.joblib` + `modelo_churn.json`. Rollback = `POST /modelo/promover/{run_id}` de um run anterior.

```
python/modelos/
├── treinamentos/<run_id>/     # binario + metadados.json + eda/
├── avaliacoes/<eval_id>/
├── modelo_churn.joblib        # PRODUCAO (o que a API carrega)
├── modelo_churn.json
└── MANIFESTO.json             # producao atual + historico
```

O container **nao** commita. O `retreinar.sh` no host versiona `python/modelos` depois do monitorar.
