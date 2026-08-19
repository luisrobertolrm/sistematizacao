# Sistematizacao - Previsao de Churn

Pipeline de ML derivado do notebook `sistematizacao_escolha_modelo2.ipynb`
(dataset [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)),
quebrado em modulos executaveis e exposto por uma API HTTP.

## Estrutura

| Arquivo | Etapa do notebook | O que faz |
|---|---|---|
| `src/sistematizacao/carregamento_inicial.py` | 1 e 3 | download/leitura do CSV, limpeza, engenharia de atributos, split treino/teste e pre-processador |
| `src/sistematizacao/eda.py` | 2 | analise exploratoria; graficos salvos em `relatorios/` |
| `src/sistematizacao/treinamento_avaliacao.py` | 7 | tuning da LogisticRegression com Optuna, avaliacao no hold-out e gravacao do binario |
| `src/sistematizacao/observacao.py` | 8 | alimenta o binario em lotes e acompanha a metrica acumulada |
| `src/sistematizacao/carregar_modelo.py` | — | carrega o `.joblib` e faz predicao sobre dados novos |
| `src/sistematizacao/web.py` | — | API FastAPI para enviar novos dados de analise |
| `modelos/` | — | binarios treinados (`modelo_churn.joblib` + metadados JSON) |

## Rodando local

```bash
uv sync --all-extras

# de onde vem o CSV, em ordem de prioridade:
#   1) variavel de ambiente CHURN_CSV
#   2) qualquer .csv em python/dados/
#   3) download via kagglehub (exige credenciais do Kaggle)

uv run python -m sistematizacao.eda                    # etapa 2
uv run python -m sistematizacao.treinamento_avaliacao  # etapa 7 -> gera modelos/modelo_churn.joblib
uv run python -m sistematizacao.observacao             # etapa 8
uv run python -m sistematizacao.web                    # API em http://localhost:8000/docs
```

## API

| Metodo | Rota | Descricao |
|---|---|---|
| GET | `/health` | processo de pe + se o binario existe |
| GET | `/modelo` | params, metricas e data do treino |
| POST | `/prever` | analisa um cliente |
| POST | `/prever/lote` | analisa uma lista de clientes |
| POST | `/modelo/recarregar` | recarrega o binario do disco apos novo treino |

```bash
curl -X POST http://localhost:8000/prever -H "Content-Type: application/json" -d '{
  "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 2,
  "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
  "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
  "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
  "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5
}'
```

O campo `tenure` e enviado cru: a API aplica a mesma engenharia de atributos do
treino (`faixa_tenure`) antes de chamar o modelo.

## Docker

```bash
docker build -t sistematizacao:local .
docker run --rm -p 8000:8000 -v "$PWD/modelos:/app/modelos" sistematizacao:local
```

A imagem embarca o conteudo de `modelos/`; o volume acima e util para trocar o
binario sem rebuildar.

## CI

`.github/workflows/ci.yml` roda ruff (lint + formatacao), smoke test dos modulos
e da API, e faz build da imagem — publicando em `ghcr.io` nos pushes para
`master`/`main` e em tags `v*`.
