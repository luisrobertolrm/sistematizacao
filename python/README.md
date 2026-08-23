# Pacote Python (`sistematizacao`)

Documentacao do projeto (arquitetura, API, Docker, systemd): **[README na raiz](../README.md)**.

Pipeline de ML derivado do notebook `sistematizacao_escolha_modelo2_bkp_curso_2`
([Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)).

## Estrutura

| Arquivo | Etapa do notebook | O que faz |
|---|---|---|
| `src/sistematizacao/carregamento_inicial.py` | 1 e 3 | download/leitura do CSV, limpeza, engenharia de atributos (`tenure` + `faixa_tenure`), split treino/teste e pre-processador |
| `src/sistematizacao/eda.py` | 2 | analise exploratoria; graficos salvos em `relatorios/` |
| `src/sistematizacao/treinamento_avaliacao.py` | 4–7 | baseline, Optuna (Prec_churn), selecao do vencedor, limiar calibrado, avaliacao no hold-out e gravacao do binario |
| `src/sistematizacao/observacao.py` | 8 | alimenta o binario em lotes com limiar de producao e acompanha Prec_churn acumulada |
| `src/sistematizacao/monitoramento.py` | — | drift diario (Prec/AUC/volume), e-mail e retreino so se alarmar |
| `src/sistematizacao/carregar_modelo.py` | — | carrega bundle `{pipe, threshold, modelo}` e faz predicao |
| `src/sistematizacao/gerador_fake.py` | — | gera CSVs sinteticos com Faker, identicos ao original, em `dados/novos/` |
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
uv run python -m sistematizacao.treinamento_avaliacao  # etapas 4-7 -> gera modelos/modelo_churn.joblib
uv run python -m sistematizacao.observacao             # etapa 8
uv run python -m sistematizacao.monitoramento          # drift + e-mail + retreino condicionado
uv run python -m sistematizacao.web                    # API em http://localhost:8000/docs
```

## API

| Metodo | Rota | Descricao |
|---|---|---|
| GET | `/health` | processo de pe + se o binario existe |
| GET | `/modelo` | params, metricas e data do treino |
| POST | `/prever` | analisa um cliente |
| POST | `/prever/lote` | analisa uma lista de clientes |
| POST | `/dados/fake` | gera dados sinteticos e grava em `dados/novos/` |
| GET | `/dados/novos` | lista os CSVs ja gerados |
| POST | `/modelo/recarregar` | recarrega o binario do disco apos novo treino |
| POST | `/modelo/treinar` | re-treina (original + novos) e opcionalmente promove |
| POST | `/modelo/monitorar` | mede drift; e-mail; retreina so se alarmar |
| POST | `/modelo/promover/{run_id}` | promove um run para producao |

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
treino (`faixa_tenure` numerica) antes de chamar o modelo. Sem informar `limiar`
na query, usa o limiar calibrado gravado no treino.

### Postman

`postman/sistematizacao.postman_collection.json` (Collection v2.1) cobre as 5
rotas mais um caso de erro 422, cada request com testes automaticos. Importe
junto `postman/local.postman_environment.json` ou edite a variavel `baseUrl`
da propria colecao.

Da para rodar tudo de uma vez pela linha de comando:

```bash
newman run postman/sistematizacao.postman_collection.json   -e postman/local.postman_environment.json
```

## Dados sinteticos

`POST /dados/fake?linhas=500&semente=7` grava em `dados/novos/` um CSV com as
mesmas 21 colunas do dataset original, na mesma ordem — inclusive as manias
dele: `customerID` no formato `4 digitos-5 letras` e `TotalCharges` em branco
quando `tenure` e 0.

Cada campo sai de um provider do [Faker](https://faker.readthedocs.io):
`fake.random_element([...])` para as categoricas, `fake.random_int` para
`tenure` e `fake.pyfloat(min_value=20, max_value=120)` para `MonthlyCharges`.
As dependencias entre colunas sao reimpostas depois do sorteio (sem telefone ->
`No phone service`, sem internet -> `No internet service`), porque sortear
campo a campo as quebraria.

O rotulo `Churn` sai da probabilidade do proprio modelo, o que deixa o arquivo
coerente para exercitar o pipeline ponta a ponta, mas **nao serve para medir a
qualidade do modelo** — o dado ja nasce concordando com ele.

Como o Faker sorteia cada categoria de forma uniforme, as marginais **nao**
reproduzem as do dataset real (a taxa de churn do gerado fica perto de 0.14
contra 0.27 do original). Para dados de volume/carga isso e irrelevante; para
comparar distribuicoes, nao use o gerado.

`semente` fixa o sorteio: mesma semente, mesmo arquivo. Sem semente, cada
chamada gera um conjunto diferente.

Tambem roda fora da API:

```bash
uv run python -m sistematizacao.gerador_fake
```

Para analisar o arquivo gerado com o pipeline, aponte o `CHURN_CSV` para ele:

```bash
CHURN_CSV=dados/novos/clientes_fake_20260819_200513.csv uv run python -m sistematizacao.observacao
```

Os CSVs gerados nao entram no git.

## Monitoramento (systemd)

O host dispara `scripts/retreinar.sh` via `scripts/retreino.timer` (06:00 diario).
O script chama `POST /modelo/monitorar?retreinar=true`:

1. Avalia o modelo de producao em `dados/novos/` (se houver) ou no hold-out.
2. Alarma se Prec_churn < 0,65, AUC < 0,80 ou volume de contatos longe de ~17%.
3. Envia e-mail (SMTP_HOST + EMAIL_TO) e so entao retreina.
4. Promove o candidato apenas se Acc >= 0,80 e AUC >= 0,82.
5. Sem drift, o modelo do dia anterior permanece.

Sem SMTP configurado o alarme vai para o log do journald (`journalctl -u retreino`).

## Docker

```bash
docker build -t sistematizacao:local .
docker run --rm -p 8000:8000 -v "$PWD/modelos:/app/modelos" sistematizacao:local
```

A imagem embarca o conteudo de `modelos/`; o volume acima e util para trocar o
binario sem rebuildar.

## CI

`.github/workflows/ci.yml` (na raiz do repo) roda ruff, smoke test dos modulos
e da API, e faz **build da imagem sem push** para registry. A imagem que roda
no servidor e construida pelo `.github/workflows/deploy.yml` (self-hosted).
