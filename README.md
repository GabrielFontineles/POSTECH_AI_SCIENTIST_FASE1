# POS-TECH AI SCIENTIST - FASE 1

Este repositório contém o desenvolvimento do Tech Challenge Fase 1 – POS TECH, cujo objetivo é analisar dados de pedidos, entregas e interações de clientes em um e-commerce para compreender os fatores que influenciam a satisfação (NPS) e propor recomendações estratégicas para melhorar a experiência do cliente.

---

## Estrutura do Repositório

```
nps-ecommerce/
│
├── data/
│   ├── raw/                              # Dados originais, NUNCA modificados
│   │   └── desafio_nps_fase_1.csv        # 2.500 registros de NPS
│   └── processed/                        # Gerado pelo notebook 03 (não versionado)
│       └── nps_processed.csv
│
├── notebooks/
│   ├── 01_business_understanding.ipynb   # Entendimento do negócio e da target
│   ├── 02_eda.ipynb                      # Análise Exploratória dos Dados
│   ├── 03_feature_engineering.ipynb      # Pré-processamento e feature engineering
│   └── 04_modelo_preditivo.ipynb         # Modelo de classificação NPS (desafio opcional)
│
├── src/
│   ├── __init__.py
│   └── utils.py                          # Funções auxiliares reutilizáveis
│
├── models/                               # Modelos serializados (gerados pelo notebook 04)
│   ├── nps_classifier.pkl
│   ├── label_encoder.pkl
│   └── feature_names.pkl
│
├── reports/
│   └── figures/                          # Gráficos exportados automaticamente pelos notebooks
│
├── docs/
│   └── 1IAST_Fase1_TechChallenge.pdf     # Enunciado original do desafio
│
├── .gitignore
├── requirements.txt
└── README.md
```

### Por que essa estrutura?

| Pasta | Responsabilidade |
|---|---|
| `data/raw/` | Dados originais imutáveis — **nunca edite arquivos aqui** |
| `data/processed/` | Artefatos gerados pelo pipeline; não versionado (`.gitignore`) |
| `notebooks/` | Análise sequencial e reproduzível, numerada por ordem de execução |
| `src/utils.py` | Código modular importado pelos notebooks — evita duplicação |
| `models/` | Modelos treinados prontos para inferência em produção |
| `reports/figures/` | Gráficos exportados para relatórios e apresentações |

---

## Objetivo do Projeto

Uma empresa de e-commerce enfrenta **alta variabilidade no NPS** entre diferentes perfis de clientes. Mesmo com indicadores operacionais aparentemente similares, alguns clientes se tornam promotores e outros, detratores.

**Pergunta central:**
> Quais fatores operacionais realmente influenciam a satisfação do cliente — e como a empresa pode agir de forma **proativa**, antes mesmo da aplicação da pesquisa de NPS?

---

## Descrição da Base de Dados

**Arquivo:** `data/raw/desafio_nps_fase_1.csv`
**Registros:** 2.500 clientes | **Colunas:** 19

| Coluna | Tipo | Descrição | Leakage? |
|---|---|---|---|
| `customer_id` | int | Identificador único do cliente | — |
| `order_id` | int | Identificador único do pedido | — |
| `customer_age` | int | Idade do cliente | — |
| `customer_region` | str | Região geográfica (5 regiões do Brasil) | — |
| `customer_tenure_months` | int | Tempo de relacionamento com a empresa (meses) | — |
| `order_value` | float | Valor total do pedido (R$) | — |
| `items_quantity` | int | Quantidade de itens no pedido | — |
| `discount_value` | float | Valor do desconto aplicado (R$) | — |
| `payment_installments` | int | Número de parcelas do pagamento | — |
| `delivery_time_days` | int | Tempo total de entrega (dias) | — |
| `delivery_delay_days` | int | Dias de atraso em relação ao prazo prometido | — |
| `freight_value` | float | Valor do frete (R$) | — |
| `delivery_attempts` | int | Número de tentativas de entrega | — |
| `customer_service_contacts` | int | Contatos do cliente com o suporte | — |
| `resolution_time_days` | int | Tempo para resolução de problemas (dias) | — |
| `complaints_count` | int | Número de reclamações registradas | — |
| `repeat_purchase_30d` | int | Recompra em até 30 dias (0/1) | ⚠️ **SIM** |
| `csat_internal_score` | float | Score interno de satisfação | ⚠️ **SIM** |
| `nps_score` | float | **Variável-alvo** — nota NPS de 0 a 10 | — |

> Colunas marcadas com **SIM** são excluídas do modelo por risco de **data leakage** — elas só existem após o evento que queremos prever.

---

## Metodologia

### 1. Entendimento do Negócio
Análise conceitual do problema, importância do NPS e impactos em recompra, boca a boca e market share. Identificação de indicadores complementares (Churn Rate, CAC, LTV).

### 2. Definição da Target e Data Leakage
Identificação da variável-alvo (`nps_score`) e das colunas que causariam contaminação do modelo se incluídas no treinamento.

### 3. Análise Exploratória (EDA)
- Matriz de correlação com `nps_score`
- Ponto de ruptura: queda de **~39% no NPS** entre o 4º e 5º dia de atraso
- NPS por região geográfica
- Perfil comparativo: Detratores vs Promotores

**Principais achados:**

| Fator | Correlação com NPS | Insight |
|---|---|---|
| `delivery_delay_days` | **−0.60** | Maior driver de insatisfação |
| `complaints_count` | **−0.50** | Reclamações destroem o NPS |
| `customer_service_contacts` | **−0.35** | Múltiplos contatos frustram |

### 4. Feature Engineering
Criação de novas variáveis: `discount_ratio`, `has_delay`, `high_contact`, `freight_per_item`.

### 5. Modelo Preditivo *(desafio opcional)*
Classificação multiclasse (Detrator / Neutro / Promotor) com **Random Forest** e **XGBoost**. Avaliação via F1-Score Macro e validação cruzada estratificada (5 folds).

---

## Como Reproduzir os Resultados

### Pré-requisitos
- Python 3.10+
- Git

### Passo a passo

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/nps-ecommerce.git
cd nps-ecommerce

# 2. Crie um ambiente virtual
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Inicie o JupyterLab
jupyter lab
```

### Ordem de execução dos notebooks

Execute **nessa ordem** — cada notebook depende do anterior:

| # | Notebook | Saída gerada |
|---|---|---|
| 01 | `01_business_understanding.ipynb` | Análise conceitual |
| 02 | `02_eda.ipynb` | Gráficos em `reports/figures/` |
| 03 | `03_feature_engineering.ipynb` | `data/processed/nps_processed.csv` |
| 04 | `04_modelo_preditivo.ipynb` | Modelos em `models/` |

### Execução via linha de comando (sem interface)

```bash
jupyter nbconvert --to notebook --execute notebooks/01_business_understanding.ipynb --output notebooks/01_business_understanding_executed.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_eda.ipynb --output notebooks/02_eda_executed.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_feature_engineering.ipynb --output notebooks/03_feature_engineering_executed.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_modelo_preditivo.ipynb --output notebooks/04_modelo_preditivo_executed.ipynb
```

---

## Boas Práticas Aplicadas

| Prática | Como foi implementada |
|---|---|
| **Código organizado** | Notebooks numerados por etapa + módulo `src/utils.py` |
| **Nomes claros** | Snake_case em todas as variáveis; sem abreviações ambíguas |
| **Comentários** | Cada bloco de código tem comentário explicativo |
| **Estrutura de pastas** | `data/`, `notebooks/`, `models/`, `reports/`, `src/` |
| **README completo** | Objetivo, dados, metodologia e instruções de reprodução |
| **Controle de leakage** | Função `remove_leakage_cols()` centraliza e documenta exclusões |
| **Reprodutibilidade** | `random_state=42` em todos os modelos e splits |
| **Versionamento** | `.gitignore` configurado para não versionar dados e artefatos |

---
Tech Challenge — Fase 1 | PosTech | Maio 2026
