# Data Pipeline Dashboard

Pipeline de limpeza e análise de dados com dashboard web interativo. Lê um CSV bruto, aplica limpeza automática (duplicatas, nulos, tipos inválidos), calcula estatísticas e exibe tudo num dashboard com tabela filtrável e detecção de outliers.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)
![NumPy](https://img.shields.io/badge/NumPy-2.x-013243?logo=numpy)

---

## O que faz

- Lê qualquer CSV (separador `,` ou `;`, detecta automaticamente)
- Remove linhas duplicadas e completamente vazias
- Converte colunas para o tipo certo: numérico, datetime, string
- Suporta valores monetários brasileiros (`R$ 1.234,56`)
- Detecta outliers pelo método IQR de Tukey
- Salva o CSV limpo em `output/dados_limpos.csv`
- Exibe um dashboard web com estatísticas, tipos de dados e tabela interativa

## Stack

| Ferramenta | Uso |
|---|---|
| **pandas** | Carregamento, limpeza e transformação dos dados |
| **NumPy** | Cálculos estatísticos (percentis, desvio padrão, IQR) |
| **Flask** | Servidor web do dashboard |
| **Bootstrap 5** | Layout e componentes visuais |

## Como rodar

### 1. Instalar dependências

```bash
pip install pandas numpy flask
```

### 2. Subir o dashboard

```bash
python app.py
```

Acesse **http://localhost:5000** no navegador.

### 3. Usar o pipeline via linha de comando

```bash
# Com o CSV de exemplo incluído
python data_analyzer.py

# Com seu próprio arquivo
python data_analyzer.py seu_arquivo.csv output/resultado.csv
```

## Estrutura

```
.
├── data_analyzer.py      # Pipeline principal (load → clean → analyze → save)
├── app.py                # Servidor Flask do dashboard
├── run_tests.py          # Testes de validação
├── sample_data.csv       # CSV de exemplo com dados sujos
├── templates/
│   └── index.html        # Dashboard web
└── output/
    └── dados_limpos.csv  # Gerado após a primeira execução
```

## Dashboard

O dashboard mostra:

- **Cards de limpeza** — quantas linhas foram removidas e por qual motivo
- **Tipos de dados** — tipo final de cada coluna e cobertura (% de preenchimento)
- **Estatísticas** — média, mediana, desvio padrão, Q1/Q3 e contagem de outliers por coluna numérica
- **Tabela de dados** — todos os registros limpos, com linhas outliers destacadas em vermelho, busca em tempo real e ordenação por coluna

## Testes

```bash
python run_tests.py
```

19 testes cobrindo: remoção de duplicatas e linhas vazias, conversão de tipos, precisão das estatísticas, detecção de outliers e leitura/escrita de arquivos.

## API

O endpoint `/api/data` retorna o resultado completo do pipeline em JSON — útil para integrar com outros sistemas.

```bash
curl http://localhost:5000/api/data
```

## Formato do CSV de entrada

O pipeline aceita qualquer CSV com cabeçalho. Colunas com valores mistos (ex: `R$ 1.200` junto com `1500`) são normalizadas automaticamente. Valores não conversíveis (ex: `abc` numa coluna numérica) viram `null`.
