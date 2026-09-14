# Intent Classifier

Classificador híbrido de intenções para consultas em linguagem natural, desenvolvido como parte de um pipeline experimental para um futuro assistente de IA integrado ao SUAP/IFMA.

O projeto recebe uma pergunta do usuário, calcula similaridade semântica e correspondência lexical com um conjunto de intenções conhecidas, combina os resultados em um score híbrido e retorna as intenções mais prováveis.

O componente **não utiliza LLM em runtime, não realiza RAG e não gera a resposta final ao usuário**. Sua responsabilidade é exclusivamente **identificar a intenção da consulta** e fornecer candidatos ranqueados para as etapas seguintes do pipeline.

## Objetivo

O objetivo deste projeto é criar uma primeira camada de compreensão de consultas antes da etapa de geração/reformulação da pergunta.

Exemplo:

```text
"Quero trancar minha matrícula esse semestre"   ->  [pergunta]
                    │
             Intent Classifier
                    │
            trancamento_curso                   ->  [intenção]
```

Essa informação poderá posteriormente ser utilizada por um componente de `query-rewriting`, que transformará a pergunta original em uma representação mais padronizada para as etapas de templates, RAG e geração de respostas.

## Como funciona

O classificador utiliza duas fontes de informação:

1. **Similaridade semântica** — compara a consulta com os exemplos cadastrados para cada intenção utilizando embeddings.
2. **Correspondência lexical** — verifica a sobreposição entre os termos da consulta e as keywords de cada intenção.

Os dois valores são combinados em um score híbrido:

```text
score = semantic_score × 0.75 + lexical_score × 0.25
```

O candidato com maior score é selecionado.

Existe também um mecanismo simples de rejeição (`unknown`):

```text
se score < 0.40:
    predicted_intent = "unknown"
```

O threshold é configurável.

### Fluxo

```text
Consulta do usuário
        │
        ▼
Normalização do texto
        │
        ├───────────────┐
        ▼               ▼
 Similaridade       Match lexical
 semântica              │
        │               │
        └───────┬───────┘
                ▼
        Score híbrido
                │
                ▼
       Ranking de intenções
                │
          ┌─────┴─────┐
          ▼           ▼
      score ≥ 0.40  score < 0.40
          │           │
          ▼           ▼
      intenção     unknown
```

## Modelo semântico

O projeto utiliza:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Os embeddings são gerados utilizando [FastEmbed](https://github.com/qdrant/fastembed).

A similaridade utilizada para comparar os vetores é baseada em similaridade de cosseno.

Para cada intenção, o sistema compara a consulta com seus exemplos e utiliza a maior similaridade encontrada como `semantic_score`.

## Estrutura do projeto

```text
intent-classifier/
├── data/
│   ├── example.batch_test.json
│   ├── example.intents.json
│
├── logs/
│   ├── batch_metrics.json
│   ├── errors_batch_metrics.json
│   └── history.jsonl
│
├── src/
│   └── intent_classifier/
│       ├── __init__.py
│       ├── classifier.py
│       ├── cli.py
│       ├── lexical.py
│       ├── loader.py
│       ├── logger.py
│       ├── models.py
│       ├── semantic.py
│       └── text.py
│
├── tests/
│   ├── test_classifier.py
│   ├── test_lexical.py
│   ├── test_semantic.py
│   └── test_text.py
│
├── pyproject.toml
├── README.md
└── uv.lock
```

## Componentes

### `classifier.py`

Responsável pela lógica principal de classificação.

Combina os scores semântico e lexical, ordena os candidatos e determina a intenção final ou `unknown`.

### `semantic.py`

Responsável pela geração dos embeddings e cálculo da similaridade semântica.

### `lexical.py`

Responsável pela correspondência entre os termos da consulta e as keywords cadastradas nas intenções.

### `text.py`

Contém as operações de normalização utilizadas antes da classificação.

### `models.py`

Define as estruturas de dados utilizadas pelo sistema, como intenções, candidatos e resultados de classificação.

### `loader.py`

Responsável pelo carregamento e validação dos arquivos de intenções e dos casos de teste.

### `logger.py`

Responsável pelos registros e relatórios gerados durante as classificações.

### `cli.py`

Interface de linha de comando para execução dos modos interativo e batch.

## Dados

### `data/intents.json`

Contém as intenções conhecidas pelo classificador.

Cada intenção possui, de forma geral:

```json
{
  "id": "trancamento_curso",
  "description": "Solicitação de trancamento de matrícula ou pausa nos estudos",
  "keywords": [
    "trancar",
    "trancamento",
    "semestre"
  ],
  "examples": [
    "Preciso trancar minha matrícula este semestre",
    "Como faço o trancamento do curso?"
  ]
}
```

O conjunto de exemplos é utilizado pelo componente semântico, enquanto as keywords fornecem o reforço lexical.

### `data/batch_test.json`

Contém casos independentes utilizados para avaliar o classificador.

Cada caso possui:

```json
{
  "query": "Quero trancar minha matrícula",
  "expected_intent": "trancamento_curso"
}
```

O dataset de teste é mantido separado de `intents.json` para evitar que os exemplos utilizados no treinamento/configuração das intenções sejam simplesmente reutilizados como benchmark.

### Arquivos `example.*`

Os arquivos:

```text
example.intents.json
example.batch_test.json
```

servem como exemplos/documentação da estrutura esperada dos datasets.

## Modo interativo

Para executar o classificador de forma interativa:

```bash
uv run intent-classifier --mode interactive
```

O sistema permite inserir consultas diretamente no terminal e visualizar a intenção prevista, score e candidatos.

## Modo batch

Para executar toda a bateria de testes:

```bash
uv run intent-classifier --mode batch
```

O modo batch executa todas as consultas presentes em:

```text
data/batch_test.json
```

e exibe uma tabela semelhante a:

```text
STATUS | PERGUNTA          | ESPERADO   | PREVISTO   | SCORE
-------|-------------------|------------|------------|-------
OK     | [pergunta]        | [intenção] | [intenção] | 0.0000
ERRO   | [pergunta]        | [intenção] | [intenção] | 0.0000
```

Ao final, o sistema apresenta a quantidade de acertos (%) e accuracy.

## Relatórios

O modo batch gera três arquivos principais em `logs/`.

### `batch_metrics.json`

Relatório consolidado contendo os resultados de todas as consultas executadas.

Inclui informações como:

* consulta;
* intenção esperada;
* intenção prevista;
* score;
* candidatos;
* `semantic_score`;
* `lexical_score`;
* configuração utilizada;
* resultado da classificação.

### `errors_batch_metrics.json`

Contém somente os casos classificados incorretamente.

Isso facilita a análise dos erros sem precisar filtrar manualmente todo o benchmark.

### `history.jsonl`

Histórico incremental das classificações.

O formato JSONL permite armazenar cada evento como um objeto JSON independente em uma linha.

## Testes

Os testes automatizados podem ser executados com:

```bash
uv run pytest
```

A suíte cobre os principais componentes do sistema:

```text
tests/
├── test_text.py
├── test_semantic.py
├── test_lexical.py
└── test_classifier.py
```

## Benchmark atual

O projeto possui uma bateria com:

```text
192 consultas
```

Após a primeira rodada de implementação, o resultado inicial foi:

```text
154/192 acertos
80.21%
```

Depois de uma revisão cirúrgica das definições, keywords e exemplos de algumas intenções:

```text
171/192 acertos
89.06%
```

Resultado atual:

```text
Accuracy: 89.06%
Acertos:   171
Erros:      21
Total:      192
```

A melhoria foi obtida sem alterar:

* o modelo de embeddings;
* a fórmula do score;
* os pesos semântico/lexical;
* o threshold de `unknown`;
* a arquitetura do classificador.

A melhoria veio principalmente do refinamento das próprias definições das intenções, keywords e exemplos.

## Limitações conhecidas

O classificador não pretende atingir 100% de acerto.

Algumas intenções são semanticamente próximas e podem ser naturalmente difíceis de separar apenas utilizando similaridade de embeddings e correspondência lexical.

Também existem falsos positivos de `unknown`, nos quais uma pergunta fora do conjunto conhecido apresenta similaridade suficiente com alguma intenção existente para ultrapassar o threshold.

Essas limitações são esperadas para uma primeira camada de classificação baseada em embeddings + regras lexicais.

## O que este projeto não faz

Este projeto **não**:

* gera respostas para o usuário;
* utiliza um LLM em runtime;
* realiza RAG;
* consulta documentos do IFMA;
* executa chamadas à API do SUAP;
* realiza geração de texto;
* reformula perguntas;
* extrai entidades complexas.

Sua responsabilidade termina na identificação e ranqueamento da intenção.

## Próximas etapas do pipeline

O `intent-classifier` é apenas a primeira etapa de um pipeline maior.

A arquitetura planejada é aproximadamente:

```text
Pergunta original
       │
       ▼
┌─────────────────────┐
│  Intent Classifier  │
└─────────────────────┘
       │
       ▼
Intent + candidatos
       │
       ▼
┌─────────────────────┐
│   Query Rewriter    │
└─────────────────────┘
       │
       ▼
Entidades + pergunta padronizada
       │
       ▼
Templates / enriquecimento
       │
       ▼
┌─────────────────────┐
│        RAG          │
└─────────────────────┘
       │
       ▼
Contexto relevante
       │
       ▼
┌─────────────────────┐
│        LLM          │
└─────────────────────┘
       │
       ▼
Resposta ao usuário
```

O próximo componente planejado é o **Query Rewriter**, responsável por utilizar a intenção identificada para transformar consultas informais em perguntas mais estruturadas e adequadas às etapas posteriores do pipeline.

## Status

**Concluído como primeira versão funcional.**

O classificador atingiu `89.06%` de accuracy na bateria atual e possui:

* classificação semântica;
* reforço lexical;
* ranking de candidatos;
* mecanismo de `unknown`;
* testes automatizados;
* benchmark independente;
* relatórios de execução;
* relatório específico de erros;
* CLI para execução interativa e batch.

Neste ponto, o foco do projeto passa para as próximas etapas do pipeline, em vez de buscar uma otimização indefinida do classificador.

---

## License

Este projeto é experimental e desenvolvido para fins de estudo, pesquisa e prototipação.
