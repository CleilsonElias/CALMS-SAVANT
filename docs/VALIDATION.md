# Validação da versão 0.1.0

O registro de execução é `validation/validation.json`. Os resultados são desta
implementação, sem reconstrução de logs do artigo de maio.

## Escopo executado

Python 3.12.14, NumPy, CPU. Instalação editável e comandos CLI testados. A suíte
descobriu 42 testes: 40 passaram, dois foram omitidos porque `torch` e
`hnswlib` não estavam instalados. A comparação independente contra OT exato usou
`scipy.optimize.linprog`, disponível neste ambiente e opcional nos testes.
O registro final preserva os resultados e razões das omissões.

Foram verificados gradientes por diferenças finitas, marginais do transporte,
autodivergência, preservação de IDs de âncoras, oposição por amostra, orçamento de
ranks, clipping, imutabilidade em rejeições, restauração histórica, persistência,
continuação exata após reabertura, checksum e limites do arquivo/replay.

## Artefatos de experimento

| Arquivo | Conteúdo |
|---|---|
| `benchmark_linear.json` | Cinco sementes, cem épocas, sete variantes, configurações, curvas e métricas proxy |
| `benchmark_summary.csv` | Médias, desvios amostrais e SEM extraídos do JSON |
| `benchmark_linear.png` | Resumo visual do surrogate |
| `nullspace.json` | Inversão de comportamento fora das entradas observadas pelas âncoras |
| `consensus_counterexample.json` | Regra média/desvio mantém uma semente oposta; medoid/MAD a exclui nesse caso |
| `source_document.json` | Identificação e hash do PDF de origem; sem redistribuir o documento |
| `tests.log` | Saída da suíte realmente executada |
| `validation.json` | Ambiente, comandos, resultados e omissões |

## Interpretação

A taxa de falha de probes é uma fração entre quatro entradas protegidas de um
modelo linear. Não mede PSR do PPP-Bench, FRS em linguagem natural, engano,
sycophancy ou segurança de agentes. Todos os valores de treino/avaliação são
sintéticos e não representam dados de usuários.

No cenário entregue, a governança completa melhorou essa métrica em relação ao
replay sem governança. Sem identidade e sem auditoria também preservaram os
probes. O cenário não discrimina contribuição causal dessas partes; ele não
reproduz as ablações do artigo. As médias factuais ficaram próximas entre
variantes; não há demonstração de superioridade geral.

O contraexemplo de null-space conserva deriva zero e ainda inverte uma saída
não observada. O contraexemplo de consenso demonstra que a condição de minoria
e a regra do PDF não bastam para a conclusão de poda apresentada.

## O que não foi executado

Treinamento de LLaMA-3-8B ou outro transformer, avaliação com GPT-4o/Claude,
PPP-Bench original, HNSW, experimento RHAVENA, inferência concorrente, teste de
GPU, latência de produção e CI remoto em outras versões de Python. A configuração
CI para Python 3.11-3.13 acompanha o código; sua presença não significa que já
tenha rodado no GitHub.

## Reproduzir

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
calms-savant benchmark --seeds 5 --epochs 100 --output validation/benchmark_linear.json
calms-savant nullspace --output validation/nullspace.json
python scripts/summarize_results.py
```

SciPy habilita o oráculo independente de OT e matplotlib habilita a figura.
São ferramentas de verificação/relato, sem dependência de runtime do núcleo.
HNSW e Torch têm testes opcionais que passam a executar quando instalados.
