# CALMS-SAVANT Reference Implementation

**Versão 0.1.0 - implementação executável de referência para pesquisa.**

Governança de consolidação paramétrica: um grafo de identidade explícito, transporte
ótimo regularizado entre âncoras correspondentes, atenuação e rejeição de propostas,
adaptação LoRA por domínio, arquivo limitado de assinaturas e auditoria histórica.
O núcleo executa treinamento real de matrizes de baixo posto com NumPy em CPU.

Arquitetura de referência: **Cleilson Elias de Sousa**. Código integral sob licença
MIT. Leia a [análise técnica](docs/ANALISE_TECNICA.md), o
[formalismo implementado](docs/FORMALISM.md) e a [validação](docs/VALIDATION.md).

**Os valores PSR = 4,8% e FRS = 92,3% relatados no artigo não foram reproduzidos
por este repositório.** Os experimentos entregues são modelos lineares sintéticos,
identificados como proxies. Não há demonstração de segurança de um LLM real.

## Executar

Requer Python >= 3.11 e NumPy. Não exige GPU, modelo baixado nem chave de API.

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Instale e execute:

```bash
python -m pip install -e .
calms-savant demo --directory demo_run
calms-savant inspect demo_run/state.json
python -m unittest discover -s tests -v
```

Com NumPy já disponível, também é possível usar o código sem instalação:

```bash
python run_calms_savant.py demo --directory demo_run
```

O comando `demo` exige um diretório novo e grava `events.json`, `summary.json`
e `state.json`. Cada decisão contém critic, atenuação, rejeições e cadeia de hashes.
As propostas rejeitadas deixam os parâmetros intactos. A base permanece congelada.

## O que foi implementado

| Componente | Implementação e limite |
|---|---|
| Identity Graph | Direções curadas, centralidade, frequência com decaimento e pesos sigmoidais; não determina verdade |
| Geometric Stability Filter | Sinkhorn em log, convergência verificável, distância por ID de âncora; padrão usa divergência com correção de viés |
| Cognitive Diode | Atenuação exponencial e portões explícitos; a exponencial sozinha não zera atualizações |
| SAVANT | Bancos LoRA realmente treináveis, ranks inteiros sob orçamento e atualização por domínio |
| TCS | Assinaturas normalizadas, deduplicação, descarte temporal e capacidade limitada; busca exata padrão, HNSW opcional |
| Replay defensivo | Loss contrastiva com alvos seguros e negativos explicitamente fornecidos; nenhuma assinatura vira alvo seguro automaticamente |
| Bootstrap | Heurística medoid/MAD e variante média/desvio do artigo; nenhuma é certificada contra envenenamento |
| Auditoria | Comparação contra época 0, reavaliação de checkpoints, restauração e congelamento até revisão explícita |
| Persistência | JSON com checksum, escrita por substituição atômica, validação de dimensões e continuação testada |
| Transformers/PEFT | Treinador opcional com SGD direto e propostas reversíveis; não executado nesta entrega |

As alterações necessárias ao formalismo original estão mapeadas em
[docs/PAPER_TO_CODE.md](docs/PAPER_TO_CODE.md). O arquivo vetorial é um registro
de padrões rejeitados, sem promessa de compressão topológica que preserve conteúdo.

## Benchmark e falhas demonstráveis

```bash
calms-savant benchmark --seeds 5 --epochs 100 --output benchmark.json
calms-savant nullspace --output nullspace.json
calms-savant consensus examples/seeds.json --method paper_mean_std --output paper_consensus.json
calms-savant consensus examples/seeds.json --method medoid_mad --output robust_consensus.json
python examples/defensive_replay.py
```

O benchmark usa 20% de ataques e a mesma sequência para sete variantes: replay
sem governança, atenuação isolada, sistema completo, sem identidade, sem TCS,
sem auditoria e ranks uniformes. São 5 sementes x 100 épocas x 5 registros por
época por variante. Os rótulos de ataque ficam restritos ao avaliador.

Os resultados desta entrega estão em `validation/benchmark_linear.json`. A versão
completa preservou os quatro probes protegidos nesse cenário; o replay sem
governança falhou em um dos quatro. Isso **não corresponde a PSR em linguagem
natural**. A remoção de identidade ou auditoria também preservou esses probes:
o cenário não identifica uma contribuição causal dessas partes.

O exemplo `nullspace` inverte a saída de um input não observado sem alterar
qualquer âncora monitorada. A versão média/desvio do consenso mantém uma semente
oposta entre três candidatas, apesar de duas serem idênticas. Esses resultados
desfavoráveis permanecem no repositório como testes de limitação.

## Usar a API Python

```python
import numpy as np
from calms_savant import Consolidator, IdentityGraph, IdentityNode, LinearLoRA

eye = np.eye(4)
model = LinearLoRA(eye, {"facts": 2}, seed=0)
prior = IdentityGraph([IdentityNode("approved-reference", eye[0])])
engine = Consolidator(model, prior, {"stable-prompt": eye[:1]})
report = engine.consolidate(
    inputs=eye[1:2],
    targets=eye[1:2] + 0.1*eye[3:4],
    representations=eye[1:2],
    domain="facts",
)
print(report["status"])
engine.save("research_state.json")
restored = Consolidator.load("research_state.json")
```

`representations` deve corresponder ao espaço do prior; sua qualidade semântica
é uma hipótese a medir. O domínio e a aplicabilidade dos nós são fornecidos pelo
chamador. O pacote não descobre domínios nem extrai invariantes confiáveis sozinho.

Após uma auditoria que restaura e congela o estado, a retomada exige uma nota:

```bash
calms-savant resume research_state.json --review-note "Inspeção concluída" --output reviewed_state.json
```

A nota documenta uma decisão humana; não comprova a correção da decisão.

## Treinar um transformer local

Instalação opcional:

```bash
python -m pip install -e ".[transformers]"
python examples/train_transformer.py --help
```

O treinador exige um diretório de modelo existente e usa `local_files_only=True`.
Veja [docs/MODELS.md](docs/MODELS.md) para dados, pooling, domínios, custo e limites.
Gradientes e pesos de transformer não foram executados no ambiente desta entrega.

## Publicar integralmente no GitHub

O nome sugerido é `CALMS-SAVANT`, separado de `LLI-CAMS`. O pacote contém código,
testes, licença MIT, `CITATION.cff`, documentação, exemplos, CI e resultados brutos.
Veja [docs/PUBLISHING.md](docs/PUBLISHING.md). O código não publica automaticamente.

O PDF original, conversas pessoais, pesos de terceiros e bancos de pesquisa não
fazem parte do pacote. `validation/source_document.json` registra somente os
metadados e o hash do documento analisado. Um commit de código de novembro de
2026 não comprova anterioridade em maio nem valida resultados experimentais.
