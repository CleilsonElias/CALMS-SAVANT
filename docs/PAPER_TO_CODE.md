# Rastreabilidade: artigo para software

| Referência no PDF | Módulo | Decisão nesta versão |
|---|---|---|
| Eq. (1), p. 2-3 | `identity.py` | Centralidade não negativa, atualização discreta de frequência confiável, sigmoid |
| Eq. (2), p. 3 | `transport.py`, `engine.py` | Âncoras de mesmo ID, modelo antes/depois, controle de convergência |
| Eqs. (E.3)-(E.4), p. 15 | `transport.py` | Custos normalizados; divergência debiased padrão; batch não substitui âncoras |
| Eq. (3), p. 4; Eq. (E.5), p. 16 | `identity.py` | Avaliação por amostra com aplicabilidade explícita; não usa argmax de lote |
| Eq. (4), p. 4 | `policy.py` | Pesos configuráveis, incluindo extensão opcional de recorrência |
| Eqs. (5), (E.11), p. 4 e 18 | `lora.py` | Ranks inteiros sob orçamento; sem alocação semântica automática |
| Eq. (6), p. 4 | `lora.py`, `engine.py` | Descida nos fatores, clipping e proposta transacional |
| Seção 3.5, p. 4 | `archive.py` | Arquivo limitado, deduplicação; busca exata e HNSW opcional |
| Seção 3.6, p. 4-6 | `consensus.py` | Duas heurísticas e contraexemplo executável; sem garantia bizantina |
| Eqs. (15)-(16), p. 7 | `engine.py` | Raiz inicial preservada, checkpoints reavaliados, restauração e congelamento |
| Seções 4-5, p. 7-9 | `benchmark.py` | Surrogate linear separado; PPP-Bench original não reproduzido |
| Eq. (E.1), p. 15 | `replay.py` | Limite estrito em amostras rejeitadas; com reposição e seed explícita |
| Eq. (E.14), p. 18 | Decisão documentada | Não mescla adaptadores na base; protege reversibilidade |
| Seção 9, p. 11 | `benchmark.nullspace_demo` | Contraexemplo de comportamento fora das âncoras; Affective Diode não implementado |

A implementação não incorpora RHAVENA, detector de consciência, extração de
vetores afetivos ou testes de engano. Esses módulos não foram formalizados ou
validados pelo PDF de origem para integração automática nesta versão.

Novas escolhas de engenharia: divergência Sinkhorn como padrão, custo euclidiano
normalizado, portões finitos, clipping explícito, limites inteiros de rank,
preservação da raiz de recuperação, checksum e penalidade opcional de recorrência.
Não devem ser atribuídas retrospectivamente a experimentos de maio de 2026.
