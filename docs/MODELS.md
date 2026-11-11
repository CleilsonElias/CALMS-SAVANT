# Integração opcional com Transformers e PEFT

O núcleo NumPy foi executado. A integração deste documento foi escrita e teve
sua sintaxe conferida; PyTorch, Transformers, PEFT e modelos reais não estavam
disponíveis no ambiente desta entrega. O teste opcional roda automaticamente
quando PyTorch está instalado, mas não substitui validação em um transformer.

## Executar localmente

Instale as dependências opcionais e tenha um modelo causal compatível já salvo
em um diretório local. O código não baixa pesos nem executa código remoto.

```bash
python -m pip install -e ".[transformers]"
python examples/train_transformer.py --model /caminho/modelo_local --data examples/train.jsonl --anchors examples/anchors.jsonl --identity examples/identity.jsonl --inhibition examples/inhibition.json --output transformer_run --max-steps 8 --max-length 128 --rank-budget 8 --target-modules q_proj,v_proj
```

No Windows, substitua o caminho por um diretório existente. O comando deve ser
executado na raiz do repositório. `transformer_run` precisa ser novo. Os nomes
`q_proj,v_proj` dependem da arquitetura; ajuste `--target-modules` às camadas
reais do seu modelo. `--device cuda` ativa o dispositivo somente se seu PyTorch
e hardware estiverem configurados.

O exemplo tem frases curadas mínimas para mostrar o formato, sem demonstrar
cobertura semântica, resistência adversarial ou adequação de uma política real.

| Entrada | Campos obrigatórios | Regra |
|---|---|---|
| Treino JSONL | `domain`, `text` | Domínio explícito, pelo menos dois tokens por texto |
| Âncoras JSONL | `id`, `text` | IDs únicos e conjunto fixo |
| Identidade JSONL | `id`, `text`; `domain` opcional | Fonte curada e independente das amostras rejeitadas |
| Inibição JSON | domínio -> número positivo | Deve cobrir exatamente os domínios de treino |

## O que o treinador faz

Cria um adaptador LoRA por domínio, com ranks inteiros sob orçamento. Ativa um
por vez no treino e extrai representações por média da última camada oculta,
excluindo padding. Todas as rotas de domínio são executadas para medir âncoras.
Pooling é uma escolha operacional, não uma distribuição completa de ativações.

Cada passo obtém gradientes reais de autograd, limita sua norma, guarda os
adaptadores em CPU, aplica uma proposta, mede a geometria e restaura o estado.
Em seguida aplica a decisão e valida o passo final. Usa SGD direto, sem estado
de Adam, momentum ou weight decay que pudesse deslocar pesos durante rejeição.

Dropout fica desativado no passo supervisionado e nas medições para permitir
comparações determinísticas. O exemplo não usa AMP, quantização, paralelismo,
gradient accumulation ou treinamento distribuído. Não é uma implementação de
QLoRA. A base e os biases precisam permanecer congelados.

O prior é extraído de frases curadas no mesmo modelo inicial, o que **não**
equivale a inicialização confiável independente nem consenso multi-semente.
O módulo de consenso é separado e exige candidatas preparadas no mesmo espaço.

## Saídas e limitações

`adapters/` contém os adaptadores PEFT finais ou restaurados, serializados com
safetensors. `tokenizer/` contém o tokenizer usado. `training_log.json` informa
decisões, configuração, ranks e congelamento. Não há avaliação PSR/FRS, juízes
externos ou conclusão de segurança. Os snapshots intermediários são voláteis;
essa integração não promete retomada exata de treinamento após encerrar o processo.

A auditoria pode restaurar e congelar. O script para nesse ponto e salva o estado
restaurado; a classe admite retomada explícita com nota de revisão na sessão.
Uma falha numérica também congela e reverte os adaptadores. O treinamento não
deve compartilhar a mesma instância de modelo com atendimento concorrente.

O valor `task_weight=1` preserva a inclusão da loss no critic do artigo. Para
cross-entropy alta, exp(-kappa*loss) pode praticamente paralisar o aprendizado
benigno. `--task-weight 0` permite investigar uma variante que usa apenas os
termos estruturais na atenuação; registre essa diferença e calibre em validação
separada. Não ajuste limiares no conjunto de teste para obter um resultado desejado.

Comece com um modelo pequeno e sequências curtas. O custo inclui diversas
execuções de âncoras por passo, snapshots e ativações de treino; não depende
apenas do número de parâmetros treináveis. Não foi medido consumo de VRAM ou
throughput nesta entrega. Não se promete que um backbone 8B caiba em uma GPU
específica usando este script em precisão padrão.

## Relação com LLI-CAMS

LLI-CAMS organiza memória externa e o ciclo offline; CALMS-SAVANT governa
propostas de alteração paramétrica. Esses papéis são complementares. Esta versão
é um repositório independente: não altera o código publicado de LLI-CAMS e não
se anuncia como um hook já validado naquele projeto.

Uma integração futura precisa transportar evidência e IDs do replay até a
proposta, usar a mesma versão de encoder, registrar aceite/rejeição e publicar
atomicamente o adaptador aprovado. Descartar uma memória externa não desfaz
automaticamente sua influência em pesos já treinados.
