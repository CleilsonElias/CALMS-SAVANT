# Publicação integral

Nome sugerido: `CleilsonElias/CALMS-SAVANT`. O pacote está pronto para criar um
repositório independente; ele não modifica `CleilsonElias/LLI-CAMS`.

## Pelo navegador

Na sua conta GitHub, crie um repositório vazio chamado `CALMS-SAVANT`, com a
visibilidade que desejar. Não peça ao GitHub para gerar README ou licença:
ambos já existem. Extraia o ZIP e envie **o conteúdo de CALMS-SAVANT**, incluindo
`.github/`, `.gitignore` e os demais arquivos da raiz. O ZIP não é o código
navegável de um repositório até seus arquivos serem extraídos.

## Com Git e GitHub CLI

Na raiz extraída, com autenticação GitHub já configurada:

```bash
git init -b main
git add .
git commit -m "Release CALMS-SAVANT reference implementation v0.1.0"
gh repo create CleilsonElias/CALMS-SAVANT --public --source . --remote origin --push
```

Esses comandos criam um repositório público e enviam o código. Se quiser mantê-lo
privado, troque `--public` por `--private`. Não execute `gh repo create` se já
existir um repositório com esse nome. Para um repositório **novo e vazio** criado
no navegador, substitua o último comando por:

```bash
git remote add origin https://github.com/CleilsonElias/CALMS-SAVANT.git
git push -u origin main
```

Depois do envio, confira a execução da CI no GitHub. Esta entrega só registra
os testes locais; o resultado remoto ainda não existe.

Para marcar uma versão após conferir o commit:

```bash
git tag -a v0.1.0 -m "CALMS-SAVANT reference v0.1.0"
git push origin v0.1.0
```

## Descrição sugerida

Research reference implementation of CALMS-SAVANT: identity-guided consolidation,
Sinkhorn drift checks, domain-stratified LoRA, bounded rejection signatures,
historical audits, and reproducible synthetic benchmarks.

## Atribuição e alegações

`CITATION.cff` identifica autor, versão e data da implementação. Não inclui DOI
inventado nem data retroativa de publicação do código. O artigo de referência
mantém sua própria data declarada. Novos arquivos de software não comprovam
anterioridade, novidade jurídica ou reprodução de experimentos passados.

A licença MIT cobre o código novo. Nenhum peso, dataset de terceiros, conversa
pessoal ou PDF original acompanha o pacote. Os dados de exemplo são fictícios.
Não apresente `validation/benchmark_linear.json` como PPP-Bench original ou
como confirmação das porcentagens do artigo.
