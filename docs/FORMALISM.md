# Formalismo operacional

Esta versão distingue expressamente equações originais, escolhas numéricas e
extensões de engenharia. A referência é o PDF de 19 páginas, datado no próprio
texto de 18 de maio de 2026; a análise e o software são de 11 de novembro de 2026.

## Direções, identidade e aplicabilidade

Para vetores não nulos, z = h / ||h||. Vetores nulos, dimensões incompatíveis e
valores não finitos são rejeitados. Cada nó curado possui ID, direção unitária,
domínio, frequência confiável f e reforço r.

\[
\rho_j=\operatorname{sigmoid}(\alpha c_j+\beta f_j+\gamma r_j),
\quad f_j(t+\Delta t)=e^{-\lambda\Delta t}f_j(t)+a_j.
\]

A frequência usa ativações discretas por evento, não uma integral contínua
resolvida exatamente. Centralidade é obtida por iteração de potência em A+I;
grafo desconectado pode ter centralidade não única. O reforço é fornecido, não
inferido automaticamente. Nenhum episódio rejeitado fortalece o prior.

\[
\Phi_{immune}=\frac1B\sum_b\sum_{j\in J(b)}
\rho_j\max(0,-z_b^T v_j).
\]

J(b) contém os nós aplicáveis ao domínio e à lista explícita de IDs. Cada amostra
é avaliada individualmente; selecionar o item de maior cosseno de um lote pode
ocultar uma contradição presente em outro item. Não se presume que oposição
vetorial seja equivalente a contradição lógica.

## Transporte regularizado

Cada ID de âncora é executado antes e depois da proposta pelo modelo completo,
incluindo adaptadores ativos. Com o extrator inteiramente congelado e independente
dos parâmetros atualizados, esta medição seria incapaz de observar a atualização.

O custo padrão é c(x,y)=||x-y||/2 para vetores unitários, portanto está em [0,1].
O custo opcional (1-x^T y)/2 também está nesse intervalo, mas não é uma métrica.
O artigo usa 1-cos sem o fator 1/2 e atribui intervalo [0,1]: essa normalização
não decorre da fórmula original.

Para medidas empíricas uniformes a,b, o solver resolve aproximadamente:

\[
OT_\varepsilon(a,b)=\min_{T\in\Pi(a,b)}
\langle T,C\rangle+\varepsilon\,KL(T\|a\otimes b).
\]

São expostos o custo de transporte, o objetivo regularizado, o erro de marginais
e as iterações. Ultrapassar a tolerância não produz uma aprovação silenciosa.
O padrão usa a divergência com autocustos removidos:

\[
S_\varepsilon(a,b)=OT_\varepsilon(a,b)
-\tfrac12 OT_\varepsilon(a,a)-\tfrac12 OT_\varepsilon(b,b).
\]

Isso não é W1 exato. O modo `transport` expõe o custo do plano regularizado e
pode produzir autocusto positivo. A deriva é a média entre âncoras de mesmo ID,
sem misturar prompts diferentes numa nuvem global intercambiável. A comparação
local usa o estado anterior; a histórica usa o estado inicial aprovado.

## Critic e passo

\[
L_c=w_tL_{task}+w_d\Delta+w_i\Phi_{immune}+w_a\Phi_{recurrence},
\qquad s=\exp(-\kappa L_c).
\]

O termo de recorrência é uma extensão explícita para padrões semelhantes ao
arquivo. Seu peso é zero no `RiskConfig` padrão e 0,5 no benchmark operacional.
Sua busca só considera assinaturas do mesmo domínio. Não equivale a uma prova
de que um texto recuperado é nocivo.

Calcula-se primeiro uma proposta sem atenuação e mede-se sua deriva. Limiares
podem rejeitá-la por deriva, oposição ou recorrência. Caso elegível, aplica-se
o passo atenuado e mede-se novamente a deriva real. A política conservadora pode
rejeitar uma proposta cujo passo atenuado seria aceitável; essa escolha é auditável.

O otimizador NumPy faz descida de gradiente nos fatores A,B, com clipping global
dos gradientes desses fatores. Para o banco k:

\[
W=W_0+\sum_k B_kA_k,\quad
\theta_k=\min(1,\lambda_k^{-\beta})g_k,\quad
(A_k,B_k)\leftarrow(A_k,B_k)-\eta_0\theta_ks\,clip_G(\nabla L_{task}).
\]

O sinal negativo corrige a ambiguidade da Eq. (6). Clipping limita o gradiente dos
fatores, não diretamente a norma da mudança funcional de W. A norma dos fatores
e o termo de segunda ordem delta-B delta-A também importam. O filtro posterior
mede a consequência nas âncoras, sem assumir equivalência entre os gradientes.

Ranks são inteiros, limitados pelas dimensões e por orçamento total. O mínimo é
um por banco; não é congelamento absoluto. Só o banco selecionado recebe gradiente,
mas isso não elimina interferência funcional entre domínios. A base W0 nunca é
mesclada nesta versão, para manter reversibilidade. NumPy soma bancos na inferência;
a integração PEFT usa um banco ativo por vez e mede âncoras em todas as rotas.

Escalar um gradiente antes do Adam não reproduz necessariamente o mesmo fator
no deslocamento dos parâmetros, devido à normalização por momentos. A integração
opcional implementa SGD direto sem momentum nem weight decay.

## Arquivo, replay e recuperação

O TCS armazena vetor, ID, domínio, severidade, contador e passo. Semelhanças acima
do limiar são fundidas; o excesso é removido por severidade com decaimento temporal.
Não há reconstrução de conteúdo apagado. A busca exata padrão custa O(Md).
O backend HNSW é opcional e reconstrói seu índice após mutações; não é uma
implementação otimizada de exclusões incrementais nem tem garantia logarítmica.

O exemplo defensivo exige um alvo seguro aprovado e um alvo negativo explícito:

\[
L_{def}=L_{MSE}+w\max(0,m+\|y-y_{safe}\|^2-\|y-y_{bad}\|^2).
\]

Rejeitar um episódio não fornece, por si só, o alvo seguro. A razão de 15% limita
amostras rejeitadas no sampler de replay; os 20% do benchmark são um orçamento de
ataque do fluxo de avaliação e têm outro denominador.

Auditorias periódicas comparam o estado corrente à época 0. Diante de violação,
reavaliam checkpoints do mais recente ao mais antigo e restauram um com deriva
no máximo metade do limiar. A época 0 é preservada separadamente. O modelo fica
congelado até `resume(review_note=...)`. Uma raiz contaminada continua contaminada;
proximidade à própria raiz não detecta esse problema.

Persistência usa JSON com checksum e substituição de arquivo no mesmo volume.
Isso detecta alterações acidentais e evita arquivos parcialmente escritos, mas
não autentica um prior nem constitui transação distribuída. Há exclusão mútua
para o processo local. Nenhuma thread de atendimento deve observar propostas
transitórias; use uma cópia separada do modelo em um serviço real.
