# Roadmap do DIP Toolkit

## 1. Objetivo da disciplina

O DIP Toolkit deve apoiar o ensino progressivo de Processamento Digital de
Imagens com APIs claras, testes automatizados e notebooks executáveis no Google
Colab. O objetivo da disciplina é concluir blocos didáticos utilizáveis e
coerentes entre si, que também sirvam como material de consulta, e não cobrir
todo o conteúdo de maneira superficial.

As issues DIP-02 — Revisar e evoluir o `ImageLoader` — e DIP-03 — Revisar e
evoluir o `ImageCreator` — já existem e são mantidas como base do planejamento.
O novo backlog deste documento vai de DIP-04 a DIP-12.

## 2. Direcionamento do orientador

A prioridade definitiva da disciplina é:

1. fundamentos, carregamento e representação de imagens;
2. transformações de intensidade, histogramas e contraste;
3. filtragem espacial;
4. Fourier apenas em nível introdutório;
5. processamento de imagens coloridas;
6. morfologia;
7. segmentação e descritores;
8. representação e descrição;
9. restauração e reconstrução somente como conteúdo complementar.

Fourier não deve receber filtros avançados, restauração em frequência ou
desenvolvimento matemático aprofundado. Restauração e reconstrução são
opcionais, sem aprofundamento, e não bloqueiam nenhuma entrega obrigatória.

## 3. Capacidade da equipe

O projeto será executado principalmente por duas pessoas durante a disciplina.
Por isso:

- cada pessoa deve manter no máximo uma issue em desenvolvimento;
- pode haver no máximo duas issues simultaneamente;
- os pull requests devem receber revisão cruzada;
- testes e notebook pertencem à própria issue, não a entregas posteriores;
- somente as próximas issues a executar devem ser cadastradas no GitHub;
- as demais permanecem neste roadmap até haver capacidade;
- DIP-12 só deve ser iniciada se houver capacidade após o escopo obrigatório.

## 4. Princípios de execução

1. Preservar compatibilidade com Python 3.10 ou superior e Google Colab.
2. Entregar blocos didáticos completos e utilizáveis antes de ampliar a
   cobertura.
3. Manter API, validações, type hints, docstrings, testes e notebook na mesma
   issue, quando aplicável.
4. Separar processamento e visualização para permitir testes numéricos.
5. Usar nos notebooks somente imagens obtidas por `download_course_image`.
6. Preferir arrays sintéticos pequenos nos testes e não adicionar imagens ou
   outros arquivos binários ao Git.
7. Tratar materiais legados como referências, nunca como código pronto.
8. Não transportar `runfile`, `from dip import *`, estado global, caminhos
   locais, `cv2.imshow` ou dependências implícitas de imagens.
9. Confirmar autoria, licença e disponibilidade das imagens antes de adaptar
   qualquer exemplo legado.

## 5. Backlog reduzido

### Issues já existentes

- **DIP-02 — Revisar e evoluir o ImageLoader.**
- **DIP-03 — Revisar e evoluir o ImageCreator.**

### DIP-04 — Consolidar fundamentos e representação digital de imagens

**Escopo:**

- integrar `ImageLoader` e `ImageCreator` ao fluxo didático;
- oferecer visualização adequada para Colab;
- trabalhar shape, dimensões, canais, dtype e faixas;
- apresentar representação grayscale e colorida;
- introduzir amostragem, reamostragem e quantização;
- comparar interpolação nearest, bilinear e bicúbica;
- corrigir os scripts auxiliares relacionados.

**Dependências:** DIP-02 e DIP-03.

### DIP-05 — Implementar transformações de intensidade, histogramas e contraste

**Escopo:**

- negativo;
- transformação logarítmica;
- transformação gama;
- transformação linear por partes;
- planos de bits, quando o escopo permitir;
- histogramas grayscale e por canal;
- CDF;
- equalização;
- comparação antes/depois.

**Dependências:** DIP-04.

### DIP-06 — Evoluir filtragem espacial

**Escopo:**

- aplicação de kernels;
- correlação ou convolução;
- filtro de média;
- filtro gaussiano;
- filtro de mediana;
- Sobel;
- Laplaciano;
- sharpening;
- Canny parametrizável.

**Dependências:** DIP-04 e DIP-05.

### DIP-07 — Implementar Fourier introdutório

**Escopo limitado:**

- DFT;
- IDFT;
- magnitude;
- fase;
- centralização do espectro;
- reconstrução;
- filtros passa-baixa e passa-alta simples;
- comparação introdutória entre espaço e frequência.

**Não incluir:**

- filtros avançados;
- restauração em frequência;
- desenvolvimento matemático aprofundado;
- filtros Wiener ou inversos.

**Dependências:** DIP-06.

### DIP-08 — Evoluir processamento de imagens coloridas

**Escopo:**

- convenção explícita RGB/BGR;
- separação e combinação de canais;
- conversões RGB/BGR, HSV, YCrCb, Lab e grayscale;
- histogramas por canal;
- operações básicas por canal;
- exemplos didáticos.

**Dependências:** DIP-04 e DIP-05.

### DIP-09 — Implementar morfologia matemática

**Escopo:**

- imagens binárias;
- elementos estruturantes;
- erosão;
- dilatação;
- abertura;
- fechamento;
- gradiente morfológico;
- top-hat e black-hat quando houver capacidade.

**Dependências:** DIP-06.

### DIP-10 — Implementar segmentação e extração inicial de regiões

**Escopo:**

- limiarização global;
- Otsu;
- limiarização adaptativa;
- segmentação baseada em bordas;
- refinamento de máscaras com morfologia;
- componentes conectados ou contornos iniciais.

**Dependências:** DIP-05, DIP-06 e DIP-09.

DIP-10 produz a segmentação e as regiões que serão consumidas por DIP-11.

### DIP-11 — Implementar representação, descrição e descritores básicos

**Escopo:**

- representação de fronteiras e regiões;
- contornos;
- área;
- perímetro;
- centroide;
- bounding box;
- razão de aspecto;
- circularidade;
- momentos básicos;
- descritores básicos derivados das regiões segmentadas.

**Dependências:** DIP-10.

DIP-11 representa e descreve as regiões produzidas por DIP-10; não redefine o
processo de segmentação.

### DIP-12 — Introduzir restauração e reconstrução

**Caráter:** issue explicitamente opcional, iniciada somente se houver
capacidade.

**Escopo limitado:**

- modelos simples de ruído;
- ruído gaussiano;
- sal-e-pimenta;
- motion blur introdutório;
- comparação simples de filtros de redução de ruído;
- reconstrução apenas quando necessária ao exemplo introdutório.

**Não incluir:**

- filtros inversos avançados;
- restauração aprofundada;
- Wiener avançado;
- reconstrução tomográfica.

**Dependências:** DIP-06.

## 6. Dependências

| Issue | Dependências diretas | Bloqueia |
| --- | --- | --- |
| DIP-02 | — | DIP-04 |
| DIP-03 | — | DIP-04 |
| DIP-04 | DIP-02, DIP-03 | DIP-05, DIP-06, DIP-08 |
| DIP-05 | DIP-04 | DIP-06, DIP-08, DIP-10 |
| DIP-06 | DIP-04, DIP-05 | DIP-07, DIP-09, DIP-10 e, opcionalmente, DIP-12 |
| DIP-07 | DIP-06 | — |
| DIP-08 | DIP-04, DIP-05 | — |
| DIP-09 | DIP-06 | DIP-10 |
| DIP-10 | DIP-05, DIP-06, DIP-09 | DIP-11 |
| DIP-11 | DIP-10 | — |
| DIP-12 (opcional) | DIP-06 | — |

O grafo é acíclico. Nenhuma issue obrigatória depende de DIP-12.

## 7. Ordem recomendada

A ordem didática recomendada, idêntica à prioridade definida pelo orientador,
é:

1. DIP-02 e DIP-03 — preparar carregamento e criação;
2. DIP-04 — consolidar fundamentos e representação;
3. DIP-05 — transformações de intensidade, histogramas e contraste;
4. DIP-06 — filtragem espacial;
5. DIP-07 — Fourier introdutório;
6. DIP-08 — imagens coloridas;
7. DIP-09 — morfologia;
8. DIP-10 — segmentação e extração inicial de regiões;
9. DIP-11 — representação, descrição e descritores;
10. DIP-12 — restauração e reconstrução, somente se houver capacidade.

## 8. Paralelização possível

A ordem didática orienta as entregas, mas a equipe pode aproveitar até duas
frentes quando as dependências estiverem concluídas:

| Após concluir | Frente 1 | Frente 2 possível |
| --- | --- | --- |
| DIP-02 e DIP-03 | DIP-04 | — |
| DIP-04 | DIP-05 | — |
| DIP-05 | DIP-06 | DIP-08 |
| DIP-06 | DIP-07 | concluir DIP-08 ou iniciar DIP-09 |
| DIP-09 | DIP-10 | finalizar pendências independentes de DIP-07/DIP-08 |
| DIP-10 | DIP-11 | DIP-12, somente se o obrigatório estiver protegido |

A paralelização não altera a prioridade de conteúdo nem autoriza mais de duas
issues simultâneas.

## 9. Escopo obrigatório

O núcleo obrigatório compreende DIP-02 a DIP-11:

- carregamento, criação, fundamentos e representação;
- transformações de intensidade, histogramas e contraste;
- filtragem espacial;
- Fourier somente introdutório;
- processamento de imagens coloridas;
- morfologia;
- segmentação e extração de regiões;
- representação, descrição e descritores básicos.

Cada bloco deve ficar utilizável em aula antes do início de extensões
opcionais.

## 10. Escopo opcional

DIP-12 é o único item opcional deste backlog. Deve permanecer superficial e
didático, limitado a modelos simples de ruído, motion blur introdutório,
comparações básicas de redução de ruído e reconstrução estritamente necessária
ao exemplo.

Não fazem parte do escopo da disciplina filtros inversos ou Wiener avançados,
restauração aprofundada, restauração em frequência, reconstrução tomográfica,
wavelets, compressão, reconhecimento, classificação ou treinamento de
modelos. Esses temas podem permanecer registrados como lacunas ou extensões,
sem criar dependência para o backlog atual.

## 11. Critérios gerais de aceite

Quando aplicável, cada issue deve entregar:

- API clara;
- validações;
- type hints;
- docstrings;
- testes automatizados;
- notebook didático;
- compatibilidade com Google Colab;
- imagens obtidas por `download_course_image`;
- `ruff check .` aprovado;
- `pytest` aprovado;
- CI aprovado.

O notebook e os testes fazem parte da conclusão da própria issue. Uma entrega
não deve ser considerada concluída com essas partes transferidas para uma issue
futura.

## 12. Uso dos materiais legados

O inventário, as classificações, as análises técnicas e o mapeamento por issue
estão em `docs/LEGACY_MATERIALS_MAP.md`.

Os arquivos de `.reference/tiago_legacy/` são somente referências. Antes de
usar um material:

1. consultar a classificação e a indicação da issue;
2. confirmar autoria e licença quando houver origem incerta ou externa;
3. selecionar apenas conceitos, experimentos e resultados esperados;
4. reescrever o exemplo para a API atual e Python 3.10 ou superior;
5. remover `runfile`, imports curinga, estado global e caminhos locais;
6. substituir `cv2.imshow` por visualização compatível com Colab;
7. validar o nome da imagem no serviço público e usar
   `download_course_image`;
8. adicionar validações, type hints, docstrings e testes.

Códigos legados não podem ser copiados diretamente para `src/` ou notebooks.
