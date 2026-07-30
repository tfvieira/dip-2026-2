# Mapa dos materiais legados

## 1. Escopo da análise

Este documento registra a análise dos arquivos em
`.reference/tiago_legacy/` para apoiar o backlog reduzido DIP-04 a DIP-12.
Os materiais são referências históricas de aula, não componentes da
arquitetura atual.

As classificações usadas são:

- **Adaptar**: há exemplos, algoritmos ou sequências didáticas que podem ser
  reescritos para a API atual, com testes e compatibilidade com Colab.
- **Referência didática**: o conteúdo ajuda a planejar explicações ou
  atividades, mas o código não deve ser copiado.
- **Revisar origem**: há indicação de fonte externa, sem informação local
  suficiente sobre autoria ou licença, ou há forte sinal de material derivado.
- **Descartar**: o arquivo não deve orientar o backlog atual por ser duplicado,
  gerado, obsoleto, incompleto ou estar fora do escopo. A classificação não
  recomenda apagar arquivos de `.reference/`.

Nesta análise, "ponto de partida" significa reaproveitar objetivos,
experimentos e casos de teste. Não significa copiar scripts inteiros para
`src/` ou notebooks atuais.

## 2. Resumo executivo

### Fatos observados

- O diretório contém 38 arquivos: 20 scripts Python, 8 notebooks, 2 exportações
  HTML, 7 imagens e 1 README.
- Há referências úteis para todas as issues de DIP-04 a DIP-12, mas em graus
  diferentes: fundamentos, intensidade e visualização têm exemplos dispersos;
  filtragem, frequência, cor e morfologia têm roteiros amplos; segmentação,
  descrição e restauração possuem apenas recortes que exigem seleção e
  reescrita.
- A quantização aparece somente em apoio conceitual, e não foi encontrada uma
  implementação específica e reutilizável de quantização uniforme de níveis.
- Não foi encontrada implementação de fatiamento e reconstrução de planos de
  bits; esse item de DIP-05, se couber na capacidade, exige trabalho novo.
- Os exemplos de hash em `2019-1-RV.py` e
  `pis-Image-Hashing-with-OpenCV-and-Python.py` convertem comparações binárias
  em inteiros, mas não implementam bit-plane slicing.
- `motion-detection.py` e `watershed_example.py` declaram Python 2 no
  *shebang*.
- Seis arquivos usam `runfile`; cinco scripts ou notebooks usam
  `from dip import *`. O arquivo `dip.py` esperado por parte desses materiais
  não existe no diretório.
- Há chamadas ativas de `cv2.imshow` e `cv2.waitKey`, caminhos relativos como
  `../img` e `../db`, e caminhos de usuário em notebooks.
- Os códigos e notebooks citam 59 nomes de imagens. Apenas quatro desses nomes
  aparecem no próprio diretório legado. `cameraman_original.png` está no
  diretório `assets/images/` atual; a disponibilidade dos demais nomes no
  Google Drive público não é demonstrada pelos arquivos analisados.
- `008_morphology.py` usa os aliases removidos `np.bool` e `np.int`.
- `utils.py` define `bgr2rgb` duas vezes e repete funções existentes em
  `dip_script.py`, como cálculo de histogramas, transformação logarítmica,
  normalização e criação de discos.

### Recomendações para o planejamento

- Tratar `lec001-introduction.ipynb`, `lec-2019-10-24.ipynb`,
  `dip-lec-2019-10-31.ipynb`, `mocv-001-intro-read-write-histograms.py` e
  funções selecionadas de `utils.py` como fontes principais para fundamentos e
  intensidade.
- Usar `mocv-003-transformations.py`,
  `006_Filtering_Frequency_Domain.py`, os materiais `007_*`, os materiais
  `008_*` e `watershed_example.py` somente nos recortes temáticos indicados no
  mapeamento, respeitando os alertas de origem e compatibilidade.
- Reescrever os exemplos como chamadas pequenas à biblioteca, usando
  `download_course_image`, `ImageLoader` e Matplotlib.
- Não transportar estado global, imports curinga, loops de janela, caminhos ou
  funções de desenho de histograma acopladas ao cálculo.
- Confirmar autoria e licença antes de reaproveitar qualquer trecho marcado
  como **Revisar origem**.
- Validar cada imagem pelo nome no serviço público antes de incluí-la nos
  critérios de aceite de uma issue. Quando possível, preferir
  `cameraman_original.png` ou arrays sintéticos pequenos nos testes.

## 3. Mapeamento para DIP-04 a DIP-12

| Issue | Material legado que pode iniciar o trabalho | Aproveitamento realista | Lacuna que permanece |
| --- | --- | --- | --- |
| DIP-04 | `lec001-introduction.ipynb`; `lec-2019-10-24.ipynb`; `mocv-001-intro-read-write-histograms.py`; `dip_script.py`; `mocv-003-transformations.py`; `ab1-solution.py` | Reaproveitar sequências de leitura, criação de arrays, inspeção de shape/canais/dtype, visualização grayscale/RGB e experimentos de resize, rotação, translação, recorte e interpolação. | Integrar `ImageLoader` e `ImageCreator`, definir faixas e quantização introdutória, corrigir scripts auxiliares e criar APIs e visualizações próprias para Colab. `mocv-003-transformations.py` exige revisão de origem. |
| DIP-05 | `mocv-001-intro-read-write-histograms.py`; `lec001-introduction.ipynb`; `dip-lec-2019-10-31.ipynb`; `utils.py`; `dip_script.py`; `expcurve.py` | Adaptar histogramas grayscale/por canal, negativo, log, transformação linear por partes e gráficos de CDF. | Implementar gama e equalização com contratos claros; corrigir casos-limite dos helpers; planos de bits não existem no legado e só entram se houver capacidade. |
| DIP-06 | `mocv-003-transformations.py`; `dip_script.py`; `utils.py`; materiais `007_*` | Selecionar demonstrações de `filter2D`, média, Gauss, mediana, sharpening, Sobel, Laplaciano e Canny. | Definir correlação/convolução, bordas, dtype e parâmetros em API testável. O material principal exige revisão de origem e usa janelas OpenCV. |
| DIP-07 | `006_Filtering_Frequency_Domain.py` | Reaproveitar a sequência conceitual de DFT, espectro, centralização, reconstrução e filtros passa-baixa/passa-alta simples. | Reduzir o roteiro ao nível introdutório, separar magnitude e fase e eliminar trackbars, loops de janela, caminhos locais e trechos de restauração em frequência. |
| DIP-08 | `007_images_in_different_colorspaces.ipynb`; `007_images_in_different_colorspaces_i.py`; `mocv-001-intro-read-write-histograms.py`; `Problems.ipynb`; `utils.py`; `dip_script.py` | Adaptar separação/composição de canais, convenções BGR/RGB, HSV, grayscale, histogramas e exercícios de cor. | Completar e validar YCrCb e Lab, tornar a convenção de canais explícita e substituir estado global, imports curinga, caminhos locais e janelas OpenCV. |
| DIP-09 | `008_morphology.py`; `lec008-morphology.ipynb`; `008_morphology_old.py`; `mocv-003-transformations.py`; `Problems.ipynb` | Usar a sequência didática de imagens binárias, elementos estruturantes, erosão, dilatação, abertura e fechamento; selecionar gradiente, top-hat e black-hat conforme capacidade. | Restringir o conteúdo ao escopo da issue, atualizar APIs obsoletas e confirmar autoria/licença antes de qualquer adaptação. |
| DIP-10 | `watershed_example.py`; `mocv-003-transformations.py`; `Problems.ipynb`; `008_morphology.py`; `lec008-morphology.ipynb` | Adaptar limiarização global, Otsu, limiarização adaptativa, watershed como referência, refinamento morfológico e componentes conectados. | Organizar um fluxo de segmentação testável, parametrizar operações e produzir regiões sem incorporar os descritores de DIP-11. |
| DIP-11 | `008_morphology.py`; `008_morphology_old.py`; `dip_script.py`; `Problems.ipynb` | Usar `findContours`, `convexHull`, `drawContours`, `plotContour` e exercícios de regiões como referências para fronteiras e contornos. | Implementar APIs atuais para área, perímetro, centroide, bounding box, razão de aspecto, circularidade e momentos; o legado não fornece esse conjunto completo e validado. |
| DIP-12 (opcional) | `utils.py`; `dip_script.py`; `dip-lec-2019-10-31.ipynb`; `mocv-003-transformations.py`; `006_Filtering_Frequency_Domain.py` | Reaproveitar geradores simples de ruído normal e sal-e-pimenta e comparações de média, Gauss e mediana; usar o ruído periódico apenas como referência conceitual. | Corrigir helpers incompletos, implementar motion blur introdutório e limitar reconstrução ao exemplo; não há base para filtros inversos/Wiener avançados ou tomografia. |

## 4. Análise dos materiais principais

### `mocv-001-intro-read-write-histograms.py` — Adaptar

**Fatos observados:** contém leitura grayscale e colorida, inspeção de
dimensões, separação BGR, conversão para cinza/HSV e histogramas com
`cv2.calcHist` e `plt.hist`. Depende de `runfile('dip.py')`, da variável global
`folder`, de `printImgDims` e de `plotMultipleImgs`. Usa `devilwall.jpg` e
`baboon.png`, ausentes do diretório legado.

**Uso recomendado:** converter as sequências de leitura, inspeção e
visualização em exemplos para DIP-04; levar histogramas e intensidade para
DIP-05; e usar separação de canais/conversões como referência para DIP-08. Não
adaptar a gravação de `test.jpg` nem o carregamento por caminho local.

### `mocv-003-transformations.py` — Revisar origem

**Fatos observados:** reúne translação, rotação, resize com vários
interpoladores, pirâmides, recorte, operações com escalares, lógica, filtros,
limiarização, morfologia, bordas e transformações afins/perspectivas. Usa
`runfile('dip.py')`, muitos caminhos `images/...`, `cv2.imshow` e `cv2.waitKey`.
O arquivo aponta para um comparativo externo de interpolação e para
documentação beta do OpenCV 3.0, mas não informa a origem de todos os blocos.

**Uso recomendado:** após verificação de origem, selecionar interpoladores e
transformações geométricas para DIP-04, filtragem espacial e bordas para DIP-06,
operações morfológicas para DIP-09 e limiarização para DIP-10. O arquivo não
deve ser migrado como uma unidade.

### `006_Filtering_Frequency_Domain.py` — Referência didática

**Fatos observados:** implementa demonstrações de DFT, espectro, filtros
passa-baixa/passa-alta, imagens senoidais e ruído periódico. Também define
helpers como `scaleImage2_uchar`, `applyLogTransform` e `create2DGaussian`.
Depende de `../img`, trackbars, loops com `cv2.waitKey` e muitas chamadas a
`cv2.imshow`.

**Uso recomendado:** usar como referência principal de DIP-07, selecionando
somente DFT, IDFT, magnitude, fase, centralização, reconstrução e filtros
passa-baixa/passa-alta simples. O ruído periódico pode informar uma comparação
conceitual em DIP-12, mas restauração em frequência, trackbars e aprofundamento
matemático ficam fora do backlog.

### `007_images_in_different_colorspaces.ipynb` — Referência didática

**Fatos observados:** apresenta espaço vetorial RGB, gradiente de cor e
visualizações RGB/HSV. Importa `from dip import *`, usa `../img` e contém saída
de uma execução em caminho local.

**Uso recomendado:** preservar a explicação de canais e as visualizações
RGB/HSV como referência didática de DIP-08. A composição visual também pode
informar DIP-04.

### `007_images_in_different_colorspaces_i.py` — Referência didática

**Fatos observados:** é um roteiro amplo de BGR, YRB, HSV, CMYK, gradiente,
filtragem e segmentação por cor. Executa `runfile('dip_script.py')`, usa a
variável global `folder`, várias imagens externas e janelas OpenCV.

**Uso recomendado:** usar como inventário didático de DIP-08 e selecionar
somente separação/composição de canais, convenções BGR/RGB e conversões
previstas na issue. Os trechos de filtragem podem informar DIP-06 e os de
segmentação por cor devem permanecer fora do escopo de DIP-08. Não copiar
funções.

### `008_morphology.py` — Revisar origem

**Fatos observados:** cobre operações de conjuntos, erosão, dilatação, abertura,
fechamento, reconstrução, componentes, afinamento e morfologia grayscale. Usa
`from dip import *`, dezenas de imagens não presentes, `cv2.imshow`,
`np.bool`/`np.int` e APIs de SciPy/scikit-image. As seções de granulometria,
esqueleto e textura registram URLs externas.

**Uso recomendado:** após verificar fontes e licenças, usar a sequência de
operações básicas como referência de DIP-09, os exemplos de componentes como
apoio a DIP-10 e os trechos de contornos como apoio a DIP-11. Afinamento,
esqueleto, granulometria, textura e reconstrução morfológica ficam fora do
recorte atual.

### `lec008-morphology.ipynb` — Revisar origem

**Fatos observados:** organiza morfologia em uma sequência de aula mais clara
que os scripts, mas repete operações e a função de esqueleto associada a uma
URL externa. Usa `from dip import *` e imagens ausentes.

**Uso recomendado:** depois da revisão de origem, considerar a estrutura
didática como referência principal de notebook para DIP-09 e os exemplos de
componentes como apoio a DIP-10. O código não é ponto de partida direto de
implementação.

### `watershed_example.py` — Revisar origem

**Fatos observados:** declara Python 2, não importa `os`, depende de `folder`,
`bgr2rgb` e `water_coins.jpg`, e implementa uma sequência completa de watershed
com Otsu, abertura, transformada de distância e componentes conectados. A
autoria indicada no cabeçalho não esclarece a origem da sequência do exemplo.

**Uso recomendado:** usar como referência de fluxo para DIP-10 somente após
confirmar a proveniência. Otsu, abertura, transformada de distância e
componentes podem informar o encadeamento didático; watershed não amplia o
escopo mínimo da issue. `watershed_example.png` é apenas um resultado
renderizado e não deve ser migrado.

### `Problems.ipynb` — Referência didática

**Fatos observados:** contém exercícios e soluções de cor, morfologia,
segmentação e álgebra linear. Importa `utils` por curinga e fixa
`PATH = '/home/vieira/digital-image-processing/img'`.

**Uso recomendado:** reaproveitar enunciados selecionados de cor, morfologia,
segmentação e regiões em DIP-08, DIP-09, DIP-10 e DIP-11, respectivamente,
somente depois de separar solução e atividade e confirmar as imagens públicas.
Conteúdo de álgebra linear não deve ampliar o backlog.

### `dip_script.py` — Adaptar

**Fatos observados:** funciona como script de inicialização global: importa
várias bibliotecas, define `folder = '../img'` e agrega funções sem uma API
modular. `printImgDims`, `iminfo`, `plotMultipleImgs`,
`compute_piecewise_linear_val`, `scaleImage2_uchar`,
`compute_histogram_1C` e `compute_histogram_3C` se relacionam ao backlog.
`show_imgs` usa `cv2.imshow`. Vários helpers aparecem novamente em `utils.py`.

**Uso recomendado:** adaptar apenas campos, casos de aula e resultados
esperados: inspeção/visualização para DIP-04, intensidade e histogramas para
DIP-05, Sobel para DIP-06, canais para DIP-08, `plotContour` para DIP-11 e
sal-e-pimenta para a DIP-12 opcional. Não preservar o script de inicialização,
os nomes camelCase, o estado global nem o acoplamento entre cálculo e desenho.

### `utils.py` — Adaptar

**Fatos observados:** contém helpers menores para conversão BGR/RGB,
normalização, informação de imagem, histogramas, transformação linear por
partes e log. `bgr2rgb` é definido duas vezes. `im_info` desempacota
`h, w = img.shape` e portanto não atende imagens coloridas. Em
`get_piecewise_transformed_img`, o ramo de dtype diferente de `uint8` não
inicializa `out`. `color_gradient` referencia `F` antes da atribuição.

**Uso recomendado:** usar os exemplos e defeitos como fonte de casos de teste
para DIP-04 (representação e faixas), DIP-05 (intensidade e histogramas),
DIP-06 (Sobel), DIP-08 (canais) e DIP-12 (ruído normal). Nenhuma função deve
ser copiada sem reescrita e validação.

### Outros notebooks de aula

| Arquivo | Classificação | Fatos observados e uso recomendado |
| --- | --- | --- |
| `lec001-introduction.ipynb` | Adaptar | Melhor fonte inicial para DIP-04 e DIP-05: leitura, canais, histogramas, criação de arrays e conversão de faixa; também apoia a convenção de canais de DIP-08. Usa `../img/baboon.png`; uma célula cria `255 * ones` com `int8`, útil para explicar overflow, mas inadequada como implementação. |
| `2019-2-lec-001.ipynb` | Descartar | Repete leitura e separação de canais do notebook introdutório, fixa um caminho `/Users/...` e contém checagens incorretas `img.all() == None`. A exportação `2019-2-lec-001.slides.html` também deve ser descartada do planejamento. |
| `lec-2019-10-24.ipynb` | Adaptar | Fonte direta para DIP-04: helper de múltiplas imagens, translação, rotação, resize, pirâmides e recorte. As operações por canal podem apoiar DIP-08. Depende de `../db/lena.png`. |
| `dip-lec-2019-10-31.ipynb` | Adaptar | Fonte para DIP-05 pelo negativo e pelas demonstrações de intensidade. Mistura ponderada, máximo e diferença absoluta podem apoiar operações por canal em DIP-08. Usa `../db`; a célula de ruído contém `noise.` incompleto e deve ser ignorada em DIP-12. |
| `Building-an-Image-Hashing-Search-Engine-with-VP-Trees-and-OpenCV.ipynb` | Revisar origem | Possui título e objetivo de um tutorial de hashing, praticamente sem implementação. Está fora do núcleo atual e a origem deve ser confirmada. |

## 5. Inventário complementar

| Arquivo | Classificação | Motivo |
| --- | --- | --- |
| `008_morphology_old.py` | Descartar | Versão menor e duplicada de `008_morphology.py`, com o mesmo `from dip import *` e as mesmas dependências de imagens. |
| `008_morphology_scikit_image_000.py` | Revisar origem | Exemplo autocontido de `skimage.data.horse()` e `skeletonize`, com estrutura de galeria e sem licença/origem registrada no arquivo. |
| `2019-1-RV.py` | Descartar | Experimento de hash binário, depende de `runfile('dip.py')` e de `ckbd2.png`, ausente; não cobre planos de bits. |
| `ab1-solution.py` | Referência didática | Solução de avaliação com conceitos, criação de imagem e rotação matricial. Pode inspirar perguntas e comparações introdutórias em DIP-04, mas não deve ser copiada para a biblioteca. |
| `expcurve.py` | Referência didática | Gráficos de log e CDF podem apoiar DIP-05; a expressão rotulada como normal usa `np.exp2`, portanto precisa de correção conceitual antes de uso. |
| `getcameraframe.py` | Descartar | Captura local, `runfile`, HOG e `cv2.imshow`; extensão de webcam fora do backlog e incompatível com Colab. |
| `ip_camera.py` | Descartar | Captura RTSP com threads e janela OpenCV, fora do núcleo e dependente de hardware/rede. |
| `motion-detection.py` | Descartar | Declara Python 2, está incompleto e depende de `imutils` e câmera local. |
| `pis-Image-Hashing-with-OpenCV-and-Python.py` | Revisar origem | Registra explicitamente uma URL do PyImageSearch e contém parte de um tutorial de dHash; está fora do backlog. |
| `mocv-002-draw-shapes.py` | Descartar | Desenho de primitivas já não faz parte do backlog reduzido e o script depende de `runfile('dip.py')`. |
| `Pode apagar 001_intro.py` | Descartar | O próprio nome indica descarte; duplica leitura, criação e histogramas, usa `cv2.imshow` e contém uma função de URL com APIs incorretas de `urllib`. |
| `tmp.py` | Descartar | Arquivo de experimentação com arrays e superfície 3D, sem objetivo de aula ou função reutilizável. |
| `README.md` | Referência didática | Registra que o diretório servia como material auxiliar da disciplina e identifica o professor; não documenta licença dos exemplos. |
| `lec001-introduction.slides.html` | Descartar | Exportação gerada e duplicada de notebook, com dependências web incorporadas; não deve orientar implementação. |
| `frame.png` | Descartar | Asset do exemplo externo de hashing; arquivo binário fora do backlog. |
| `lab_histograms.png` | Descartar | Resultado gráfico sem código ou proveniência suficiente; os histogramas devem ser regenerados. |
| `mask.png` | Descartar | Imagem auxiliar sem vínculo explícito com uma atividade atual; máscaras podem ser criadas por arrays nos testes. |
| `moon.png` | Descartar | Asset do experimento de hash, fora do backlog. |
| `moon_result.png` | Descartar | Resultado do experimento de hash, fora do backlog. |
| `test.jpg` | Descartar | Saída gerada por `mocv-001-intro-read-write-histograms.py`; não é fonte didática. |
| `watershed_example.png` | Descartar | Resultado renderizado do exemplo de watershed; binário fora do backlog. |

## 6. Duplicações e precedência

### Fatos observados

- `utils.py` repete `compute_histogram_1C`, `compute_histogram_3C`,
  `compute_piecewise_linear_val`, `applyLogTransform`/`log_transform`,
  `scaleImage2_uchar` e helpers geométricos de `dip_script.py`.
- `008_morphology_old.py`, `008_morphology.py` e
  `lec008-morphology.ipynb` compartilham a mesma sequência de operações e
  várias imagens.
- `2019-2-lec-001.ipynb`, `lec001-introduction.ipynb`,
  `mocv-001-intro-read-write-histograms.py` e
  `Pode apagar 001_intro.py` repetem leitura, separação de canais e
  histogramas.
- Os dois arquivos `.slides.html` são exportações geradas de notebooks.

### Recomendação

Para DIP-04 a DIP-12, usar a seguinte precedência:

1. notebook de aula mais claro para preservar a sequência didática;
2. `utils.py` para localizar algoritmos e casos-limite;
3. `dip_script.py` apenas para recuperar comportamento histórico ausente;
4. versões `old`, exportações HTML e resultados de imagem somente para
   comparação, nunca como fonte de implementação.

## 7. Condições para adaptação

Todo material classificado como **Adaptar** ainda deve:

1. ser reescrito para Python 3.10 ou superior;
2. usar imports explícitos de `dip_toolkit`;
3. eliminar `runfile`, variáveis globais e caminhos de usuário;
4. usar `download_course_image` e validar a existência do asset público;
5. substituir janelas OpenCV por Matplotlib nos notebooks;
6. separar cálculo, retorno de dados e visualização;
7. receber type hints, docstrings, validações e testes;
8. ter a origem confirmada quando houver qualquer indício de material externo.
