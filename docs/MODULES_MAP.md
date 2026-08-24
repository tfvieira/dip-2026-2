# Mapa dos módulos existentes

## 1. Resumo executivo

Esta análise considera os 18 arquivos existentes em
`src/dip_toolkit/modules/`, além de `src/dip_toolkit/assets.py`, dos testes em
`tests/`, do notebook `notebooks/00_validacao_colab.ipynb` e da documentação
principal do projeto.

Fato observado: o toolkit já possui módulos com nomes ligados a carregamento,
criação, operações aritméticas, transformações geométricas, histogramas,
filtragem, Fourier, wavelets, cor, extração de bordas, estatística,
classificação, treinamento de modelos, geração de dados, visualização e webcam.
No entanto, a maior parte desses módulos ainda é inicial ou parcial. O bloco
mais funcional é o fluxo de infraestrutura composto por
`src/dip_toolkit/assets.py`, `src/dip_toolkit/modules/image_loader.py`,
`tests/test_assets.py`, `tests/test_image_loader.py` e
`notebooks/00_validacao_colab.ipynb`.

Áreas com alguma cobertura observada:

- fundamentos de entrada e saída de imagens;
- representação básica de arrays e metadados;
- operações geométricas simples;
- operações aritméticas simples;
- histograma básico;
- pontos iniciais para Fourier, wavelets, cor e classificação genérica.

Áreas com menor cobertura observada:

- amostragem e quantização;
- transformações de intensidade;
- melhoria de contraste;
- restauração e reconstrução;
- compressão;
- morfologia matemática;
- segmentação;
- representação de fronteiras e regiões;
- descritores de características além de bordas Canny.

Principais riscos técnicos observados:

- ausência ampla de type hints em métodos públicos, exceto em `ImageLoader`;
- ausência ampla de docstrings em classes e em alguns métodos públicos;
- validações insuficientes de forma, tipo, faixa, canais, parâmetros e estado;
- inconsistência conceitual entre nomes RGB e uso efetivo de BGR em OpenCV;
- aleatoriedade global e não reproduzível em vários pontos;
- dependência de interface gráfica em `WebcamStreamReader.show_stream`;
- falta de testes para quase todos os módulos;
- falta de notebooks didáticos para quase todos os temas;
- responsabilidades complementares misturadas ao núcleo clássico do livro.

Conclusão: a distância para cobrir Gonzalez e Woods ainda é grande. O
repositório tem uma fundação útil para Colab e para carregamento de imagens,
mas nenhuma área temática clássica possui, neste momento, cobertura didática
adequada com API coerente, validações, testes e notebook próprio. A evolução
deve priorizar padronização técnica, testes e notebooks junto com cada bloco
didático.

## 2. Critério de maturidade

A maturidade abaixo avalia o estado técnico de cada módulo. Ela é diferente da
cobertura temática: um módulo pode estar tecnicamente funcional, mas cobrir uma
área pequena do conteúdo; ou pode tocar em uma área importante, mas ainda estar
incompleto.

- `Inicial`: possui poucas funcionalidades ou apenas um esqueleto.
- `Parcial`: possui funcionalidades úteis, mas ainda está incompleto.
- `Funcional`: atende ao objetivo principal, mas pode ser melhorado.
- `Necessita revisão`: possui problemas técnicos, conceituais ou de
  organização relevantes.

## 3. Inventário detalhado dos módulos

### `src/dip_toolkit/modules/__init__.py`

- Responsabilidade aparente: marcar o pacote `dip_toolkit.modules` e documentar
  que os módulos serão amadurecidos gradualmente.
- Classes públicas: nenhuma.
- Funções e métodos públicos: nenhum.
- Funcionalidades implementadas: apenas docstring de pacote.
- Principais dependências: nenhuma.
- Testes existentes: nenhum teste direto.
- Notebooks existentes: nenhum uso direto.
- Problemas técnicos observados: nenhum erro técnico observado no arquivo.
- Limitações: não organiza uma API agregada para `dip_toolkit.modules`; isso é
  uma limitação de organização, não uma falha funcional.
- Conteúdos ausentes relacionados: política de exportação e organização de
  submódulos.
- Área temática: organização interna do toolkit.
- Maturidade: `Inicial`.
- Evidências no código: docstring em `src/dip_toolkit/modules/__init__.py`.

### `src/dip_toolkit/modules/feature_extractor.py`

- Responsabilidade aparente: extrair características de imagens.
- Classes públicas: `FeatureExtractor`.
- Funções e métodos públicos: `FeatureExtractor.extract_edges(image)`.
- Funcionalidades implementadas: aplica `cv.Canny(image, 100, 200)` e retorna
  a imagem de bordas.
- Principais dependências: OpenCV (`cv2`).
- Testes existentes: nenhum.
- Notebooks existentes: nenhum.
- Problemas técnicos observados: classe sem docstring; método sem docstring e
  sem type hints; limiares Canny fixos; não valida tipo, dimensionalidade,
  imagem vazia ou canal; não diferencia detecção de bordas de descritores.
- Limitações: cobre apenas bordas Canny com parâmetros fixos.
- Conteúdos ausentes relacionados: gradientes, descritores de fronteira,
  descritores regionais, momentos, textura, cantos, blobs, HOG, LBP e
  descritores invariantes.
- Área temática: descritores e características; segmentação por bordas.
- Maturidade: `Inicial`.
- Evidências no código: `FeatureExtractor` em
  `src/dip_toolkit/modules/feature_extractor.py:4` e `extract_edges` em
  `src/dip_toolkit/modules/feature_extractor.py:8`.

### `src/dip_toolkit/modules/fourier_transformer.py`

- Responsabilidade aparente: apresentar o fluxo introdutório da transformada de
  Fourier para imagens grayscale 2D e a aplicação de máscaras ideais no domínio
  da frequência.
- Classes públicas: `FourierTransformer`.
- Funções e métodos públicos:
  `dft(image)`, `idft(spectrum)`, `magnitude(spectrum)`, `phase(spectrum)`,
  `ideal_low_pass_mask(shape, cutoff)`,
  `ideal_high_pass_mask(shape, cutoff)` e `apply_mask(spectrum, mask)`.
- Funcionalidades implementadas: DFT 2D centralizada e inversa com NumPy,
  magnitude e fase numéricas, máscaras circulares ideais passa-baixa e
  passa-alta e aplicação de máscara binária ao espectro complexo.
- Principais dependências: NumPy.
- Testes existentes: `tests/test_fourier_transformer.py`, cobrindo o contrato,
  as validações e os fluxos de reconstrução e filtragem ideal.
- Notebooks existentes: `notebooks/06_intro_fourier.ipynb`, com exemplos de DFT,
  magnitude, fase, reconstrução e máscaras ideais.
- Limitações: escopo deliberadamente introdutório; não cobre tópicos avançados
  de filtragem no domínio da frequência, restauração ou reconstrução.
- Conteúdos ausentes relacionados: convolução no domínio da frequência,
  teorema da convolução, padding e filtros de restauração inversa e Wiener.
- Área temática: transformadas e filtragem no domínio da frequência.
- Maturidade: `Funcional` para o escopo introdutório.
- Evidências no código: `FourierTransformer` em
  `src/dip_toolkit/modules/fourier_transformer.py`, testes em
  `tests/test_fourier_transformer.py` e exemplos em
  `notebooks/06_intro_fourier.ipynb`.

### `src/dip_toolkit/modules/generate_data.py`

- Responsabilidade aparente: gerar dados sintéticos 2D para exemplos.
- Classes públicas: `GenerateData`.
- Funções e métodos públicos:
  `GenerateData.generate_halfmoon(...)`.
- Funcionalidades implementadas: gera pontos em formato de meia-lua por
  amostragem polar uniforme ou normal; pode retornar tupla `(x, y)` ou array
  `N x 2`.
- Principais dependências: NumPy.
- Testes existentes: nenhum.
- Notebooks existentes: nenhum.
- Problemas técnicos observados: ausência de type hints; classe sem docstring;
  `__init__(seed=None)` chama `np.random.seed(seed)` e altera o estado global
  de aleatoriedade; não valida `n_samples`, intervalos de raio e ângulo,
  desvios ou tipo de retorno.
- Limitações: extensão de dados sintéticos, não processamento de imagem
  clássico; gera apenas meia-lua.
- Conteúdos ausentes relacionados: datasets didáticos de imagem, máscaras,
  formas binárias e ruídos controlados para restauração/segmentação.
- Área temática: extensão do toolkit, geração de dados sintéticos.
- Maturidade: `Parcial`.
- Evidências no código: `GenerateData` em
  `src/dip_toolkit/modules/generate_data.py:6` e `generate_halfmoon` em
  `src/dip_toolkit/modules/generate_data.py:11`.

### `src/dip_toolkit/modules/image_analysis.py`

- Responsabilidade aparente: análise básica de imagens.
- Classes públicas: `ImageAnalysis`.
- Funções e métodos públicos: `ImageAnalysis.compute_histogram(image)`.
- Funcionalidades implementadas: calcula histograma de 256 bins do canal 0 com
  `cv.calcHist([image], [0], None, [256], [0, 256])`.
- Principais dependências: OpenCV.
- Testes existentes: nenhum.
- Notebooks existentes: nenhum.
- Problemas técnicos observados: classe e método sem docstring; ausência de
  type hints; não valida tipo, dtype, número de canais, faixa, máscara ou
  número de bins; usa sempre o canal 0.
- Limitações: histograma básico apenas; não implementa histograma normalizado,
  cumulativo, equalização, especificação ou métricas de contraste.
- Conteúdos ausentes relacionados: histogramas por canal, equalização,
  matching, estatísticas descritivas por região e contraste.
- Área temática: histogramas e melhoria de contraste.
- Maturidade: `Inicial`.
- Evidências no código: `ImageAnalysis` em
  `src/dip_toolkit/modules/image_analysis.py:4` e `compute_histogram` em
  `src/dip_toolkit/modules/image_analysis.py:8`.

### `src/dip_toolkit/modules/image_classifier.py`

- Responsabilidade aparente: treinar e avaliar classificadores tradicionais.
- Classes públicas: `ImageClassifier`.
- Funções e métodos públicos: `fit(X, y)`, `predict(X)` e
  `evaluate(X_test, y_test)`.
- Funcionalidades implementadas: instancia KNN, SVM, árvore de decisão,
  GaussianNB, regressão logística e MLP; treina todos; retorna predições e
  acurácia.
- Principais dependências: scikit-learn.
- Testes existentes: nenhum.
- Notebooks existentes: nenhum.
- Problemas técnicos observados: ausência de type hints; classe sem docstring;
  não valida dados, rótulos, dimensionalidade ou estado treinado; modelos com
  aleatoriedade não recebem `random_state`; mistura vários algoritmos em uma
  única API rígida.
- Limitações: não define fluxo de características de imagem; não há divisão de
  treino/teste, métricas além de acurácia ou persistência de modelo.
- Conteúdos ausentes relacionados: pipeline completo de reconhecimento com
  extração de atributos, validação, matriz de confusão e avaliação didática.
- Área temática: reconhecimento e classificação; extensão de aprendizado de
  máquina.
- Maturidade: `Parcial`.
- Evidências no código: `ImageClassifier` em
  `src/dip_toolkit/modules/image_classifier.py:10`, `fit` em
  `src/dip_toolkit/modules/image_classifier.py:19`, `predict` em
  `src/dip_toolkit/modules/image_classifier.py:30` e `evaluate` em
  `src/dip_toolkit/modules/image_classifier.py:44`.

### `src/dip_toolkit/modules/image_color_processor.py`

- Responsabilidade aparente: processar imagens coloridas e visualizar canais.
- Classes públicas: `ColorImageProcessor`.
- Funções e métodos públicos:
  `plot_rgb_histograms(image)`, `convert_color_space(image, conversion)`,
  `rgb_to_cmyk(image)` e `plot_rgb_3d_cube(image)`.
- Funcionalidades implementadas: histogramas por canal, conversões BGR para
  HSV/YCrCb/Lab/cinza, conversão manual para CMYK e dispersão 3D de cores.
- Principais dependências: OpenCV, NumPy e Matplotlib.
- Testes existentes: nenhum.
- Notebooks existentes: nenhum.
- Problemas técnicos observados: ausência de type hints; classe sem docstring;
  nomes e docstrings usam RGB, mas as conversões usam constantes BGR do OpenCV
  (`cv2.COLOR_BGR2HSV`, `cv2.COLOR_BGR2YCrCb`, `cv2.COLOR_BGR2Lab`,
  `cv2.COLOR_BGR2GRAY`); `plot_rgb_histograms` usa cores `("b", "g", "r")`;
  `plot_rgb_3d_cube` nomeia `r, g, b = cv2.split(image)`, embora OpenCV carregue
  BGR por padrão; não valida canais ou dtype.
- Limitações: não cobre processamento por pseudo-cor, fatiamento de intensidade,
  correção de cor, balanço de branco ou modelos de cor com API consistente.
- Conteúdos ausentes relacionados: modelos de cor, transformação entre espaços,
  histogramas por canal, segmentação por cor e operações de cor didáticas.
- Área temática: processamento de imagens coloridas.
- Maturidade: `Necessita revisão`.
- Evidências no código: `ColorImageProcessor` em
  `src/dip_toolkit/modules/image_color_processor.py:6`,
  `convert_color_space` em
  `src/dip_toolkit/modules/image_color_processor.py:24` e `rgb_to_cmyk` em
  `src/dip_toolkit/modules/image_color_processor.py:40`.

### `src/dip_toolkit/modules/image_creator.py`

- Responsabilidade aparente: criar arrays de imagem e ruídos sintéticos.
- Classes públicas: `ImageCreator`.
- Funções e métodos públicos:
  `create_filled_image(shape, value=0, dtype=np.uint8)`,
  `create_random_image(shape, distribution="uniform", dtype=np.uint8, **kwargs)`,
  `change_image_dtype(image, dtype)`,
  `create_zeros_image(shape, dtype=np.uint8)`,
  `create_ones_image(shape, dtype=np.uint8)` e
  `create_salt_and_pepper_noise(height=100, width=100, salt_prob=0.05, pepper_prob=0.05)`.
- Funcionalidades implementadas: cria imagens preenchidas, zeros, uns,
  aleatórias uniformes/normais/Rayleigh, converte dtype e gera ruído
  sal-e-pimenta em matriz `float64`.
- Principais dependências: NumPy.
- Testes existentes: nenhum.
- Notebooks existentes: nenhum.
- Problemas técnicos observados: ausência de type hints; classe sem docstring;
  uso excessivo de `**kwargs`; não valida shape, dtype, probabilidades ou
  distribuição; usa `np.random` global; para dtype de ponto flutuante, o padrão
  `high=255` é cortado para `1.0`, gerando comportamento possivelmente
  surpreendente; `create_salt_and_pepper_noise` usa fundo `0.5`, sal `1.0` e
  pimenta `-1.0`, faixa diferente da convenção usual `[0, 1]`.
- Limitações: não cria padrões didáticos clássicos, rampas, tabuleiros,
  impulsos, degraus, discos, máscaras ou imagens com quantização controlada.
- Conteúdos ausentes relacionados: representação, amostragem, quantização,
  imagens sintéticas para convolução, ruído e restauração.
- Área temática: fundamentos, representação e geração didática de imagens.
- Maturidade: `Parcial`.
- Evidências no código: `ImageCreator` em
  `src/dip_toolkit/modules/image_creator.py:4`, `create_random_image` em
  `src/dip_toolkit/modules/image_creator.py:12` e
  `create_salt_and_pepper_noise` em
  `src/dip_toolkit/modules/image_creator.py:49`.

### `src/dip_toolkit/modules/image_drawer.py`

- Responsabilidade aparente: desenhar formas e texto sobre imagens.
- Classes públicas: `ImageDrawer`.
- Funções e métodos públicos:
  `draw_circle`, `draw_line`, `draw_rectangle`, `draw_ellipse`, `put_text` e
  `mask_image`.
- Funcionalidades implementadas: chama funções OpenCV para desenhar em cópia da
  imagem e aplica máscara com `cv.bitwise_and`.
- Principais dependências: OpenCV.
- Testes existentes: nenhum.
- Notebooks existentes: nenhum.
- Problemas técnicos observados: ausência de type hints; classe sem docstring;
  não valida coordenadas, cores, espessura, dtype ou número de canais; `mask_image`
  exige `image.shape == mask.shape`, o que impede o uso comum de máscara 2D
  sobre imagem colorida.
- Limitações: utilitário gráfico, não cobre morfologia, segmentação ou
  representação por si só.
- Conteúdos ausentes relacionados: desenho de máscaras binárias, regiões,
  contornos, fronteiras e marcadores com semântica didática.
- Área temática: fundamentos e visualização auxiliar; extensão utilitária.
- Maturidade: `Parcial`.
- Evidências no código: `ImageDrawer` em
  `src/dip_toolkit/modules/image_drawer.py:4` e `mask_image` em
  `src/dip_toolkit/modules/image_drawer.py:43`.

### `src/dip_toolkit/modules/image_loader.py`

- Responsabilidade aparente: carregar imagens locais, em lote, de diretórios e
  de URLs diretas.
- Classes públicas: `ImageLoader`.
- Funções e métodos públicos:
  `load_image(filename: str | Path, flags: int = cv.IMREAD_COLOR) -> np.ndarray`,
  `load_images(filenames: Iterable[str | Path], flags: int = cv.IMREAD_COLOR, *, skip_invalid: bool = False) -> list[np.ndarray]`,
  `load_images_from_directory(directory: str | Path, flags: int = cv.IMREAD_COLOR, *, recursive: bool = False, skip_invalid: bool = False) -> list[np.ndarray]` e
  `load_image_from_url(url: str, flags: int = cv.IMREAD_COLOR, *, timeout_seconds: float = 30.0) -> np.ndarray`.
- Funcionalidades implementadas: valida arquivo local, decodifica com OpenCV,
  carrega listas, carrega diretórios filtrando extensões válidas, ordena arquivos
  alfabeticamente, suporta busca recursiva, baixa imagem de URL direta e
  decodifica com `cv.imdecode`.
- Principais dependências: OpenCV, NumPy, `pathlib`, `urllib`.
- Testes existentes: `tests/test_image_loader.py` cobre carregamento colorido,
  escala de cinza, arquivo ausente e carregamento de diretório em ordem.
- Notebooks existentes: `notebooks/00_validacao_colab.ipynb` usa
  `ImageLoader().load_image(image_path)`.
- Problemas técnicos observados: não há testes para `load_images`,
  `skip_invalid`, recursão, arquivo inválido e URL; não valida flags OpenCV; não
  converte BGR/RGB explicitamente, ficando acoplado ao padrão OpenCV.
- Limitações: não cobre metadados, normalização de dtype, carregamento com
  conversão de espaço de cor ou integração com cache remoto além de URL direta.
- Conteúdos ausentes relacionados: fundamentos de leitura, representação,
  canais, dtype e conversões didáticas.
- Área temática: fundamentos de imagens digitais; entrada e saída.
- Maturidade: `Parcial`.
- Evidências no código: `ImageLoader` em
  `src/dip_toolkit/modules/image_loader.py:14`, métodos em
  `src/dip_toolkit/modules/image_loader.py:27`,
  `src/dip_toolkit/modules/image_loader.py:43`,
  `src/dip_toolkit/modules/image_loader.py:65` e
  `src/dip_toolkit/modules/image_loader.py:91`; testes em
  `tests/test_image_loader.py:17`, `tests/test_image_loader.py:28`,
  `tests/test_image_loader.py:37` e `tests/test_image_loader.py:42`.
  A maturidade foi mantida como `Parcial`, não `Funcional`, porque métodos
  públicos como `load_images`, `load_images_from_directory(..., recursive=True)`,
  `skip_invalid` e `load_image_from_url` ainda não têm testes diretos.

### `src/dip_toolkit/modules/image_operator.py`

- Responsabilidade aparente: operações aritméticas e bitwise entre imagens.
- Classes públicas: `ImageOperator`.
- Funções e métodos públicos:
  `add_images`, `subtract_images`, `multiply_images`, `divide_images` e
  `bitwise_multiply_images`.
- Funcionalidades implementadas: encapsula `cv2.add`, `cv2.subtract`,
  `cv2.multiply`, `cv2.divide` e `cv2.bitwise_and`.
- Principais dependências: OpenCV.
- Testes existentes: nenhum.
- Notebooks existentes: nenhum.
- Problemas técnicos observados: ausência de type hints; classe sem docstring;
  não valida shapes, dtype, canais, divisão por zero ou semântica de saturação;
  `bitwise_multiply_images` usa `bitwise_and`, que não é multiplicação
  aritmética.
- Limitações: não cobre operações com escalares, normalização, blend ponderado,
  diferença absoluta ou operações lógicas completas.
- Conteúdos ausentes relacionados: aritmética de imagens, operações lógicas,
  máscaras e saturação/overflow de forma didática.
- Área temática: fundamentos e operações básicas de intensidade.
- Maturidade: `Parcial`.
- Evidências no código: `ImageOperator` em
  `src/dip_toolkit/modules/image_operator.py:4` e métodos em
  `src/dip_toolkit/modules/image_operator.py:8` a
  `src/dip_toolkit/modules/image_operator.py:32`.

### `src/dip_toolkit/modules/image_preprocessor.py`

- Responsabilidade aparente: pré-processamento simples com redimensionamento e
  filtros.
- Classes públicas: `ImagePreprocessor`.
- Funções e métodos públicos:
  `resize_image(image, width, height)` e
  `apply_filter(image, filter_type="blur", **kwargs)`.
- Funcionalidades implementadas: redimensiona com `cv.resize`, aplica
  `cv.GaussianBlur` para `"blur"` e `cv.medianBlur` para `"median"`.
- Principais dependências: OpenCV.
- Testes existentes: nenhum.
- Notebooks existentes: nenhum.
- Problemas técnicos observados: métodos sem docstring; ausência de type hints;
  uso de `**kwargs`; não valida `width`, `height`, `ksize`, imagem vazia ou
  dtype; para blur gaussiano, `ksize` par não é ajustado antes de chamar
  `cv.GaussianBlur`.
- Limitações: cobre apenas dois filtros de suavização; não há filtros lineares
  genéricos, kernels explícitos, bordas, sharpening ou documentação didática.
- Conteúdos ausentes relacionados: convolução, correlação, filtros de média,
  mediana, Laplaciano, Sobel, gradiente, realce e ruído.
- Área temática: filtragem espacial.
- Maturidade: `Inicial`.
- Evidências no código: `ImagePreprocessor` em
  `src/dip_toolkit/modules/image_preprocessor.py:4` e `apply_filter` em
  `src/dip_toolkit/modules/image_preprocessor.py:11`.

### `src/dip_toolkit/modules/image_transformer.py`

- Responsabilidade aparente: transformações geométricas básicas.
- Classes públicas: `ImageTransformer`.
- Funções e métodos públicos:
  `translate`, `rotate`, `mirror`, `stretch`, `crop` e `resize`.
- Funcionalidades implementadas: translação e rotação com `cv.warpAffine`,
  espelhamento com `cv.flip`, escala e resize com `cv.resize`, recorte por
  slicing NumPy.
- Principais dependências: OpenCV e NumPy.
- Testes existentes: nenhum.
- Notebooks existentes: nenhum.
- Problemas técnicos observados: ausência de type hints; classe sem docstring;
  não valida imagem, coordenadas, dimensões, interpolação, borda, escala ou
  recorte fora dos limites; `crop` aceita índices negativos por semântica NumPy.
- Limitações: não cobre amostragem/re-amostragem didática, interpolação
  configurável, transformações afins completas, perspectiva ou quantificação de
  efeitos.
- Conteúdos ausentes relacionados: amostragem, quantização espacial,
  interpolação nearest/bilinear/bicúbica e aliasing.
- Área temática: representação, amostragem e transformações geométricas.
- Maturidade: `Parcial`.
- Evidências no código: `ImageTransformer` em
  `src/dip_toolkit/modules/image_transformer.py:5` e métodos em
  `src/dip_toolkit/modules/image_transformer.py:9` a
  `src/dip_toolkit/modules/image_transformer.py:42`.

### `src/dip_toolkit/modules/model_trainer.py`

- Responsabilidade aparente: treinar um modelo Random Forest genérico.
- Classes públicas: `ModelTrainer`.
- Funções e métodos públicos: `train_model(X, y)`.
- Funcionalidades implementadas: divide dados em treino/teste, treina
  `RandomForestClassifier` e retorna score no conjunto de teste.
- Principais dependências: scikit-learn.
- Testes existentes: nenhum.
- Notebooks existentes: nenhum.
- Problemas técnicos observados: ausência de type hints; classe e método sem
  docstring; sem validação de entrada; sem `random_state`; divisão aleatória não
  reproduzível; escopo genérico, não específico de imagem.
- Limitações: não expõe avaliação detalhada, seleção de features, pipeline ou
  persistência; duplica parcialmente a intenção de `ImageClassifier`.
- Conteúdos ausentes relacionados: treino de classificadores com atributos de
  imagem e validação didática.
- Área temática: extensão de aprendizado de máquina.
- Maturidade: `Inicial`.
- Evidências no código: `ModelTrainer` em
  `src/dip_toolkit/modules/model_trainer.py:5` e `train_model` em
  `src/dip_toolkit/modules/model_trainer.py:9`.

### `src/dip_toolkit/modules/statistical_tools.py`

- Responsabilidade aparente: estatísticas de imagem e estatísticas genéricas.
- Classes públicas: `StatisticalTools`.
- Funções e métodos públicos: `get_image_info`, `print_image_info`,
  `compute_image_similarity`, `calculate_mean`, `calculate_std`,
  `calculate_variance`, `calculate_r2`, `calculate_adjusted_r2`,
  `calculate_correlation`, `perform_pca`, `calculate_moving_average`,
  `calculate_exponential_moving_average`, `calculate_autocorrelation` e
  `benjamini_hochberg`.
- Funcionalidades implementadas: metadados básicos de imagem, média, desvio,
  variância, correlação, R2, R2 ajustado, PCA, médias móveis, autocorrelação e
  correção Benjamini-Hochberg.
- Principais dependências: OpenCV, NumPy, pandas, SciPy e scikit-learn.
- Testes existentes: nenhum.
- Notebooks existentes: nenhum.
- Problemas técnicos observados: ausência de type hints; classe sem docstring;
  mistura responsabilidades de imagem, regressão, séries temporais, PCA e teste
  estatístico; `benjamini_hochberg` está dentro da classe mas não recebe `self`
  nem é marcado como `@staticmethod`, portanto uma chamada por instância passa o
  objeto como primeiro argumento; `compute_image_similarity` não trata arrays
  constantes, que podem gerar correlação indefinida; várias funções não validam
  tamanho, dtype, NaN, divisão por zero ou parâmetros.
- Limitações: estatísticas de imagem são básicas; as ferramentas genéricas não
  pertencem diretamente ao núcleo clássico de PDI.
- Conteúdos ausentes relacionados: estatísticas de histograma, momentos de
  regiões, descritores de textura, métricas de qualidade e ruído.
- Área temática: fundamentos de imagens digitais; extensão estatística.
- Maturidade: `Necessita revisão`.
- Evidências no código: `StatisticalTools` em
  `src/dip_toolkit/modules/statistical_tools.py:8`,
  `get_image_info` em `src/dip_toolkit/modules/statistical_tools.py:12`,
  `compute_image_similarity` em
  `src/dip_toolkit/modules/statistical_tools.py:88` e
  `benjamini_hochberg` em `src/dip_toolkit/modules/statistical_tools.py:189`.

### `src/dip_toolkit/modules/visualization.py`

- Responsabilidade aparente: visualização simples de resultados.
- Classes públicas: `Visualization`.
- Funções e métodos públicos: `plot_histogram(hist)`.
- Funcionalidades implementadas: cria figura, plota histograma e chama
  `plt.show()`.
- Principais dependências: Matplotlib.
- Testes existentes: nenhum.
- Notebooks existentes: nenhum.
- Problemas técnicos observados: classe e método sem docstring; ausência de
  type hints; não retorna figura/eixo para customização ou teste; acoplado a
  `plt.show()`.
- Limitações: cobre apenas histograma simples.
- Conteúdos ausentes relacionados: visualização de imagens, grades,
  comparação antes/depois, espectros, kernels, canais, máscaras e contornos.
- Área temática: suporte didático e visualização.
- Maturidade: `Inicial`.
- Evidências no código: `Visualization` em
  `src/dip_toolkit/modules/visualization.py:4` e `plot_histogram` em
  `src/dip_toolkit/modules/visualization.py:8`.

### `src/dip_toolkit/modules/wavelet_transformer.py`

- Responsabilidade aparente: decomposição, reconstrução, visualização e
  limiarização por wavelets.
- Classes públicas: `WaveletTransformer`.
- Funções e métodos públicos:
  `wavelet_decompose`, `wavelet_reconstruct`, `visualize_coeffs` e
  `threshold_coeffs`.
- Funcionalidades implementadas: usa `pywt.wavedec2`, `pywt.waverec2`,
  `pywt.threshold` e visualiza coeficientes de um nível.
- Principais dependências: PyWavelets e Matplotlib.
- Testes existentes: nenhum.
- Notebooks existentes: nenhum.
- Problemas técnicos observados: ausência de type hints; classe sem docstring;
  não valida wavelet, nível, forma da imagem ou coeficientes; `visualize_coeffs`
  assume exatamente um nível (`approximation, (horizontal, vertical, diagonal) =
  coeffs`), mas `wavelet_decompose(..., level>1)` retorna múltiplos níveis de
  detalhe.
- Limitações: não cobre pirâmides, subamostragem, compressão por wavelet,
  denoising completo, comparação de bases ou tratamento de bordas.
- Conteúdos ausentes relacionados: multirresolução, compressão, denoising e
  análise de detalhes por nível.
- Área temática: wavelets e multirresolução.
- Maturidade: `Parcial`.
- Evidências no código: `WaveletTransformer` em
  `src/dip_toolkit/modules/wavelet_transformer.py:5`,
  `wavelet_decompose` em
  `src/dip_toolkit/modules/wavelet_transformer.py:9` e `visualize_coeffs` em
  `src/dip_toolkit/modules/wavelet_transformer.py:23`.

### `src/dip_toolkit/modules/webcam_stream_reader.py`

- Responsabilidade aparente: capturar frames de webcam.
- Classes públicas: `WebcamStreamReader`.
- Funções e métodos públicos: `open`, `read_frame`, `show_stream` e `close`.
- Funcionalidades implementadas: abre `cv2.VideoCapture`, configura largura e
  altura, lê frame, exibe stream com `cv2.imshow` e libera recursos.
- Principais dependências: OpenCV.
- Testes existentes: nenhum.
- Notebooks existentes: nenhum.
- Problemas técnicos observados: ausência de type hints; classe e métodos sem
  docstrings; usa `cv2.imshow`, `cv2.waitKey` e `cv2.destroyAllWindows`, o que
  não é adequado para notebooks Colab; não implementa contexto (`with`), timeout
  ou fallback; `close` chama `cv2.destroyAllWindows()` mesmo sem janela.
- Limitações: extensão local dependente de hardware e interface gráfica; não
  pertence ao núcleo clássico de Gonzalez e Woods.
- Conteúdos ausentes relacionados: captura compatível com Colab, leitura de
  vídeo, fontes simuladas e processamento frame a frame.
- Área temática: extensão de captura por webcam.
- Maturidade: `Necessita revisão`.
- Evidências no código: `WebcamStreamReader` em
  `src/dip_toolkit/modules/webcam_stream_reader.py:4`,
  `show_stream` em `src/dip_toolkit/modules/webcam_stream_reader.py:26` e
  `close` em `src/dip_toolkit/modules/webcam_stream_reader.py:37`.

### `src/dip_toolkit/assets.py`

Embora não esteja em `src/dip_toolkit/modules/`, este arquivo foi analisado por
ser parte obrigatória do fluxo didático.

- Responsabilidade aparente: localizar e baixar imagens públicas da disciplina.
- Funções públicas: `download_course_image(filename, output_dir=DEFAULT_IMAGES_DIR, *, force=False) -> Path`.
- Funcionalidades implementadas: cria diretório de saída, reutiliza cache local,
  consulta `IMAGE_RESOLVER_URL` por nome, baixa com `gdown.download` e retorna
  o caminho local.
- Principais dependências: `pathlib`, `requests` e `gdown`.
- Testes existentes: `tests/test_assets.py` cobre cache local, download mockado
  e imagem desconhecida.
- Notebooks existentes: `notebooks/00_validacao_colab.ipynb` usa
  `download_course_image("cameraman_original.png", output_dir="/content/dip_images")`.
- Problemas técnicos observados: não há validação explícita de nome de arquivo;
  depende de um endpoint externo fixo; `scripts/validate_first_flow.py` importa
  `find_image`, mas `assets.py` não define essa função.
- Limitações: fluxo cobre download por nome, mas não inventário público de
  imagens, validação de extensões ou mensagens didáticas para falhas de rede.
- Evidências no código: `IMAGE_RESOLVER_URL` em
  `src/dip_toolkit/assets.py:8`, `DEFAULT_IMAGES_DIR` em
  `src/dip_toolkit/assets.py:10`, `download_course_image` em
  `src/dip_toolkit/assets.py:13`; testes em `tests/test_assets.py:9`,
  `tests/test_assets.py:38` e `tests/test_assets.py:77`.

## 4. Matriz de cobertura Gonzalez e Woods

| Área temática | Status | Módulos relacionados | Funcionalidades existentes | Testes existentes | Notebook existente | Lacunas principais | Prioridade recomendada |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fundamentos de imagens digitais | Cobertura inicial | `assets.py`, `image_loader.py`, `statistical_tools.py`, `visualization.py` | Download por nome, carga local/diretório/URL, metadados básicos e histograma plotado | `tests/test_assets.py`, parte de `tests/test_image_loader.py` | `00_validacao_colab.ipynb` | Representação didática de canais, dtype, normalização, visualização padronizada e validações amplas | Alta |
| Representação, amostragem e quantização | Cobertura inicial | `image_creator.py`, `image_transformer.py`, `statistical_tools.py` | Criação de arrays, resize, escala, rotação e recorte | Nenhum teste direto | Nenhum | Amostragem, reamostragem, interpolação, aliasing e quantização controlada | Alta |
| Transformações de intensidade | Não coberto | `image_operator.py` apenas como base indireta | Operações aritméticas simples | Nenhum | Nenhum | Negativo, log, potência/gama, fatiamento de bits, clipping e normalização | Alta |
| Histogramas e melhoria de contraste | Cobertura inicial | `image_analysis.py`, `visualization.py`, `image_color_processor.py` | Histograma de canal 0 e histogramas por canal colorido | Nenhum | Nenhum | Equalização, matching, CDF, histograma normalizado, CLAHE e exemplos didáticos | Alta |
| Filtragem espacial | Cobertura inicial | `image_preprocessor.py`, `feature_extractor.py` | Gaussian blur, mediana e Canny com parâmetros fixos | Nenhum | Nenhum | Convolução, média, kernels, Sobel, Laplaciano, sharpening, bordas e padding | Alta |
| Transformadas e filtragem no domínio da frequência | Cobertura inicial | `fourier_transformer.py` | DFT, IDFT, espectro, máscaras e notch | Nenhum | Nenhum | Validações, filtros corretos, padding, convolução, restauração e notebook | Média |
| Restauração e reconstrução | Não coberto | `image_creator.py`, `fourier_transformer.py`, `wavelet_transformer.py` como bases indiretas | Não há função de restauração; existem apenas ruído sintético e transformadas inversas isoladas | Nenhum | Nenhum | Modelos de ruído, filtros inverso/Wiener, média geométrica, mediana adaptativa, motion blur | Média |
| Processamento de imagens coloridas | Cobertura inicial | `image_color_processor.py` | Conversões BGR para HSV/YCrCb/Lab/cinza, CMYK, histogramas e cubo 3D | Nenhum | Nenhum | API RGB/BGR consistente, validações, operações por canal, pseudo-cor e segmentação por cor | Média |
| Wavelets e multirresolução | Cobertura inicial | `wavelet_transformer.py` | DWT, IDWT, limiarização e visualização de um nível | Nenhum | Nenhum | Multinível robusto, pirâmides, denoising, compressão e notebook | Baixa |
| Compressão de imagens | Não coberto | `wavelet_transformer.py` como base indireta | Nenhuma compressão implementada | Nenhum | Nenhum | Codificação, taxa, qualidade, JPEG conceitual, quantização e compressão por wavelet | Baixa |
| Morfologia matemática | Não coberto | Nenhum módulo dedicado | Nenhuma | Nenhum | Nenhum | Erosão, dilatação, abertura, fechamento, gradiente, top-hat, hit-or-miss e esqueletização | Média |
| Segmentação | Não coberto | `feature_extractor.py`, `image_color_processor.py` como bases indiretas | Não há função de segmentação; existem apenas Canny fixo e conversões de cor que poderiam apoiar trabalhos futuros | Nenhum | Nenhum | Limiarização, Otsu, crescimento de região, watershed, bordas, segmentação por cor | Média |
| Representação de fronteiras e regiões | Não coberto | `feature_extractor.py` como base indireta | Bordas Canny sem representação | Nenhum | Nenhum | Contornos, chain codes, assinaturas, aproximação poligonal, esqueletos, descritores regionais | Baixa |
| Descritores e características | Não coberto | `feature_extractor.py`, `statistical_tools.py` como bases indiretas | Não há descritores de imagem implementados; existem Canny, PCA genérico e estatísticas simples | Nenhum | Nenhum | Momentos, textura, forma, cantos, blobs, HOG/LBP e descritores invariantes | Baixa |
| Reconhecimento e classificação | Cobertura inicial | `image_classifier.py`, `model_trainer.py`, `generate_data.py` | Classificadores scikit-learn, Random Forest e dados meia-lua; não há pipeline específico de reconhecimento de imagens | Nenhum | Nenhum | Pipeline de imagem, extração de atributos, validação reproduzível, métricas e notebook | Baixa |

## 5. Funcionalidades complementares

- Aprendizado de máquina: `image_classifier.py` e `model_trainer.py`.
  Recomendação: adiar como extensão até que extração de características e
  datasets didáticos estejam definidos; revisar aleatoriedade, API e métricas.
- Treinamento de modelos: `model_trainer.py`.
  Recomendação: reorganizar ou fundir conceitualmente com uma fila de extensão,
  pois hoje é genérico e não específico de imagens.
- Análise estatística: `statistical_tools.py`.
  Recomendação: manter apenas o que apoia fundamentos de imagem no núcleo e
  reorganizar regressão, séries temporais, PCA e Benjamini-Hochberg como
  extensão ou remover do caminho didático principal.
- Webcam: `webcam_stream_reader.py`.
  Recomendação: adiar e revisar, pois depende de interface gráfica local e usa
  `cv2.imshow`, incompatível com a regra de notebooks Colab.
- Geração de dados: `generate_data.py` e parte de `image_creator.py`.
  Recomendação: manter quando gerar imagens ou máscaras didáticas; separar a
  geração de pontos 2D como extensão.
- Visualização: `visualization.py` e métodos `visualize_*` em outros módulos.
  Recomendação: manter, mas padronizar para retornar figuras/eixos e funcionar
  bem em notebooks.

## 6. Problemas transversais

Problemas observados em vários módulos:

- ausência de type hints em métodos públicos fora de `ImageLoader`;
- classes públicas sem docstrings na maioria dos arquivos;
- validações insuficientes de arrays, shapes, canais, dtypes e faixas;
- uso de `**kwargs` em `image_creator.py` e `image_preprocessor.py`;
- inconsistência RGB/BGR em `image_color_processor.py` e no fluxo que usa
  OpenCV;
- aleatoriedade não reproduzível em `image_creator.py`, `generate_data.py`,
  `image_classifier.py` e `model_trainer.py`;
- dependência de interface gráfica em `webcam_stream_reader.py`;
- falta de testes para todos os módulos exceto `image_loader.py` e `assets.py`;
- falta de notebooks didáticos para todos os módulos exceto o fluxo inicial de
  carga no Colab;
- responsabilidades misturadas em `statistical_tools.py`;
- duplicação ou sobreposição entre `image_classifier.py` e `model_trainer.py`;
- funções de visualização acopladas a `plt.show()`, o que dificulta testes e
  composição didática;
- inconsistência de integração entre `scripts/validate_first_flow.py` e
  `src/dip_toolkit/assets.py` por uso de `find_image` inexistente.

## 7. Conclusão

O nível atual do toolkit é de fundação inicial com um bloco parcialmente
validado de download/carregamento. O principal bloco já aproveitável é o fluxo
`download_course_image` + `ImageLoader` + notebook de validação no Colab.

Blocos que precisam de revisão: Fourier, processamento de cor, estatística,
webcam e os módulos de aprendizado de máquina. Blocos que precisam ser
criados ou desenvolvidos do início: transformações de intensidade, melhoria de
contraste, filtragem espacial didática completa, restauração, compressão,
morfologia, segmentação e representação/descrição.

O próximo foco deve ser consolidar a fundação técnica: revisar `ImageLoader`
e `ImageCreator` nas issues DIP-02 e DIP-03 já existentes, padronizar API,
type hints, docstrings, validações, testes e notebooks, e então avançar em
ordem didática para representação, intensidade, histogramas e filtragem
espacial.
