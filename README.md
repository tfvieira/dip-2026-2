# DIP Toolkit — Processamento Digital de Imagens

Biblioteca didática para apoiar o estudo e a prática de Processamento Digital
de Imagens, com exemplos preparados para a disciplina e para consulta
posterior.

## Objetivo

O DIP Toolkit reúne recursos reutilizáveis para explorar, de forma progressiva,
os fundamentos de imagens digitais e as principais operações da área.

Entre os recursos disponíveis estão:

- carregamento e criação de imagens;
- representação, transformação, análise e visualização de imagens;
- notebooks didáticos compatíveis com o Google Colab;
- download sob demanda das imagens utilizadas nos exemplos.

## Notebooks no Google Colab

Os notebooks podem ser executados diretamente no navegador, sem configurar um
ambiente Python local. Comece por
`00_validacao_colab.ipynb` e siga preferencialmente a ordem numérica.

Cada notebook instala o toolkit diretamente deste repositório antes de executar
os exemplos.

**[Acessar os notebooks da disciplina](notebooks/README.md)**

## Uso local opcional

A instalação local não é necessária para quem utilizará o Google Colab. Para
usar o toolkit localmente, é necessário Python 3.10 ou superior:

```bash
git clone https://github.com/tfvieira/dip-2026-2.git
cd dip-2026-2
python -m pip install .
```

Exemplo mínimo:

```python
from dip_toolkit import download_course_image
from dip_toolkit.modules.image_loader import ImageLoader

image_path = download_course_image("cameraman_original.png")
image = ImageLoader().load_image(image_path)

print(image.shape)
print(image.dtype)
```

## Desenvolvimento

```bash
python -m pip install -e ".[dev]"
ruff format --check .
ruff check .
pytest
```

Mais informações:

- [Guia de contribuição](docs/CONTRIBUTING.md)
- [Arquitetura e funcionamento](docs/ARCHITECTURE.md)
