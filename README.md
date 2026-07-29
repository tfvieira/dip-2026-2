# DIP Toolkit — Processamento Digital de Imagens

Biblioteca didática desenvolvida para apoiar a disciplina de Processamento Digital de Imagens.

O projeto reúne módulos Python, testes automatizados e notebooks compatíveis com o Google Colab.

## Objetivo

O DIP Toolkit tem como objetivo oferecer uma biblioteca simples e reutilizável para:

- carregar e criar imagens;
- aplicar operações de processamento digital de imagens;
- demonstrar os conteúdos da disciplina;
- facilitar a execução dos exemplos no Google Colab;
- evitar configurações complexas nas máquinas dos alunos;
- servir como material de consulta durante e após a disciplina.

## Repositório

```text
https://github.com/tfvieira/dip-2026-2
```

## Estrutura

```text
.
├── src/dip_toolkit/
│   ├── assets.py
│   └── modules/
├── notebooks/
├── tests/
├── scripts/
├── docs/
└── .github/
```

### Diretórios

- `src/dip_toolkit/`: código principal da biblioteca.
- `src/dip_toolkit/assets.py`: localização e download das imagens públicas.
- `src/dip_toolkit/modules/`: módulos de Processamento Digital de Imagens.
- `notebooks/`: exemplos e atividades executáveis no Google Colab.
- `tests/`: testes automatizados.
- `scripts/`: scripts auxiliares.
- `docs/`: documentação do projeto.
- `.github/`: templates e integração contínua.

## Como o projeto funciona

Os códigos da biblioteca ficam no GitHub.

As imagens utilizadas nos exemplos ficam em uma pasta pública do Google Drive e não são armazenadas no repositório.

Quando uma imagem é solicitada:

```python
download_course_image("cameraman_original.png")
```

o sistema:

1. envia o nome da imagem para um serviço de busca;
2. localiza o arquivo na pasta pública do Google Drive;
3. obtém o identificador do arquivo;
4. baixa somente a imagem solicitada;
5. salva a imagem temporariamente no ambiente atual;
6. retorna o caminho local do arquivo.

Isso evita baixar todas as imagens da disciplina em cada notebook.

Mais detalhes estão disponíveis em [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Uso no Google Colab

Instale a biblioteca diretamente do GitHub:

```python
%pip install -q "git+https://github.com/tfvieira/dip-2026-2.git"
```

Importe a função de download:

```python
from dip_toolkit import download_course_image
```

Baixe uma imagem pelo nome:

```python
image_path = download_course_image(
    "cameraman_original.png"
)

print(image_path)
```

Carregue a imagem com o `ImageLoader`:

```python
from dip_toolkit.modules.image_loader import ImageLoader

loader = ImageLoader()
image = loader.load_image(image_path)

print("Formato:", image.shape)
print("Tipo:", image.dtype)
```

Exiba a imagem:

```python
import cv2
import matplotlib.pyplot as plt

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB,
)

plt.figure(figsize=(7, 7))
plt.imshow(image_rgb)
plt.axis("off")
plt.show()
```

## Cache das imagens

As imagens já baixadas são reutilizadas automaticamente.

```python
image_path = download_course_image(
    "cameraman_original.png"
)
```

Para forçar um novo download:

```python
image_path = download_course_image(
    "cameraman_original.png",
    force=True,
)
```

No Google Colab, os arquivos permanecem disponíveis apenas durante a sessão atual.

Quando o ambiente do Colab é encerrado, os pacotes instalados, imagens baixadas e variáveis são apagados.

Por isso, as células de instalação e download devem permanecer nos notebooks.

## Instalação local

Recomendação: Python 3.10 ou superior.

Clone o repositório:

```bash
git clone https://github.com/tfvieira/dip-2026-2.git
cd dip-2026-2
```

Crie um ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente no Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Ative no Linux ou macOS:

```bash
source .venv/bin/activate
```

Instale o projeto:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Exemplo local

```python
from dip_toolkit import download_course_image
from dip_toolkit.modules.image_loader import ImageLoader

image_path = download_course_image(
    "cameraman_original.png"
)

image = ImageLoader().load_image(image_path)

print(image.shape)
print(image.dtype)
```

## Validação do projeto

Formate o código:

```bash
ruff format .
```

Execute a análise estática:

```bash
ruff check .
```

Execute os testes:

```bash
pytest
```

## Documentação

- [Fluxo de desenvolvimento](docs/WORKFLOW.md)
- [Arquitetura e funcionamento](docs/ARCHITECTURE.md)
- [Guia de contribuição](docs/CONTRIBUTING.md)