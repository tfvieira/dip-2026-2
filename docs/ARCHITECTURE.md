# Arquitetura e funcionamento do DIP Toolkit

Este documento explica como o DIP Toolkit funciona e como seus componentes se relacionam.

## Visão geral

O projeto possui três fontes principais:

```text
GitHub
  ├── Biblioteca Python
  ├── Testes
  ├── Notebooks
  └── Documentação

Google Drive
  └── Imagens públicas da disciplina

Google Colab
  └── Ambiente temporário de execução
```

## Biblioteca Python

A biblioteca está localizada em:

```text
src/dip_toolkit/
```

Os módulos de Processamento Digital de Imagens estão em:

```text
src/dip_toolkit/modules/
```

Exemplos:

```text
image_loader.py
image_creator.py
image_transformer.py
fourier_transformer.py
statistical_tools.py
```

Cada módulo deve possuir:

- responsabilidade clara;
- API simples;
- type hints;
- docstrings;
- validações;
- testes;
- exemplos didáticos.

## Instalação pelo GitHub

No Google Colab, a biblioteca é instalada com:

```python
%pip install -q "git+https://github.com/tfvieira/dip-2026-2.git"
```

O `pip` realiza este processo:

```text
GitHub
  ↓
Download temporário do repositório
  ↓
Leitura do pyproject.toml
  ↓
Construção do pacote
  ↓
Instalação no ambiente Python
  ↓
Remoção dos arquivos temporários de instalação
```

Por isso, normalmente não aparece uma pasta chamada `dip-2026-2` no diretório principal do Colab.

O pacote fica instalado nos diretórios internos do Python.

É possível verificar com:

```python
import dip_toolkit

print(dip_toolkit.__file__)
```

O resultado será semelhante a:

```text
/usr/local/lib/python3.12/dist-packages/dip_toolkit/__init__.py
```

## Importação dos módulos

Depois da instalação, o Python consegue localizar o pacote:

```python
from dip_toolkit.modules.image_loader import ImageLoader
```

O Python procura o pacote nos diretórios de bibliotecas instaladas e carrega o módulo solicitado.

## Armazenamento das imagens

As imagens não ficam no GitHub.

Elas permanecem em uma pasta pública do Google Drive.

Isso mantém o repositório:

- leve;
- focado em código;
- mais rápido para clonar;
- mais simples de manter;
- adequado para publicação.

## Busca de imagens por nome

O usuário solicita uma imagem:

```python
download_course_image("cameraman_original.png")
```

O processo é:

```text
Notebook
  ↓
Envia o nome da imagem ao serviço de resolução
  ↓
O serviço consulta a pasta pública do Google Drive
  ↓
O serviço encontra o arquivo pelo nome
  ↓
O serviço retorna o ID do arquivo
  ↓
O gdown baixa somente o arquivo solicitado
  ↓
A imagem é salva no ambiente atual
  ↓
A função retorna o caminho local
```

O serviço de resolução foi criado com Google Apps Script.

Ele recebe o nome do arquivo:

```text
?name=cameraman_original.png
```

E retorna:

```json
{
  "ok": true,
  "id": "ID_DO_ARQUIVO",
  "name": "cameraman_original.png",
  "mimeType": "image/png"
}
```

O aluno não precisa conhecer o ID do arquivo.

## Cache das imagens

Quando a função é chamada pela primeira vez:

```python
download_course_image("cameraman_original.png")
```

a imagem é baixada.

Quando a função é chamada novamente na mesma sessão, o arquivo local é reutilizado.

Para baixar novamente:

```python
download_course_image(
    "cameraman_original.png",
    force=True,
)
```

## Uso com o ImageLoader

O download e o carregamento possuem responsabilidades diferentes.

### `download_course_image`

Responsável por:

- localizar a imagem no Google Drive;
- baixar a imagem;
- devolver o caminho local.

### `ImageLoader`

Responsável por:

- abrir a imagem local;
- validar o arquivo;
- converter os dados para um array NumPy;
- devolver a imagem para processamento.

Exemplo:

```python
from dip_toolkit import download_course_image
from dip_toolkit.modules.image_loader import ImageLoader

image_path = download_course_image("cameraman_original.png")

image = ImageLoader().load_image(image_path)
```

## Ambiente temporário do Colab

O Google Colab utiliza uma máquina virtual temporária.

Durante a sessão ficam disponíveis:

- biblioteca instalada;
- imagens baixadas;
- variáveis;
- arquivos gerados;
- resultados intermediários.

Quando a sessão termina ou expira, esses dados são removidos.

O notebook continua salvo, mas o ambiente precisa ser preparado novamente.

Por isso, os notebooks devem conter:

1. instalação da biblioteca;
2. importação dos módulos;
3. download das imagens;
4. execução dos exemplos.

## Fluxo dos alunos

O fluxo esperado será:

```text
Aluno abre o notebook
  ↓
Salva uma cópia no próprio Drive
  ↓
Executa a célula de instalação
  ↓
Executa a célula de download das imagens
  ↓
Executa os exemplos
  ↓
Realiza as atividades
```

Os alunos não precisam:

- clonar o repositório manualmente;
- configurar Python localmente;
- instalar OpenCV no computador;
- montar o Google Drive pessoal;
- baixar todas as imagens da disciplina;
- conhecer os IDs dos arquivos.

## Separação de responsabilidades

```text
GitHub
Código, notebooks, testes e documentação

Google Drive
Imagens e materiais da disciplina

Google Apps Script
Busca de imagens pelo nome

Google Colab
Execução temporária dos exemplos

DIP Toolkit
Operações de Processamento Digital de Imagens
```

Essa separação evita duplicação de arquivos e facilita a manutenção do projeto.
