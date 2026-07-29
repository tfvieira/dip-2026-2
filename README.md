# DIP Toolkit — Processamento Digital de Imagens

Repositório para os módulos didáticos da disciplina
de Processamento Digital de Imagens.

## Estrutura

```text
.
├── src/dip_toolkit/          # Biblioteca Python
│   ├── assets.py             # Download/localização das imagens públicas
│   └── modules/              # Módulos recebidos e evolução incremental
├── notebooks/                # Notebooks didáticos e de validação
├── tests/                    # Testes automatizados
├── scripts/                  # Scripts auxiliares
├── docs/                     # Processo, roadmap e decisões
└── .github/                  # Templates e integração contínua
```

## Instalação local

Recomendação: Python 3.10 ou superior.

```bash
python -m venv .venv
```

Ative o ambiente no Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Ou no Linux/macOS:

```bash
source .venv/bin/activate
```

Instale o projeto com as dependências de desenvolvimento:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Teste local do primeiro fluxo

Baixe a pasta pública de imagens:

```bash
python scripts/download_course_image.py
```

Valide o carregamento de `cameraman_original.png`:

```bash
python scripts/validate_first_flow.py
```

Execute os testes:

```bash
pytest
```

Execute a análise estática:

```bash
ruff check .
```

## Fluxo de contribuição

1. Escolha ou receba uma issue.
2. Crie uma branch a partir de `main`.
3. Implemente código, testes e exemplo didático.
4. Abra uma pull request.
5. Resolva os comentários da revisão.
6. Faça o merge apenas após aprovação e CI verde.

Consulte [CONTRIBUTING.md](docs/CONTRIBUTING.md).

## Situação dos módulos

Os módulos recebidos foram preservados como ponto de partida. O `ImageLoader` já
recebeu uma primeira padronização para validar a infraestrutura. Os demais serão
revisados progressivamente por issues específicas.
