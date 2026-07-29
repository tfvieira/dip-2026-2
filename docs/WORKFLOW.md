# Fluxo de desenvolvimento

Este documento descreve o processo utilizado para desenvolver, testar e revisar as funcionalidades do DIP Toolkit.

## Repositório

```text
https://github.com/tfvieira/dip-2026-2
```

## Fluxo resumido

```text
Issue
  ↓
Branch da issue
  ↓
Desenvolvimento
  ↓
Testes e formatação
  ↓
Pull request
  ↓
Revisão
  ↓
Correções
  ↓
Merge na main
```

Não utilizamos uma branch `develop`.

Todas as branches partem da `main` e retornam para a `main` por pull request.

## 1. Preparar o ambiente

Clone o repositório:

```bash
git clone https://github.com/tfvieira/dip-2026-2.git
cd dip-2026-2
```

Crie um ambiente virtual:

```bash
python -m venv .venv
```

Ative no Windows PowerShell:

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

## 2. Escolher uma issue

Antes de iniciar uma implementação:

1. escolha ou receba uma issue;
2. leia todo o contexto;
3. confira o escopo;
4. confira os critérios de aceite;
5. confirme que ninguém já está trabalhando nela;
6. atribua a issue ao responsável.

Nenhuma implementação deve começar sem uma issue associada.

## 3. Criar uma branch

Atualize a `main`:

```bash
git checkout main
git pull origin main
```

Crie uma branch:

```bash
git checkout -b feat/02-image-loader
```

Padrões:

```text
feat/<numero>-<descricao>
fix/<numero>-<descricao>
test/<numero>-<descricao>
docs/<numero>-<descricao>
notebook/<numero>-<descricao>
```

Exemplos:

```text
feat/02-image-loader
fix/05-fourier-filter
test/03-image-creator
docs/01-modules-map
```

## 4. Desenvolver

Cada entrega deve incluir, quando aplicável:

- implementação;
- validações;
- type hints;
- docstrings;
- testes;
- notebook;
- exemplos didáticos;
- atualização da documentação.

Não devem ser adicionados ao Git:

- imagens da disciplina;
- slides;
- ambientes virtuais;
- arquivos temporários;
- resultados gerados localmente;
- credenciais;
- tokens.

## 5. Validar localmente

Formate:

```bash
ruff format .
```

Verifique:

```bash
ruff check .
```

Teste:

```bash
pytest
```

A pull request só deve ser aberta quando os três comandos estiverem passando.

## 6. Criar commits

Utilize mensagens curtas e objetivas.

Exemplos:

```text
feat: add image loading modes
fix: validate unsupported image format
test: cover invalid directory loading
docs: document ImageLoader usage
```

Evite mensagens genéricas:

```text
update
changes
fix code
final
```

## 7. Enviar a branch

```bash
git push -u origin feat/02-image-loader
```

## 8. Abrir a pull request

A pull request deve:

- apontar para a `main`;
- referenciar a issue;
- explicar o que foi implementado;
- informar como a alteração foi testada;
- incluir imagens dos resultados quando relevante;
- manter o CI aprovado.

Utilize no corpo:

```text
Closes #NUMERO_DA_ISSUE
```

## 9. Revisão

O revisor deve verificar:

- atendimento aos critérios da issue;
- clareza da API;
- legibilidade;
- validações;
- testes;
- notebook;
- compatibilidade com o Colab;
- impacto nos demais módulos.

O autor deve aplicar as correções solicitadas.

Os integrantes podem revisar pull requests uns dos outros.

Dúvidas técnicas ou decisões de arquitetura devem ser discutidas com Edvar.

## 10. Merge

Após a aprovação:

1. realize o merge na `main`;
2. confirme o fechamento da issue;
3. mova a atividade para concluída no Project;
4. exclua a branch remota, quando não for mais necessária.

## Critério geral de conclusão

Uma issue está concluída quando:

- todos os critérios de aceite foram atendidos;
- o código está formatado;
- o Ruff está passando;
- os testes estão passando;
- o CI está aprovado;
- o notebook foi validado;
- a documentação foi atualizada quando necessário.
