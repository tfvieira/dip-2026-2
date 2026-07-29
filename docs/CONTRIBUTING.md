# Guia de contribuição

## Fluxo de desenvolvimento

1. Escolha ou receba uma issue.
2. Crie uma branch a partir da `main`.
3. Desenvolva e teste a alteração.
4. Envie a branch para o GitHub.
5. Abra um pull request para a `main`.
6. Aguarde a revisão.
7. Corrija os comentários, quando necessário.
8. Após aprovação, realize o merge.

### Nomes de branches

- `feat/<numero>-<descricao>`
- `fix/<numero>-<descricao>`
- `docs/<numero>-<descricao>`
- `test/<numero>-<descricao>`

Exemplo:

`feat/01-image-loader`

## Commits

Adote Conventional Commits:

```text
feat: add grayscale image loading example
fix: validate invalid OpenCV flags
 test: cover missing image error
 docs: document Colab setup
```

## Pull requests

Toda pull request deve:

- referenciar uma issue;
- explicar o problema e a solução;
- incluir testes quando aplicável;
- incluir exemplo didático ou atualização de notebook quando aplicável;
- executar no Colab quando a mudança afetar notebooks;
- evitar imagens, slides e arquivos grandes no Git;
- estar formatada e com a suíte de testes aprovada.

## Critério de aceite de módulos

- API clara e consistente;
- parâmetros principais explícitos;
- type hints;
- docstrings;
- validação das entradas;
- erros compreensíveis;
- testes automatizados;
- exemplo de uso;
- compatibilidade com Colab;
- visualização com Matplotlib, não com `cv2.imshow` nos notebooks.

## Revisão

O autor da implementação deve aplicar as correções solicitadas. O revisor evita
alterar diretamente a branch do autor, exceto em situações previamente alinhadas.
