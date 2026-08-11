# Equalização de histograma em imagem grayscale

## Objetivo

Implementar a equalização de histograma de uma imagem grayscale `uint8` usando
somente NumPy. A transformação deve redistribuir as intensidades a partir do
histograma e da CDF, sem utilizar `cv2.equalizeHist` ou outra função pronta de
equalização.

Esta atividade dá continuidade à Task 01. O algoritmo esperado é:

1. calcular o histograma da imagem;
2. calcular a CDF acumulada;
3. encontrar o primeiro valor não nulo da CDF;
4. construir uma tabela de consulta (LUT) para o intervalo de 0 a 255;
5. aplicar a LUT a todos os pixels.

## Contrato da função

Implemente `equalize_grayscale(image)` no arquivo
`task-02-histogram-equalization.py`.

A função deve:

- aceitar somente arrays NumPy grayscale 2D com dtype `uint8`;
- devolver uma nova imagem com o mesmo shape e dtype;
- não modificar a imagem de entrada;
- preservar imagens constantes;
- calcular a equalização com NumPy.

## O que deve ser implementado

A região entre `### START CODE HERE ###` e `### END CODE HERE ###` contém o
trecho destinado ao aluno. As funções auxiliares já disponíveis no script
podem ser reutilizadas.

## Como executar

```bash
python tasks/task-02-histogram-equalization/task-02-histogram-equalization.py
```

O teste verifica um mapeamento conhecido, a preservação de shape e dtype, a
imutabilidade da entrada e o caso de uma imagem constante. Ao final, a
mensagem `Test passed!` confirma a solução.
