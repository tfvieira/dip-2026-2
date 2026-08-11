# Histograma e CDF de uma imagem grayscale

## Objetivo

Implementar, usando apenas NumPy, o histograma de uma imagem grayscale
`uint8` e sua função de distribuição acumulada (CDF) normalizada.

Esta atividade prepara a etapa de equalização de histograma. Ao concluí-la,
você deverá conseguir identificar quantos pixels existem em cada intensidade e
como esses valores se acumulam ao longo da faixa de tons de cinza.

## Contrato das funções

Implemente as duas funções no arquivo
`task-01-histogram-cdf.py`:

- `compute_histogram(image)`: recebe uma imagem grayscale 2D com dtype
  `uint8` e devolve um array de 256 posições. A posição `i` deve conter a
  quantidade de pixels com intensidade `i`.
- `compute_normalized_cdf(histogram)`: recebe um histograma unidimensional e
  devolve sua CDF normalizada. Quando o histograma possuir pixels, o último
  valor da CDF deve ser `1.0`. Para um histograma sem contagens, devolva um
  array de zeros com o mesmo shape.

Não utilize OpenCV (`cv2`) nem funções prontas de equalização. Para o
histograma, `np.bincount` é uma alternativa apropriada.

## O que deve ser implementado

As regiões marcadas entre `### START CODE HERE ###` e
`### END CODE HERE ###` indicam exatamente o trecho que deve ser preenchido.

## Como executar

Execute o próprio script:

```bash
python tasks/task-01-histogram-cdf/task-01-histogram-cdf.py
```

O teste usa uma imagem pequena, com valores conhecidos, e verifica as
contagens, a CDF normalizada e o caso de histograma vazio. Ao final, a mensagem
`Test passed!` confirma a solução.
