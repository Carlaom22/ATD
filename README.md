# Identificação de Dígitos através de Características Extraídas de Sinais de Áudio

## Introdução

Este projeto tem como objetivo identificar dígitos (0 a 9) através da análise de sinais de áudio. Utilizam-se técnicas de análise no domínio do tempo e da frequência para extrair características dos sinais que permitam discriminar os diferentes dígitos.

O projeto foi desenvolvido no âmbito da disciplina de Análise e Transformação de Dados do curso de Engenharia Informática no ano letivo 2023/2024.

## Instalação

### Pré-requisitos

- Python 3
- Bibliotecas necessárias:
  - `numpy`
  - `matplotlib`
  - `librosa`
  - `scipy`

### Download dos Dados

Os dados utilizados no projeto podem ser baixados do Kaggle através do seguinte link:

[AudioMNIST Dataset](https://www.kaggle.com/datasets/sripaadsrinivasan/audio-mnist)

Descompacte o arquivo baixado e considere apenas os sinais em bruto disponíveis na pasta `data`.

### Configuração

Clone o repositório do projeto e instale as dependências:

```bash
git clone https://github.com/Carlaom22/ATD.git
cd AudioDigitRecognition
pip install -r requirements.txt
```

## Uso

### Importação e Visualização dos Sinais

O script `mile1.py` importa os sinais de áudio, reproduz e representa graficamente um exemplo dos sinais importados, e extrai características temporais como energia e amplitude.

Para executar o script, utilize o seguinte comando:

```bash
python mile1.py
```

Certifique-se de ajustar o caminho do diretório `data_dir` no script para o local onde os dados de áudio estão armazenados.

### Análise no Domínio da Frequência

O script `mile2.py` calcula o espectro de amplitude mediano dos sinais de áudio e plota o espectro.

Para executar o script, utilize o seguinte comando:

```bash
python mile2.py
```

Certifique-se de ajustar o caminho do diretório `data_dir` no script para o local onde os dados de áudio estão armazenados.

### Extração de Características Espectrais e Plotagem

O script `mile3.py` calcula várias características espectrais dos sinais de áudio e plota gráficos de boxplot para cada característica, além de um gráfico de dispersão 3D.

Para executar o script, utilize o seguinte comando:

```bash
python mile3.py
```

Certifique-se de ajustar o caminho do diretório `directory` no script para o local onde os dados de áudio estão armazenados.

### Definição de Regras de Decisão e Avaliação

O script `mile4.py` define regras de decisão para classificar os dígitos com base nas características extraídas e calcula a precisão da classificação. Também gera gráficos de boxplot para as características e um gráfico de dispersão 3D.

Para executar o script, utilize o seguinte comando:

```bash
python mile4.py
```

Certifique-se de ajustar o caminho do diretório `directory` no script para o local onde os dados de áudio estão armazenados.

## Contribuição

Apenas e eu. Mas sinta-se à vontade.

## Créditos

Este projeto foi desenvolvido por Carlos Soares.

Referências:

- [AudioMNIST Dataset](https://github.com/soerenab/AudioMNIST)
- [Artigo relacionado no arXiv](https://arxiv.org/abs/1807.03418)
