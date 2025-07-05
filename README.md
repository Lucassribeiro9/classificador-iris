# 🌳 Classificador de Espécies de Íris

Este projeto é um estudo de ponta a ponta que aborda os passos fundamentais para a criação de um modelo de Machine Learning, desde a análise dos dados até a avaliação e salvamento do modelo final. Utilizando o clássico dataset "Iris", o objetivo é construir um sistema capaz de classificar corretamente as espécies de flores Íris com base em suas características físicas.

Este repositório foi construído de forma interativa e didática, servindo como um guia prático para os meus próximos projetos.

## 🎯 Funcionalidades

O notebook `analise-iris.ipynb` cobre o seguinte fluxo de trabalho:

- **Análise Exploratória de Dados (EDA):** Investigação inicial para entender a estrutura, distribuição e estatísticas do dataset.

- **Visualização de Dados:** Criação de gráficos, como o `pairplot`, para identificar visualmente os padrões e a separabilidade das classes.
- **Treinamento de Modelos:** Implementação e treinamento de dois algoritmos de classificação distintos:
  - K-Nearest Neighbors (KNN)
  - Árvore de Decisão (Decision Tree)
- **Avaliação de Performance:** Análise dos resultados utilizando métricas como Acurácia e a Matriz de Confusão.
- **Otimização e Interpretação:**
  - Teste com diferentes hiperparâmetros (número de vizinhos `k` no KNN).
  - Análise de **Importância de Features** para entender quais características foram mais decisivas para o modelo.
  - Visualização da Árvore de Decisão para interpretar seu processo de decisão.
- **Validação Robusta:** Uso de **Validação Cruzada (Cross-Validation)** para garantir uma estimativa mais confiável da performance do modelo.
- **Persistência do Modelo:** Demonstração de como salvar o modelo treinado em um arquivo (`.joblib`) e carregá-lo para uso futuro.

## 🛠️ Tecnologias Utilizadas

- **Python 3.12**
- **Jupyter Notebook**
- **Pandas:** Para manipulação e análise de dados.
- **Scikit-learn:** Para os algoritmos de Machine Learning, pré-processamento e métricas de avaliação.
- **Matplotlib & Seaborn:** Para a visualização de dados.
- **Joblib:** Para salvar e carregar o modelo treinado.

## 🚀 Como Executar o Projeto

Siga os passos abaixo para configurar e executar o projeto em sua máquina local.

**1. Clone o Repositório**

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

**2. Crie e Ative o Ambiente Virtual**

É uma boa prática isolar as dependências do projeto.

```bash
# Crie o ambiente virtual
python -m venv venv

# Ative o ambiente (o comando varia entre os sistemas operacionais)
# No Linux ou macOS:
source venv/bin/activate
# No Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# No Windows (CMD):
.\venv\Scripts\activate.bat
```

**3. Instale as Dependências**

Crie um arquivo chamado `requirements.txt` na raiz do projeto com o seguinte conteúdo:

```txt
pandas
scikit-learn
matplotlib
seaborn
jupyter
joblib
```

Em seguida, instale todas as bibliotecas de uma vez com o comando:

```bash
pip install -r requirements.txt
```

**4. Execute o Jupyter Notebook**

Com o ambiente virtual ativo e as bibliotecas instaladas, inicie o servidor Jupyter:

```bash
jupyter notebook
```

Isso abrirá uma aba no seu navegador. Clique no arquivo `analise-iris.ipynb` para abri-lo e executar as células de código.

## 📈 Resultados

Os modelos treinados apresentaram excelente performance, alcançando **100% de acurácia** no conjunto de teste para as configurações otimizadas. A análise de importância de features revelou que as características da **pétala** (`petal length` e `petal width`) são significativamente mais informativas para a classificação do que as características da sépala.

A validação cruzada confirmou a robustez do modelo de Árvore de Decisão, apresentando uma acurácia média consistente e com baixo desvio padrão.
