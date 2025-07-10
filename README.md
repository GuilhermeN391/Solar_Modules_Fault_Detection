# Modelo de Detecção de Problemas em Painéis fotovoltaicos

Este reposiório hospeda um modelo de aprendizado de máquina em uma Rede Neural de Convolução (CNN) desenvolvido para detecção de anomalias em painéis fotovoltaicos, implementado usando PyTorch.

## Informações do projeto

- **Disciplina**: PPGEEC2318 - APRENDIZADO DE MÁQUINA
- **Turma**: T01 (2025.1)
- **Discentes - Matrícula**: Guilherme Nascimento da Silva - 20251011640 ; Israel da Silva Félix de Lima - 20241028222
- **Docente**: Prof. Dr. Ivanovitch Medeiros Dantas da Silva ([GitHub](https://github.com/ivanovitchm))
- **Dataset**: [Infrared Solar Modules](https://www.kaggle.com/datasets/marcosgabriel/infrared-solar-modules)
- **Inspirado em**: 
  - [Solar Modules Fault Detection with CNN+PyTorch](https://www.kaggle.com/code/aliakbaryaghoubi/solar-modules-fault-detection-with-cnn-pytorch)
  - [PPGEEC2318 - Week11](https://github.com/ivanovitchm/ppgeec2318/blob/main/lessons/week11/week11.ipynb)

## Model Card

Para apresentar as informações importantes do modelo, esse Model Card foi desenvolvido seguindo os princípios do artigo [Model Cards for Model Reporting](https://arxiv.org/pdf/1810.03993) para documentação de projetos envolvendo *Machine learning*.

### Detalhes do Modelo

- **Desenvolvido por**: Guilherme Nascimento da Silva e Israel da Silva Félix de Lima
- **Data de desenvolvimento**: 07/2025
- **Tipo de modelo**: Rede Neural Convolucional para classificação em Multiclasse implementada com PyTorch
- **Versão**: 1.0
- **Framework**: PyTorch
- **Número de features**: Variados ao longo do trabalho
- **Dispositivo utilizado**: CPU
- **Hyperparâmetros**: Variados ao longo do trabalho

### Uso Pretendido

- **Uso primário**: Detectar Anomalias em painéis solares, para correção do equipamento;
- **Usuários pretendidos**: Clientes proprietários de usinas fotovoltaicas de variadas cargas geradas;
- **Casos fora do escopo**: Não deve ser usado como único mecanismo para indicação de falhas em placas solares.

### Fatores

- **Fatores relevantes**: Buscando indicar a propensão a doença, fatores como idade, nível de colesterol, presença de angina (dor no peito), pressão arterial, frequência cardiaca, glicemina em jejum;
- **Fatores de avaliação**: A CNN foi aplicada com variações de hyperparâmetros a partir da observação das métricas resultantes, escolheu-se o melhor caso;
- **Fatores de avaliação**: A configuração das camadas da CNN também foi alterada também para verificação de melhoras no modelo;

### Métricas

As métricas de avaliação foram escolhidas considerando o forte desbalanceamento dos dados:

- **Acurácia**: Proporção de predições corretas;
- **Perdas**: Perdas totais de treinamento no final do processo;
- **Precisão**: Proporção de verdadeiros positivos entre os casos classificados como positivos;
- **Matriz de confusão**: Apresentação gráfica dos previstos e verdadeiros;

### Informações do Dataset

- O Dataset completo consiste no armazenamento de 20.000 imagens de diferentes estados de placas fotovoltaicas.
- Existe uma classe no Dataset que indica o estado a respectiva imagem: "anomaly_class". Ela possui 12 divisões para classificações das falhas:

| Nome da Classe | Descrição |
|---------|------------|
| Cell | Ponto quente ocorrendo com uma geometria quadrada em uma única célula. |
| Cell-Multi | Pontos quentes ocorrendo com umaa geometria quadrada em múltiplas células.  |
| Cracking | Anomalia causada por rachadura na superfície da placa  |
| Hot-Spot | Ponto quente em uma placa de película fina |
| Hot-Spot-Multi | Vários pontos quentes em uma placa de película fina|
| Shadowing | Luz solar obstruída por vegetação, estruturas artificiais ou fileiras adjacentes|
| Diode| Diodo de bypass ativado, tipicamente em 1/3 do módulo|
| Diode-Multi | Vários diodos de bypass ativados, tipicamente em 2/3 do módulo|
| Vegetation | Painéis bloqueados por vegetação|
| Soiling | Sujeira, poeira ou outros detritos na superfície do módulo.|
| Offline-Module | Módulo inteiro superaquecido|
| No-Anomaly | Painel funcionando normalmente|

### Resumo do Dataset original

| Aspecto | Informação |
|---------|------------|
| Total de imagens | 20.000 |
| Classe alvo | "anomaly_class" |
| Sub-Classes | 12  |
| Valores nulos | 0 (0.0%)|

### Dados de Avaliação

- **Conjunto de dados**: Conjunto de teste (15% do dataset original), divididos de acordo com a anomalia;
- **Tamanho**: 3.000 imagens;
- **Pré-processamento**: Normalização dos dados a partir do ImageFolder, utilizando o composer para redefinição da definição das imagens para 46x46 e parametrização a partir do procedimento de Standardization.

### Dados de Treinamento

- **Conjunto de dados**: Repositórios com as imagens (.jpg) divididas de acordo com a sua classificação;
- **Tamanho**: 17.000 instâncias - 85% do Dataset;
- **Características**: As 12 classes foram utilizadas inicialmente;
- **Definição das imagens**: 46x46 na aplicação do composer.

### Modelo inicial da Rede Neural Convolucional

Para o treinamento inicial dos dados, foi aplicado o modelo apresentado nas aulas para Redes Neurais Convolucionais, que constitui-se nos seguintes parâmetros para os dados aplicados:

- 1ª Convolução: Entrada com dimensões 3 x 46 x 46 após tratamento dos dados.
  - in_channels=3; out_channels=n_feature; kernel_size=3 -> Resize: 46 - 3 + 1 = 44;
  - Dimensões de Saída: n_feature x 44 x 44;

- 2ª Convolução: Entrada com dimensões n_feature x 22 x 22 após o max_pool2d com kernel_size=2 na saída da 1ª Convolução.
  - in_channels=n_feature; out_channels=n_feature; kernel_size=3 -> Resize: 22 - 3 + 1 = 20;
  - Dimensões de Saída: n_feature x 20 x 20;
 
Após a 2ª convolução, com o max_pool2d utilizando kernel_size=2, as dimensões para a aplicação do dropout se estabeleceram em n_feature x 10 x 10 x 50. Para a saída após a realização do **dropout**, aplicando o parâmetro p, se estabeleceram as 12 classificações para os dados.

Com os hiperparâmetros: n_feature = 5; p = 0.3 e lr = 0,0003. As métricas resultantes a saída com e seu dropout estão apresentadas na Tabela a seguir, com os gráficos de perdas e Matriz de confusão na secção de visualizações.

| Métrica | Informação |
|---------|------------|
| Acurácia de treinamento (sem Dropout) | 0,6794 |
| Acurácia de validação (sem Dropout) | 0,6446 |
| Acurácia de treinamento | 0,6482 |
| Acurácia de validação | 0,6237 |

Para simplificação dos futuros modelos, foi feita a aplicação do dropout em todos.

### Dados da Rede Neural Convolucional após variação de hiperparâmetros

A variação dos hiperparâmetros se consituiu na utilização da mesma arquitetura desenvolvida no modelo inicial, apenas variando os hiperparâmetros, com o objetivo de diminuir as perdas de treinamento e validação. Para fazer isso, de forma automática a partir da biblioteca _optuna mlflow scikit-learn_ foram testados 20 cenários com diversos valores para: n_feature; p e lr, e o cenário que entregou as melhores métricas de saída será apresentado, com os hiperparâmetros resultantes e as métricas de saída. 

Com os hiperparâmetros: n_feature = 14; p = 0,3371287734951104 e lr = 0,00047886451006673013. As perdas de validação nesse caso foram de 0,88912 e as perdas de treinamento foram 0,93424 com os seus gráficos apresentados na secção de visualizações.

### Nova Configuração das camadas da CNN

Outra abortagem para busca de otimização do modelo de CNN aplicado foi a alteração das camadas da rede.

## Visualizações

### 1. CNN base com duas camadas

#### 1.1 Aplicação dos parâmetros iniciais

**Matriz de Confusão:**  
   ![Matriz de confusão CNN2 padrão](images/DefaultmodelCNN2/conf_matrix.png)

**Evolução das perdas:**
  ![Evolução das perdas CNN2 padrão](images/DefaultmodelCNN2/loss_fig.png)
   

#### 1.2 Melhor modelo com variação dos hiperparâmetros

**Matriz de Confusão:**  
   ![Matriz de confusão CNN2 melhor](images/BestmodelCNN2/best_model_hyper_conf_matrix.png)
   
**Evolução das perdas:**  
   ![Evolução das perdas CNN2 melhor](images/BestmodelCNN2/best_model_hyper_loss_fig.png)

**Comparação entre os hiperparâmetros testados:**  
   ![Comparação entre os hiperparâmetros](images/hyper_CNN2.png)

#### 1.3 Hooks na saída de cada camada da rede

  ![Hooks CNN2](images/Hooks_CNN2.png)

### 2. CNN com modificação do número de camadas para quatro

#### 2.1 Aplicação dos parâmetros iniciais

**Matriz de Confusão:**  
   ![Matriz de confusão CNN4 padrão](images/DefaultmodelVeryDeep/conf_matrix_verydeep_default.png)

**Evolução das perdas:**  
   ![Evolução das Perdas CNN4 padrão](images/DefaultmodelVeryDeep/Loss_fig_verydeep_default.png)

#### 2.2 Melhor modelo com variação dos hiperparâmetros

**Matriz de Confusão:**  
   ![Matriz de confusão CNN4 melhor](images/BestmodelVeryDeep/conf_matrix_verydeep_best_model.png)

**Evolução das perdas:**  
   ![Evolução das Perdas CNN4 melhor](images/BestmodelVeryDeep/Loss_fig_verydeep_best_model.png)

**Comparação entre os hiperparâmetros testados:**  
   ![Comparação entre os hiperparâmetros](images/hyper_verydeep.png)

#### 2.3 Hooks na saída de cada camada da rede

  ![Hooks CNN4](images/Hooks_verydeep.png)

## Tabelas Adicionais

### Métricas resultantes por modelo


| Métrica | CNN2 - padrão | CNN2 - melhor | CNN4 - padrão | CNN4 - melhor |
|:---------:|:------------:|:------------:|:------------:|:------------:|
| Perdas de treinamento |	1,2597 | 0,9342 | 1,1028 | 0,6199 |
| Perdas de validação | 1,1821 | 0,8891 |  1,0292 | 0,7206 |
| Acurácia de treinamento | 0,6482 | 0,7444 | 0,6898 | 0,8306 |
| Acurácia de validação | 0,6482 | 0,7027 |  0,6680 | 0,7657 |
| Precisão ponderada | 0,5731 | 0,6719 | 0,5970 | 0,7578 |

## Principais Observações do projeto

## Como usar o modelo

## Referências

1. Yaghoubi, A. A. (2025). Solar Modules Fault Detection with CNN+PyTorch. Kaggle. https://www.kaggle.com/code/aliakbaryaghoubi/solar-modules-fault-detection-with-cnn-pytorch

2. Dantas da Silva, I. M. (2025). PPGEEC2318 - Week11: Machine Learning and Computer Vision - Part III. GitHub. https://github.com/ivanovitchm/ppgeec2318/blob/main/lessons/week11/week11.ipynb

