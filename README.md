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
- **Precisão**: Proporção de verdadeiros positivos entre os casos classificados como positivos;
- **Recall**: Proporção de fraudes corretamente identificadas;
- **Especificidade**: Proporção de transações legítimas corretamente identificadas;
- **F1-Score**: Média harmônica entre precisão e recall;
- **AUC-ROC**: Área sob a curva ROC.

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
| Perdas de treinamento (sem Dropout) | 0,6794 |
| Perdas de validação (sem Dropout) | 0,6446 |
| Perdas de treinamento | 0,6482 |
| Perdas de validação | 0,6237 |

### Dados da Rede Neural Convolucional após variação de hiperparâmetros

A variação dos hiperparâmetros se consituiu na utilização da mesma arquitetura desenvolvida no modelo inicial, apenas variando os hiperparâmetros. Para fazer isso, de forma automática a partir da biblioteca _optuna mlflow scikit-learn_ foram testados 20 cenários com diversos valores para: n_feature; p e lr, e o cenário que entregou as melhores métricas de saída será apresentado, com os hiperparâmetros resultantes e as métricas de saída. 

Com os hiperparâmetros: n_feature = 14; p = 0,3371287734951104 e lr = 0,00047886451006673013. As perdas de validação nesse caso foram de 0,88912 com os seus gráficos apresentados na secção de visualizações.

### Nova Configuração das camadas da CNN

Outra abortagem para busca de otimização do modelo de CNN aplicado foi a alteração das camadas da rede.

## Visualizações

### CNN com variação de Hiperparâmetros

**Evolução das perdas:**  
   ![Evolução das perdas](images/loss_graf.png)

**Matriz de Confusão:**  
   ![Matriz de Confusão](images/conf_matrix.png)

## Principais Observações do projeto

## Como usar o modelo

## Referências

1. Yaghoubi, A. A. (2025). Solar Modules Fault Detection with CNN+PyTorch. Kaggle. https://www.kaggle.com/code/aliakbaryaghoubi/solar-modules-fault-detection-with-cnn-pytorch

2. Dantas da Silva, I. M. (2025). PPGEEC2318 - Week11: Machine Learning and Computer Vision - Part III. GitHub. https://github.com/ivanovitchm/ppgeec2318/blob/main/lessons/week11/week11.ipynb

