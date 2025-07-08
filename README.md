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
- **Número de features**: 20 [A ALTERAR]
- **Dispositivo utilizado**: CPU
- **Hyperparâmetros**:
  - Batch size: 16
  - Learning rate: 0,0005659641838443741 - Modelo Final
  - Número de épocas: 14 - Modelo Final

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
- **Tamanho**: 17.000 instâncias;
- **Características**: As 12 classes foram utilizadas inicialmente;
- **Definição das imagens**: 46x46 na aplicação do composer.

### Modelo inicial da Rede Neural Convolucional

Para o treinamento inicial dos dados, foi aplicado o modelo apresentado nas aulas para Redes Neurais Convolucionais, que constitui-se nos seguintes parâmetros para os dados aplicados:

- 1ª Convolução: Entrada com dimensões 3x46x46 após tratamento dos dados.
  - in_channels=3; out_channels=n_feature; kernel_size=3 -> Resize: 46 - 3 + 1 = 44;
  - Dimensões de Saída: n_featurex44x44;

- 2ª Convolução: Entrada com dimensões n_featurex22x22 após o max_pool2d com kernel_size=2 na saída da 1ª Convolução.
  - in_channels=n_feature; out_channels=n_feature; kernel_size=3 -> Resize: 22 - 3 + 1 = 20;
  - Dimensões de Saída: n_featurex20x20;
 

### Dados da Rede Neural Convolucional após variação

### Nova Configuração das camadas da CNN 

## Visualizações

## Principais Observações do projeto

## Como usar o modelo

## Referências

1. Yaghoubi, A. A. (2025). Solar Modules Fault Detection with CNN+PyTorch. Kaggle. https://www.kaggle.com/code/aliakbaryaghoubi/solar-modules-fault-detection-with-cnn-pytorch

2. Dantas da Silva, I. M. (2025). PPGEEC2318 - Week11: Machine Learning and Computer Vision - Part III. GitHub. https://github.com/ivanovitchm/ppgeec2318/blob/main/lessons/week11/week11.ipynb

