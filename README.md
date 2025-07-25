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
  - [Deep Learning with PyTorch Step-by-Step: A Beginner's Guide - Chapter 6](https://github.com/dvgodoy/PyTorchStepByStep/blob/master/Chapter06.ipynb)

## Model Card

Para apresentar as informações importantes do modelo, esse Model Card foi desenvolvido seguindo os princípios do artigo [Model Cards for Model Reporting](https://arxiv.org/pdf/1810.03993) para documentação de projetos envolvendo *Machine learning*.

### Detalhes do Modelo

- **Desenvolvido por**: Guilherme Nascimento da Silva e Israel da Silva Félix de Lima;
- **Data de desenvolvimento**: 07/2025;
- **Tipo de modelo**: Rede Neural Convolucional para classificação em Multiclasse implementada com PyTorch;
- **Versão**: 1.0;
- **Framework**: PyTorch;
- **Número de features**: Variados ao longo do trabalho de acordo com o relatado;
- **Dispositivo utilizado**: CPU;
- **Hyperparâmetros**: Variados ao longo do trabalho de acordo com o relatado.

### Uso Pretendido

- **Uso primário**: Detectar Anomalias em painéis solares, para realização de manutenção, correção ou até troca do equipamento;
- **Usuários pretendidos**: Clientes proprietários de usinas fotovoltaicas de variadas cargas geradas;
- **Casos fora do escopo**: Não deve ser usado como único mecanismo para indicação de falhas em placas solares.

### Fatores

- **Fatores relevantes**: Buscando o tipo de falha presente em placas solares, há 11 (onze) tipos de defeitos catalogados, que serão explorados a seguir;
- **Fatores de avaliação**: A CNN foi aplicada com variações de hyperparâmetros a partir da observação das métricas resultantes, escolheu-se o melhor caso;
- **Fatores de avaliação**: A configuração das camadas da CNN também foi alterada também para verificação de melhoras no modelo;

### Métricas

As métricas de avaliação foram escolhidas considerando o forte desbalanceamento dos dados:

- **Acurácia**: Proporção de predições corretas;
- **Perdas**: Perdas totais de treinamento no final do processo;
- **Precisão ponderada**: Proporção de verdadeiros positivos entre os casos classificados como positivos, sendo ponderada pela quantidade de valores da classe para fornecer um panorama geral;
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

- A classe "No-Anomaly" possui metade dos dados do Dataset, com 10.000 imagens, enquanto as 10.000 restantes estão distribuídas entre aas 11 classes restantes, que simbolizam defeitos presentes nos módulos fotovoltaicos relatados.

### Resumo do Dataset original

| Aspecto | Informação |
|:---------:|:------------:|
| Total de imagens | 20.000 |
| Classe alvo | "anomaly_class" |
| Sub-Classes | 12  |
| Maior classe | No-Anomally (10.000 amostras) |
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

Para o treinamento inicial dos dados, foi aplicado o modelo apresentado nas aulas para Redes Neurais Convolucionais (CNN2), que constitui-se nos seguintes parâmetros para os dados aplicados:

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

Para simplificação na execução dos seguintes modelos, foi feita a aplicação do dropout em todos, restringindo a inibição do dropout para esse modelo apenas.

### Dados da Rede Neural Convolucional após variação de hiperparâmetros

A variação dos hiperparâmetros se consituiu na utilização da mesma arquitetura desenvolvida no modelo inicial, apenas variando os hiperparâmetros, com o objetivo de diminuir as perdas de treinamento e validação. Para fazer isso, de forma automática a partir da biblioteca _optuna mlflow scikit-learn_ foram testados 20 cenários com diversos valores para: n_feature; p e lr, e o cenário que entregou as melhores métricas de saída será apresentado, com os hiperparâmetros resultantes e as métricas de saída. 

Com os hiperparâmetros: 
- n_feature = 14; 
- p = 0,33713;
- lr = 0,00047886; e
- Número de épocas = 15. 

Aplicando esses hiperparâmetros ao modelo CNN2, as perdas de validação foram de **0,88912** e as perdas de treinamento foram **0,93424** com os seus gráficos apresentados na secção de visualizações, com uma acurácia de validação igual a **0,7027** e uma precisão ponderada igual a **0,6719**.

### Nova Configuração das camadas da CNN

Outra abordagem para busca de otimização do modelo de CNN aplicado foi a alteração das camadas da rede. A partir do modelo fornecido, o CNN2, foi feita uma alteração na quantidade de camadas convolucionais, inserindo mais **duas camadas**, totalizando 4 camadas. Além disso, foi feita alteração na quantidade de canais de entrada e saída das camadas internas, caracterizando cada camada da seguinte forma, sem alteração dos dados de entrada.

- 1ª Convolução: in_channels = 3; out_channels = n_feature; kernel_size = 3;

- 2ª Convolução: in_channels = n_feature; out_channels = n_feature * 2; kernel_size = 3;

- 3ª Convolução: in_channels = n_feature * 2; out_channels = n_feature * 4; kernel_size = 3;

- 4ª Convolução: in_channels = n_feature * 4; out_channels = n_feature * 8; kernel_size = 3;

Com as dimensões da imagem em 46 x 46, após a 4ª convolução, com o max_pool2d utilizando kernel_size= 2, as dimensões para a aplicação do dropout se estabeleceram em n_feature * 8 x 2 x 2 x 50. Para a saída após a realização do **dropout**, aplicando o parâmetro p, se estabeleceram as 12 classificações para os dados.

Foram feitos testes utilizando os hiperparâmetros iguais aos testados no modelo base CNN2 e com um modelo melhorado aplicando a biblioteca _optuna mlflow scikit-learn_ no mesmo procedimento feito com a CNN2. Focando na apresentação do modeleo melhorado, com os resultados e visualizações dos dois modelos relatados em seguida, após o mapeamento dos hiperparâmetros, o melhor modelo encontrado e aplicado foi:

- n_feature = 14; 
- p = 0,16571;
- lr = 0,00040153; 

Com um treinamento de 15 épocas, as perdas finais de validação foram **0,7990**, a acurácia de validação foi **0,7466** e a precisão ponderada foi **0,7376**.

### Melhoria da taxa de aprendizado com o LRFinder

A partir do melhor modelo apresentado anteriormente, buscando formas de maximizar ainda mais os resultados, foram aplicados conceitos de procura do melhor _learning rate_ no melhor modelo de CNN, o VeryDeepCNN. Para fazer esse processo, foi implementada a biblioteca _LRFinder_ do Pytorch, seguindo os procedimentos presentes em [Deep Learning with PyTorch Step-by-Step: A Beginner's Guide - Chapter 6](https://github.com/dvgodoy/PyTorchStepByStep/blob/master/Chapter06.ipynb), que realiza essa procura baseado no modelo fornecido para a função de procura da melhor taxa de aprendizado. Com os parâmetros iniciais para aplicação:

- n_feature = 14; 
- p = 0,16571;
- lr = 0,00040153; 

A aplicação dos procedimentos da LRFinder retornou uma sugestão de _learning rate_ de **0,000876**. Aplicando esse novo parâmetro o modelo anterior com as mesmas configurações anteriores, as perdas finais de validação foram **0,8239**, a acurácia de validação foi **0,7507** e a precisão ponderada foi **0,7532**. As visualizações referentes a aplicação do LRFinder estão dispostas a seguir.

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

#### 2.4 Aplicação do LRFinder

**Procura da melhor taxa de aprendizado:**

  ![Gráfico do LRFinder](images/LRFinder_VDCNN.png)

**Evolução da perdas com o novo _learning rate_:** 
  ![Evolução das perdas novo LR](images/Loss_fig_bestLR.png)  

**Matriz de Confusão:**  
   ![Matriz de confusão CNN4 melhor](images/Conf_matriz_bestLR.png)

## Tabelas Adicionais

### Métricas resultantes por modelo


| Métrica | CNN2 - padrão | CNN2 - melhor | CNN4 - padrão | CNN4 - melhor |
|:---------:|:------------:|:------------:|:------------:|:------------:|
| Perdas de treinamento |	1,2597 | 0,9342 | 1,1028 | 0,6199 |
| Perdas de validação | 1,1821 | 0,8891 |  1,0292 | 0,7206 |
| Acurácia de treinamento | 0,6482 | 0,7444 | 0,6898 | 0,8306 |
| Acurácia de validação | 0,6237 | 0,7027 |  0,6680 | 0,7657 |
| Precisão ponderada | 0,5731 | 0,6719 | 0,5970 | 0,7578 |

## Principais Observações do projeto

- As 12 classificações do sistema estão desbalanceadas, uma vez que existem 11 tipos diferentes de anomalias catalogadas e apenas 1 classe para os módulos em defeitos, como já mencionado. Fazendo com que haja uma variação de amostras, variando dos 10.000 presentes na classe "No-Anomally", para 1.877 na classe "Cell", com o valor mínimo de amostras presentes na classe "Diode-Multi" com 175;
- Ao aplicar tanto no modelo de duas camadas, quanto no de quatro camadas, houve melhores resultados quando foram aplicados novos hiperparâmetros, que possibilitaram o melhor aproveitamento de abmos os modelos, entregando métricas de resultados maximizadas;
- Outro fator de melhora para o modelo de Rede Neural Convolucional foi a adição de mais camadas internas, com variação no número de features internas, havendo uma potencialização na melhora das métricas de resultados, principalmente ao alinhar o aumento de camadas à variação dos hiperparâmetros;
- A classe principal do Dataset, a "No-Anomally" concentrou metade dos dados totais do Dataset, ou seja, há uma grande influência nos resultados finais do modelo, por exemplo: a presição nela foi de 0.8825, um valor acima da precisão poderada, indicando que o modelo trabalha bem com clases mais povoadas, havendo mais amostras para validação e/ou treinamento;
- O processo de melhoria da _learning rate_ trouxe um aprimoramento ao modelo VeryDeepModel, com uma súbita melhora nos resultados do modelo, após a realização dos testes das diferentes taxas de aprendizado através do LRFinder.

## Como usar o modelo

```python
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype, Normalize
import cv2
import os

# 1. Definição completa da classe do modelo (a mesma usada no treinamento)
class VeryDeepCNN(nn.Module):
    def __init__(self, n_feature, p=0.0):
        super(VeryDeepCNN, self).__init__()
        self.n_feature = n_feature
        self.p = p

        # Camadas convolucionais
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_feature, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=n_feature, out_channels=n_feature * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=n_feature * 2, out_channels=n_feature * 4, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=n_feature * 4, out_channels=n_feature * 8, kernel_size=3, padding=1)

        # Camadas lineares (classificador)
        # Para uma imagem de entrada 46x46, a saída da última camada de pooling é 2x2
        self.fc1 = nn.Linear(in_features=(n_feature * 8) * 4, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=12) # 12 classes de saída

        # Camada de dropout
        self.drop = nn.Dropout(self.p)

    def featurizer(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2)
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=2)
        x = F.max_pool2d(F.relu(self.conv4(x)), kernel_size=2)
        return nn.Flatten()(x)

    def classifier(self, x):
        if self.p > 0:
            x = self.drop(x)
        x = F.relu(self.fc1(x))
        if self.p > 0:
            x = self.drop(x)
        x = self.fc2(x)
        return x

    def forward(self, x):
        x = self.featurizer(x)
        x = self.classifier(x)
        return x

# 2. Instanciar e carregar o modelo treinado
# Use os mesmos hiperparâmetros do modelo salvo
best_n_feature = 14
best_dropout_rate = 0.1657
model = VeryDeepCNN(n_feature=best_n_feature, p=best_dropout_rate)

# Carregar os pesos salvos
try:
    model.load_state_dict(torch.load("modelo_classificacao.pth"))
    print("Modelo carregado com sucesso!")
except FileNotFoundError:
    print("Erro: 'modelo_classificacao.pth' não encontrado. Certifique-se de que o arquivo está no diretório correto.")
    model = None

# Colocar o modelo em modo de avaliação
if model:
    model.eval()

# 3. Definir o pré-processamento e a lista de classes
# As médias e desvios padrão devem ser os mesmos calculados no treinamento
norm_mean = [0.6187, 0.6187, 0.6187] 
norm_std = [0.0734, 0.0734, 0.0734]

preprocess_transform = Compose([
    Resize((46, 46)),
    ToImage(),
    ToDtype(torch.float32, scale=True),
    Normalize(mean=norm_mean, std=norm_std)
])

# A ordem dos nomes deve corresponder exatamente à ordem do treinamento
class_names = ['Cell', 'Cell-Multi', 'Cracking', 'Diode', 'Diode-Multi',
               'Hot-Spot', 'Hot-Spot-Multi', 'No-Anomaly', 'Offline-Module',
               'Shadowing', 'Soiling', 'Vegetation']

# 4. Funções para realizar a predição em uma nova imagem
def preprocess_image(image_path):
    """Lê uma imagem, aplica as transformações e a prepara para o modelo."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Converte de BGR (OpenCV) para RGB
    image_tensor = preprocess_transform(image)
    return image_tensor.unsqueeze(0) # Adiciona a dimensão do batch

def detect_anomaly(image_path, model, class_names):
    """Carrega uma imagem, a pré-processa e retorna o nome da classe prevista."""
    if not model:
        print("Modelo não está carregado. Impossível prever.")
        return None

    image_tensor = preprocess_image(image_path)
    if image_tensor is None:
        print(f"Não foi possível ler a imagem em: {image_path}")
        return None

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
    
    return class_names[predicted_idx.item()]

# --- Exemplo de Uso ---
if __name__ == "__main__":
    # A MUDAR: Substitua pelo caminho da imagem que você quer classificar
    caminho_da_imagem = 'caminho/para/sua/imagem.jpg' 

    if not os.path.exists(caminho_da_imagem):
        print(f"Erro: Arquivo de imagem não encontrado em '{caminho_da_imagem}'")
    else:
        anomalia_prevista = detect_anomaly(caminho_da_imagem, model, class_names)
        if anomalia_prevista:
            print(f"\nA anomalia detectada na imagem é: {anomalia_prevista}")
```
## Referências

1. Yaghoubi, A. A. (2025). Solar Modules Fault Detection with CNN+PyTorch. Kaggle. https://www.kaggle.com/code/aliakbaryaghoubi/solar-modules-fault-detection-with-cnn-pytorch

2. Dantas da Silva, I. M. (2025). PPGEEC2318 - Week11: Machine Learning and Computer Vision - Part III. GitHub. https://github.com/ivanovitchm/ppgeec2318/blob/main/lessons/week11/week11.ipynb

3. Godoy, D. V. (2022). Deep Learning with PyTorch Step-by-Step: A Beginner's Guide - Chapter 6. GitHub. https://github.com/dvgodoy/PyTorchStepByStep/blob/master/Chapter06.ipynb
