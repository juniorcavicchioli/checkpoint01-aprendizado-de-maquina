# Checkpoint01 - aprendizado de maquina
Atividade desenvolvida para a disciplina Disruptive Architectures - IOT, IOB AND IA. Nota recebida: 10/10

Tecnologias: Python, as bibliotecas Seaborn, MatPlotLib, Pandas, Numpy e Scikit-learn bem como a plataforma Google Colab.

Conceitos aplicados: Métodologia CRISP-DM, machine learning, análise, preparação, interpretação e comunicação dos dados bem como o uso dos modelos de regressão `Linear Regressor` e `Random Forest Regressor`.

## Enunciado proposto pelo professor

Você foi contratado por uma empresa inovadora do ramo imobiliario como QuintoAndar, Loft, Terraz, grupo Zap (ZAP, Viva Real, DataZAP, FipeZAP, SuaHouse, Geoimóvel, Sub100 e Conecta Imobi) ou Imobi Conference. O seu desafio está no apoio à tomada de decisão baseada em dados (empresa data driven).

Nesse sentido, use a base de dados [`aptos.csv`](/aptos.csv) para realizar a descoberta do conhecimento e responder a pergunta:

**`Quanto vale um apartamento de 72m2 com 2 quartos, 2 vagas, reformado no Centro?`**

## Solução

Mais detalhes da solução bem como as respostas estão todas no arquivo [`CheckPoint.py`](/CheckPoint.ipynb)

### Exploração dos dados
---
Comecei pela exploração dos dados. Carreguei o arquivo usando o pandas e respondi as perguntas levantadas pelo professor.

**1. Apresente uma breve descrição do significado de cada atributo.**

|Atributo|Descrição|
|-|-|
|Metros| área do apartamento em metros
|Valor| preço do imóvel dividido por 100.000
|Quartos| número de quartos
|Vagas|número de vagas de garagem
|Reformado| se foi reformado ou não
|Bairro| bairro em que o imóvel se encontra

**2. Cite 2 cenários que podem fazer sentido na leitura dos dados apresentados.**
1. Eu acho que o imóvel fica mais caro de acordo com o tamanho.
2. Eu acho que um imóvel fica mais caro quando tem mais garagens.

_(ambas afirmações foram testadas a frente)_

### Análise descritiva dos dados
---
O segundo passo foi a análise exploratória. Usando de gráficos e tabelas, tentei verificar se as minhas hipóteses estavam corretas.

Primeiro, percebi que haviam dados nulos nas colunas de valor e reformado. Para polir o dataset, removi ambas as linhas com dados faltantes.

Depois, imprimi um boxplot junto de uma impressão de listas para identificar os outliers do dataset.
```
Outliers_metros:  [199]
Outliers_vagas:  [4 4 4]
```

![boxplot](https://user-images.githubusercontent.com/101985616/229565971-4266d364-edab-4305-8b10-524075eaf727.png)


Também imprimi a contagem de bairros:
|Bairro|Qtd de imóveis|
|-|:-:|
|Centro| 34|
|Baeta Neves| 17
|Assuncao|9|
|Rudge Ramos|9
Vila Lusitania|6
Planalto |5
Demarchi  |5
Ferrazopolis|4
Taboao|4
Santa Teresinha|4
Independencia|3
Nova Petropolis|3
Iraja|3
Pauliceia|2
**Jardim do Mar**|**2**
Jordanopolis|1
Piraporinha|1

Através dos gráficos de dispersão e correlação, pude confirmar minhas hipóteses. O gráfico de correlação indica `0.8` de relação entre `valor` e `vagas` e `0.92` entre `valor` e `metros`. Isso significa que conforme o preço de um imóvel aumenta, o número de vagas e a metragem aumentam e vice-versa. Isso é vísivel no gráfico de disperção, que mostra uma linearidade positiva.

_Cores diferentas apenas para vizualização_


![corrmatx](https://user-images.githubusercontent.com/101985616/229566640-844255ca-98f1-4da5-ac8e-2861398ef727.png)

![scatterplot](https://user-images.githubusercontent.com/101985616/229566886-9e9ea8de-9e96-4bb5-bd3e-debd4835cb21.png)

E após a análise, respondi a pergunta do professor sobre os outliers:

**3. Foram localizados outliers? Qual o método de análise adotado? Se existe, como os dados foram tratados (foram removidos ou mantidos)?**

Interpretando o box plot, vemos que existem 1 outlier para `metros` e 3 para `vagas`.

Expandindo a tabela do `df` (dataframe) vemos que o outlier de metros é um outlier em comum com vagas. Também é possível descobrir facilmente que o `Jardim do Mar` que possui apenas 2 ocorrencias no dataframe faz parte dos outliers restantes de vagas e que todos esses outliers possuem valores altos (porém não discrepantes) em outras colunas também.

Decidi remover apenas o outlier de metros por ser uma ocorrencia única e ser o valor mais alto existente em todos as colunas e manter os outliers das vagas para não perder completamente todos os valores altos que temos no dataset junto com os únicos valores para o bairro Jardim do Mar.

### Desenvolvimento do modelo
---

Iniciei a modelagem escolhendo quais modelos eu usária para fazer a predição. Novamente, respondi as perguntas feitas pelo professor:

**4. O conjunto de dados indica a aplicação de qual modelo de aprendizagem de maquina para determinação do valor de um imóvel? Por que?**

O aprendizado supervisionado é melhor. Temos os valores de entrada e exemplos do valor de saída desejado. No caso, os valores: metros, vagas, quartos, reformado e o bairro são as entradas e desejamos que a nossa máquina seja capaz de prever o valor, que já possuimos exemplos de saída os quais usaremos para treinar o modelo.

**5. Qual a técnica sugerida para este projeto?**
[X] Regressão

[ ] Classificação

[ ] Clusterização

**6. Escolha 2 modelos de algoritmos para testar a performance de cada um deles.**
Pela interpretação dos gráficos, existe uma linearidade. Então escolhi:
1. Regressão linear
2. Random Forest Regressor

**7. Explique como cada algoritmo funciona.**

1. A regressão linear é um algoritmo que busca traçar uma linha reta que melhor se ajuste aos pontos de um conjunto de dados, de modo a prever uma relação entre uma variável dependente e uma ou mais variáveis independentes.
2. O Random Forest Regressor é um algoritmo de aprendizado de máquina que combina várias árvores de decisão para prever um resultado numérico. Ele funciona criando várias árvores de decisão aleatórias e combinando suas previsões para gerar uma previsão final mais precisa.

### Treinamento e teste do modelo
---
O primeiro passo que tomei foi transformar em dummies os bairros pois os algoritmos que usei lidam melhor com números. Esse método transforma os bairros em colunas e dão a eles valores de 0 (falso) e 1 (verdadeiro). Assim o algoritmo consegue entender qual bairro é o pertencente aquela linha enquanto trabalha exclusivamente com números.

Então separei os dados para treino e teste. Variei nas proporções durante minha análise como explico no próximo passo.

#### Algoritmo 1 - Regressão linear
Após rodar sem `random_state` definido várias vezes, percebi que o valor de R2 variava muito. Então decidi fazer testes e comparações com valores diferentes e variando a variável `test_size` de uma proporção 80-20 para 70-30.

|`random_state`|`test_size`|RS_treino|RS_teste|Diferença|
|:-:|:-:|:-:|:-:|:-:|
|42|0.2|0.87|0.81|0.06
|42|0.3|0.89|0.73|0.13
|100|0.2|0.86|0.86|0
|100|0.3|0.85|0.83|0.02
|12|0.2|0.87|0.74|0.13
|12|0.3|0.88|0.74|0.14
|1|0.2|0.88|0.76|0.12
|1|0.3|0.88|0.76|0.12
|2|0.2|0.84|0.90|-0.06
|2|0.3|0.85|0.82|0.03
|-|-|-|-|*Diferença média = 0.069*|

#### Algoritmo 2 - Random forest regressor
Primeiro, comparei o número de árvores para identificar se existia uma faixa de quantidade que fosse melhor. A relação que encontrei é a de que, conforme o número de árvores cresce, o resultado começa a ficar igual.
|`n_estimators`|R2_treino|R2_teste|
|:-:|:-:|:-:|
|10|0.95|0.77|
|50|0.96|0.79|
|100|0.96|0.80|
|200|0.96|0.80|
|*`random_state` da divisão = 100*|

Depois fiz a mesma comparação que fiz no primeiro algoritmo, porém variando somente o `random_state` e não mais o `test_size`.

|`random_state`|RS_treino|RS_teste|Diferença|
|:-:|:-:|:-:|:-:|
|42|0.96|0.85|0.11
|100|0.96|0.80|0.16
|12|0.96|0.71|0.25
|1|0.96|0.77|0.19
|2|0.96|0.79|0.17
|-|-|-|*Média = 0.176*|

Sumarizei os resultados que considerei os melhores conforme o professor pediu e respondi sua pergunta.

| |Treino|Teste|`random_state`
|:-:|:-:|:-:|:-:
|Linear| 0.86 | 0.86 | 100
|Random Forest|0.96	| 0.85 | 42

**8. Qual dos algoritmos obteve um resultado melhor? Justifique**

Regressão Linear.

O Random Forest me pareceu com overfitting. Todos os resultados de treino eram ótimos, porem na hora do teste eles tinham uma diferença entre 0.10 e 0.20, o que pode significar que ele decorou e não está conseguindo generalizar para predizer os resultados.

Enquanto isso, o Linear não passou de 0.90. O que significa que a predição não está tão boa quanto poderia. Apesar disso, confio mais nos resultados da linear que conseguiu resultados com menos variação entre o treino e o teste. Um dos `random_state` testados obteve uma diferença de quase 0 e outros resultados com diferença de 0.03 e 0.06

### Modelo de produção: Teste com novos dados
---
Finalmente chegou o momento de responder à pergunta: **`Quanto vale um apartamento de 72m2 com 2 quartos, 2 vagas, reformado no Centro?`**

Após inserir os valores que já sabemos sobre o imóvel, pedi para ambos os modelos preverem o valor dele. Minha saída foi:
```
Preço com LinearRegressor: 394.3286699019902
Preço com RandomForestRegressor: 370.31666666666655
```

### Conclusões finais
---
Instruído pelo professor, respondi as perguntas dele sobre o processo e resultados obtidos.

**9. O modelo desenvolvido obteve um resultado satisfatório? Justifique.**

O modelo de regressão linear obteve um resultado satisfatório, apesar de não ótimo. Ele obteve valores um pouco mais confiáveis. Observando os gráfico usados para interpretação, foi visível que existe uma linearidade nas relações dos dados de entrada com o valor, o que traz mais confiabilidade para esse modelo.

Já o modelo de regressão random forest pareceu estar com overfitting, então não considero um resultado satisfatório, apesar de não ter sido de todo ruim


**10. O que faria diferente para melhorar resultado obtido? Justifique.**

A maioria dos bairros não possui imóveis o suficiente além de valores confiáveis e críveis, apenas mais incomuns, serem considerados outliers. Basicamente o dataset é muito pequeno para o que foi proposto. Talvez se desconsiderasse o bairro a máquina tivesse um aprendizado melhor mas sabemos que, na vida real, o bairro afeta o valor de um imóvel então isso não seria uma decisão inteligente.

## Autor
Feito por [@juniorcavicchioli](https://github.com/juniorcavicchioli?tab=repositories). Entre em contato!

LinkedIn: [Adilson Roberto Cavicchioli Junior](https://www.linkedin.com/in/adilson-roberto-cavicchioli-junior-6816b7192?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BIpMh5bVEQOi82%2FRHJ6oxkg%3D%3D) <br>
Email: [cavicchioli.adilson@gmail.com](mailto:cavicchioli.adilson@gmail.com)

Sinta-se à vontade para me contatar caso tenha dificuldade em testar o programa, para perguntas, sugestões ou colaborações em projetos futuros!
