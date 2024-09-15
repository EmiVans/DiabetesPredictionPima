

#Instalar la paqueteria 
#install.packages("mlbench")

rm(list = ls(all.names = TRUE))
gc()

# Cargar la base de datos
library(caret)
library(tidyverse)
library(dbplyr)
library(psych)
library(GGally)
library(NbClust)
library(factoextra)
library(clusterCrit)
library(ggplot2)
library(mlbench)
library(metrica)
library(glmnet)
library(MASS)
library(class)
library(e1071)
library(randomForest)
data("PimaIndiansDiabetes2")

#Quitamos las filas con algún valor NA y las guardamos en la variable datos
datos <- na.omit(PimaIndiansDiabetes2)
summary(datos)




# Análisis descriptivo o una visualización de cada variable 
# predictora distinguiendo por los dos grupos a clasificar. Comente lo que observe.

X11()
ggpairs(data=datos, title="Datos sobre la diabetes", aes(colour = diabetes))



describeBy(datos ~ diabetes,mat=TRUE)

# En general, podemos ver que de la grafica "Datos sobre la diabetes" que en general
# Las medias para cada variable, son parecidas entre si presentan diabetes y no.
# Donde es mas evidente la diferencia es en las variables "glucose", "insulin" y 
# "age", donde se puede ver que en promedio, a mayor valor de estos factores es 
# mayor la probabildad de tener diabetes. Y en general, a mayor valor de cada factor
# parece ser que la probabilidad de tener diabetes aumenta. Solo "mass" y "pedigree"
# parecen no presentar un cambio entre las dos categorias.


#--------------------------------------------------------------------------------------------------------------------------------------

# Componentes Principales


dat <- datos[,-9]
p=2
pca <- principal(dat, nfactor = 2, rotate = "none",scores=TRUE)
pca
X11()
biplot(pca,group=datos$diabetes, pch=c(21,0)[datos$diabetes])

# Del Biplot de componentes principales, se puede ver que el grupo rojo parece estar
# correlacionado positivamente con cada una de las variables de nuestro estudio, 
# y donde se preenta la correlacion mayor es con las variables "pression", "glucose"
# y "insuline".
# Verificar la estructura de la variable

#---------------------------------------------------------------------------------------------------------------------------------


# Conciderando un modelo para datos binarios con liga logit. Explore modelos 
# con los efectos principales de las variables, así como su interacción 
# (y/o los cuadrados de las variables).

# Ajustamos los niveles de referencia
datos$diabetes <- factor(datos$diabetes, levels = c("neg", "pos"))
str(datos$diabetes)

#Vamos a usar repeated holdout method con B=50 cara calcular el poder predictivo
#Usaremos las mismas particiones para todos los ejercicios
set.seed(1)
B=50
Partition<- createDataPartition(datos$diabetes, p = .80, list = FALSE, times = B)

######### SOLO EFECTOS PRINCIPALES #########
#ENTRENAMIENTO DEL MODELO
mod1 <- glm(diabetes ~ ., data=datos,   family=binomial(link="logit"))
summary(mod1)

(logit=predict(mod1, newdata = datos, type = "response"))
(res=ifelse(logit>=.5,levels(datos$diabetes)[2],levels(datos$diabetes)[1]))
metrics_summary(obs = datos$diabetes, pred = res, metrics_list=c("accuracy", "recall", "specificity"),type = 'classification')
# Los indices para el modelo son accuracy=0.7831633, recall=0.5692308 y specificity 0.8893130

#ESTIMACION DEL PODER PREDICTIVO

mod1RHM=function(x, IndTrain, Dat){
  train= IndTrain[,x]
  test = (-train)
  modtr=glm(diabetes ~ ., data=Dat[train,],   family=binomial(link="logit"))
  preda=predict(modtr, newdata = Dat[test,], type = "response")
  predb=ifelse(preda>=.5,levels(Dat$diabetes)[2],levels( Dat$diabetes)[1])
  resPod=metrics_summary(obs = Dat[test,"diabetes"], pred = predb, metrics_list=c("accuracy", "recall", "specificity"),type = 'classification')
  return(resPod[,2])
}

TCC.B.mod1= sapply(1:B,mod1RHM, IndTrain=Partition, Dat=datos)
(TCC.RHM.mod1=rowMeans(TCC.B.mod1))
# Los indices del poder de prediccion son accuracy=0.7841026, recall=0.5676923 y specificity=0.8923077



######### Modelo con interacciones #########
#ENTRENAMIENTO DEL MODELO
mod2_int <- glm(diabetes ~ .^2, data=datos,   family=binomial(link="logit"))
summary(mod2_int)

(logit_int=predict(mod2_int, newdata = datos, type = "response"))
(res_int=ifelse(logit_int>=.5,levels(datos$diabetes)[2],levels(datos$diabetes)[1]))
metrics_summary(obs = datos$diabetes, pred = res_int, metrics_list=c("accuracy", "recall", "specificity"),type = 'classification')
# Los indices para el modelo son accuracy=0.0.80102, recall=0.63846 y specificity 0.88167

#ESTIMACION DEL PODER PREDICTIVO
mod2RHM_int=function(x, IndTrain, Dat){
  train= IndTrain[,x]
  test = (-train)
  modtr=glm(diabetes ~ .^2, data=Dat[train,],   family=binomial(link="logit"))
  preda=predict(modtr, newdata = Dat[test,], type = "response")
  predb=ifelse(preda>=.5,levels(Dat$diabetes)[2],levels( Dat$diabetes)[1])
  resPod=metrics_summary(obs = Dat[test,"diabetes"], pred = predb, metrics_list=c("accuracy", "recall", "specificity"),type = 'classification')
  return(resPod[,2])
}

TCC.B.mod2_int= sapply(1:B,mod2RHM_int, IndTrain=Partition, Dat=datos)
(TCC.RHM.mod2_int=rowMeans(TCC.B.mod2_int))
# Los indices del poder de prediccion son accuracy=0.7402, recall=0.5438 y specificity=0.8384




######### Modelo con variables al cuadrado #########
#ENTRENAMIENTO DEL MODELO
names(datos)
(xnames = names(which(sapply(datos, is.numeric))) )
# definir variables predictoras continuas
forexp=as.formula(  paste('diabetes ~.',"+", paste(paste('I(',xnames,'^2)',collapse = ' + ')  ) )) 
forexp
mod3_cuad=glm(forexp, data = datos, family = binomial(link="logit"))  # regla final, la que se usar?a en producci?n
summary(mod3_cuad)

logit_cuad=predict(mod3_cuad, newdata = datos, type = "response")
res_cuad=ifelse(logit_cuad>=.5,levels(datos$diabetes)[2],levels(datos$diabetes)[1])
metrics_summary(obs = datos$diabetes, pred = res_cuad, metrics_list=c("accuracy", "recall", "specificity"),type = 'classification')
# Los indices para el modelo son accuracy=0.0.80102, recall=0.63846 y specificity 0.88167


#ESTIMACION DEL PODER PREDICTIVO
mod3RHM_cuad=function(x, IndTrain, Dat, forme){
  train= IndTrain[,x]
  test = (-train)
  modtr=glm(forme, data=Dat[train,],   family=binomial(link="logit"))
  preda=predict(modtr, newdata = Dat[test,], type = "response")
  predb=ifelse(preda>=.5,levels(Dat$diabetes)[2],levels( Dat$diabetes)[1])
  resPod=metrics_summary(obs = Dat[test,"diabetes"], pred = predb, 
                         metrics_list=c("accuracy", "recall", "specificity"),
                         type = 'classification')
  return(resPod[,2])
}

TCC.B.mod3_cuad= sapply(1:B,mod3RHM_cuad, IndTrain=Partition, Dat=datos,forexp)
(TCC.RHM.mod3_cuad=rowMeans(TCC.B.mod3_cuad))
# Los indices del poder de prediccion son accuracy=0.7774, recall=0.5869 y specificity=0.8726

# De los 3 modelos analisados el que parece tener un mejor predictivo es el de 
# iteracciones de segundo orden que toma estos valores: Los indices del poder 
# de prediccion son accuracy=0.7402, recall=0.5438 y specificity=0.8384

#--------------------------------------------------------------------------------------------------------------------------

# IV) Explorando selección de variables.

############ Aplicaremos selección de variables por método lasso################

######### Solo efectos principales ############
######### ENTRENAMIENTO 

#Primero creamos la matriz diseño 
#Pero primero debemos ver que la base "datos" esté en formato XY
Xmod <- model.matrix(diabetes ~ ., data=datos)
Ymod <-  datos[,"diabetes"] 

mod.lasso = glmnet(Xmod, Ymod, family = binomial(link="logit"), nlambda = 200)
#Faltar?a tunear (definir valor) de lambda
set.seed(1)
mod.lasso.tun=cv.glmnet(Xmod, Ymod, nfolds = 5, type.measure ="class", gamma = 0, 
                        relax = FALSE, family = "binomial", nlambda = 50)
coef(mod.lasso.tun, s = "lambda.min") #Regla final que va a salir a produccion.
mod.lasso.tun$lambda.min
logit_lasso=predict(mod.lasso.tun, newx = Xmod, type = "response", s = "lambda.min")
res_lasso=ifelse(logit_lasso>=.5,levels(datos$diabetes)[2],levels(datos$diabetes)[1])
metrics_summary(obs = datos$diabetes, pred = res_lasso, metrics_list=c("accuracy", "recall", "specificity"),type = 'classification')
# Los indices para el modelo son accuracy=0.7831633, recall=0.5461538 y specificity 0.9007634



mod1RHM_sVar=function(x, IndTrain, Dat){
  train= IndTrain[,x]
  test = (-train)
  Xmodtotal = model.matrix(diabetes ~ ., data=datos)[,-1]
  Xmodt = Xmodtotal[train,]
  Ymodt = Dat[train,"diabetes"]
  modtr=cv.glmnet(Xmodt, Ymodt, nfolds = 5, type.measure ="class", gamma = 0, 
                  relax = FALSE, family = "binomial", nlambda = 50)
  preda=predict(modtr, newx = Xmodtotal[test,], type = "response", s="lambda.min")
  predb=ifelse(preda>=.5,levels(Dat$diabetes)[2],levels( Dat$diabetes)[1])
  resPod=metrics_summary(obs = Dat[test,"diabetes"], pred = predb, metrics_list=
                           c("accuracy", "recall", "specificity"),type = 
                           'classification')
  return(resPod[,2])
}

TCC.B.mod1_sVar= sapply(1:B,mod1RHM_sVar, IndTrain=Partition, Dat=datos)
(TCC.RHM.mod1_sVar=rowMeans(TCC.B.mod1_sVar))
# Los indices del poder de prediccion son accuracy=0.7764103, recall=0.5200000 y specificity=0.9046154



######### Modelo con interacciones ############
######### ENTRENAMIENTO##############

# Define todas las interacciones de segundo orden manualmente
interacciones <- combn(names(datos)[-9], 2, FUN = function(x) paste(x, collapse = ":"))
interacciones_formula <- as.formula(paste("diabetes ~ . + ", paste(interacciones, collapse = " + ")))

Xmod2 <- model.matrix(interacciones_formula, data=datos)[,-1]
Ymod2 <-  datos[,"diabetes"] 

mod.lasso2_int = glmnet(Xmod2, Ymod2, family = binomial("logit"), nlambda = 200)
#Faltar?a tunear (definir valor) de lambda
set.seed(1)
mod.lasso.tun_int=cv.glmnet(Xmod2, Ymod2, nfolds = 5, type.measure ="class", gamma = 0, 
                        relax = FALSE, family = "binomial", nlambda = 50)
coef(mod.lasso.tun_int, s = "lambda.min") #Regla final que va a salir a produccion.

logit_lasso_int=predict(mod.lasso.tun_int, newx = Xmod2, type = "response", s = "lambda.min")
res_lasso_int=ifelse(logit_lasso_int>=.5,levels(datos$diabetes)[2],levels(datos$diabetes)[1])
metrics_summary(obs = datos$diabetes, pred = res_lasso_int, metrics_list=c("accuracy", 
                        "recall", "specificity"),type = 'classification')
# Los indices para el modelo son accuracy=0.7959, recall=0.5769 y specificity 0.9045
 


#ESTIMACION PODER PREDICTIVO
mod1RHM_sVar_int=function(x, IndTrain, Dat, forme){
  train= IndTrain[,x]
  test = (-train)
  Xmodtotal = model.matrix(forme, data=datos)[,-1]
  Xmodt = Xmodtotal[train,]
  Ymodt = Dat[train,"diabetes"]
  modtr=cv.glmnet(Xmodt, Ymodt, nfolds = 5, type.measure ="class", gamma = 0, 
                  relax = FALSE, family = "binomial", nlambda = 50)
  preda=predict(modtr, newx = Xmodtotal[test,], type = "response", s="lambda.min")
  predb=ifelse(preda>=.5,levels(Dat$diabetes)[2],levels( Dat$diabetes)[1])
  resPod=metrics_summary(obs = Dat[test,"diabetes"], pred = predb, metrics_list=
                           c("accuracy", "recall", "specificity"),type = 
                           'classification')
  return(resPod[,2])
}

TCC.B.mod1_sVar_int= sapply(1:B,mod1RHM_sVar_int, IndTrain=Partition, Dat=datos, forme = interacciones_formula)
(TCC.RHM.mod1_sVar_int=rowMeans(TCC.B.mod1_sVar_int))
# Los indices del poder de prediccion son accuracy=0.7766, recall=0.5238 y specificity=0.9030


######### Modelo variables al cuadrado ############
######### ENTRENAMIENTO##############
# Crear la matriz de diseño
Xmod3 <- model.matrix(forexp, data=datos)[,-1]
Ymod3 <- datos[,"diabetes"]

set.seed(1)
mod.lasso.tun3_cuad <- cv.glmnet(Xmod3, Ymod3, nfolds = 5, type.measure = "class", 
                                 relax = FALSE, family = "binomial", 
                                 nlambda = 50)
mod.lasso.tun3_cuad$lambda.min
coef(mod.lasso.tun3_cuad, s = "lambda.min") #Regla final que va a salir a produccion.

logit_lasso_cuad=predict(mod.lasso.tun3_cuad, newx = Xmod3, type = "response", s = "lambda.min")
res_lasso_cuad=ifelse(logit_lasso_cuad>=.5,levels(datos$diabetes)[2],levels(datos$diabetes)[1])
metrics_summary(obs = datos$diabetes, pred = res_lasso_cuad, metrics_list=c("accuracy", 
                              "recall", "specificity"),type = 'classification')
# Los indices para el modelo son accuracy=0.7831, recall=0.5000 y specificity 0.9236


#ESTIMACION PODER PREDICTIVO
mod1RHM_sVar_cuad=function(x, IndTrain, Dat, forme){
  train= IndTrain[,x]
  test = (-train)
  Xmodtotal = model.matrix(forme, data=datos)[,-1]
  Xmodt = Xmodtotal[train,]
  Ymodt = Dat[train,"diabetes"]
  modtr=cv.glmnet(Xmodt, Ymodt, nfolds = 5, type.measure ="class", gamma = 0, 
                  relax = FALSE, family = "binomial", nlambda = 50)
  preda=predict(modtr, newx = Xmodtotal[test,], type = "response", s="lambda.min")
  predb=ifelse(preda>=.5,levels(Dat$diabetes)[2],levels( Dat$diabetes)[1])
  resPod=metrics_summary(obs = Dat[test,"diabetes"], pred = predb, metrics_list=
                           c("accuracy", "recall", "specificity"),type = 
                           'classification')
  return(resPod[,2])
}

TCC.B.mod1_sVar_cuad= sapply(1:B,mod1RHM_sVar_cuad, IndTrain=Partition, Dat=datos, forme = forexp)
(TCC.RHM.mod1_sVar_cuad=rowMeans(TCC.B.mod1_sVar_cuad))
# Los indices del poder de prediccion son accuracy= 0.7769, recall=0.5276 y specificity=0.9015

#------------------------------------------------------------------------------------------------------------------------------------

# v) Explorando mas modelos lineales generalizados con selección de variables.

##############################
### Efectos principales y selecci?n por pasos usando criterio BIC
##############################

modProb <- glm(diabetes ~ ., data=datos,   family=binomial(link="probit"))
summary(modProb)
# se requiere definir la penalizaci?n para BIC
pen=log(dim(datos)[1])
# Realizamos la selecci?n por pasos con la opci?n both y empezando con mod1
modProbBIC <- stepAIC(modProb, scope =list(upper = ~., lower = ~1), trace =FALSE,direction="both", k=pen)
summary(modProbBIC) # modelo con el que se calculan probabilidades
# que son usadas en la regla final (grupo de m?xima prob u otro)

Probit=predict(modProbBIC, newdata = datos, type = "response")
res=ifelse(Probit>=.5,levels(datos$diabetes)[2],levels(datos$diabetes)[1])
metrics_summary(obs = datos$diabetes, pred = res, metrics_list=c("accuracy", "recall", "specificity"),type = 'classification')
# Los indices para el modelo son accuracy=0.7908163, recall=0.5846154 y specificity=0.8931298

#Calculo del poder predictivo
modBIC=function(x,IndTrain, Dat){
  train= IndTrain[,x]
  test = (-train)
  assign("DatosAux", Dat[train,], envir = .GlobalEnv) #Cuidado stepAIC busca la base de datos en el environment global 
  modAux=glm(diabetes ~ ., data=DatosAux,   family=binomial(link="probit"))
  penAux=log(dim(DatosAux)[1])
  modtr=stepAIC(modAux, scope =list(upper = ~., lower = ~1), trace =FALSE,direction="both", k=penAux)
  preda=predict(modtr, newdata = Dat[test,], type = "response")
  predb=ifelse(preda>=.5,levels( Dat$diabetes)[2],levels( Dat$diabetes)[1])
  resPod=metrics_summary(obs = Dat[test,"diabetes"], pred = predb, metrics_list=c("accuracy", "recall", "specificity"),type = 'classification')
  return(resPod[,2])
}

TCC.B.modBIC=matrix(NA,ncol=B,nrow=3)
for(ik in 1:B){
  TCC.B.modBIC[,ik]=modBIC(ik,IndTrain=Partition, Dat=datos)
}
(TCC.RHM.modBIC=rowMeans(TCC.B.modBIC))
# Los indices del poder de prediccion son accuracy= 0.7776923, recall=0.5561538 y specificity=0.8884615

#---------------------------------------------------------------------------------------------------------------------------------------------------------

# Explorando los modelos: naive classifier, LDA, QDA y K-NN.

################################################################################
### Explorando Naive Bayes
################################################################################

naive_model <- naiveBayes(diabetes ~ ., data = datos)
TCC.B.naive <- sapply(1:B, function(x) {
  train <- Partition[, x]
  test <- -train
  calc_poder_predictivo(naive_model, datos[train, ], datos[test, ])
})
(TCC.RHM.naive <- rowMeans(TCC.B.naive))

#Los indices de poder predictivo de N-B son accuracy=0.7761538 ,recall=0.6476923 y specificity= 0.8403846

################################################################################
### Explorando LDA
################################################################################

###### Entranamiento
modLDA <- lda(diabetes ~ ., datos)
modLdaP=predict(modLDA, datos)

logitLda=modLdaP$posterior[,2] #probabilidades
resLda=modLdaP$class  # asignaci?n a clase de m?xima prob (con punto de corte .5)
resLdab=ifelse(logitLda>=.5,levels(datos$diabetes)[2],levels(datos$diabetes)[1])
sum(resLda!=resLdab)

# Medidas aparentes de poder predictivo  
metrics_summary(obs = datos$diabetes, pred = resLda, metrics_list=c("accuracy", "recall"
                                                                    , "specificity"),type = 'classification')
# Los indices con los datos de entrenamiento son accuracy= 0.7857143, recall=0.5846154 y specificity=0.8854962

###### Medicion del poder predictivo
modRHM_Lda=function(x, IndTrain, Dat){
  train= IndTrain[,x]
  test = (-train)
  modt = lda(diabetes ~ ., Dat[train,])
  modpt = predict(modt, Dat[test,])
  predb=modpt$class
  resPod=metrics_summary(obs = Dat[test,"diabetes"], pred = predb, metrics_list=
                           c("accuracy", "recall", "specificity"),type = 'classification')
  return(resPod[,2])
}

TCC.B.mod_LDA= sapply(1:B,modRHM_Lda, IndTrain=Partition, Dat=datos)
(TCC.RHM.mod_LDA=rowMeans(TCC.B.mod_LDA))
# Los indices del poder de prediccion son accuracy= 0.7830769, recall=0.5707692 y specificity=0.8892308


################################################################################
### Explorando QDA
################################################################################

###### Entrenamiento
modQDA <- qda(diabetes ~ ., datos)
modQDAp=predict(modQDA, datos)
resQDA=modQDAp$class
# Medidas aparentes de poder predictivo  
metrics_summary(obs = datos$diabetes, pred = resQDA, metrics_list=c("accuracy", "recall", "specificity"),type = 'classification')
# Los indices con los datos de entrenamiento son accuracy=0.8061, recall=0.6615 y specificity=0.8778

###### Medicion del poder predictivo

modRHM_QDA=function(x, IndTrain, Dat){
  train= IndTrain[,x]
  test = (-train)
  modt <- qda(diabetes ~ ., Dat[train,])
  modpt=predict(modt, Dat[test,])
  predb=modpt$class
  resPod=metrics_summary(obs = Dat[test,"diabetes"], pred = predb, metrics_list=
                           c("accuracy", "recall", "specificity"),type = 'classification')
  return(resPod[,2])
}

TCC.B.mod_QDA= sapply(1:B,modRHM_QDA, IndTrain=Partition, Dat=datos)
(TCC.RHM.mod_QDA=rowMeans(TCC.B.mod_QDA))
# Los indices del poder de prediccion son accuracy= 0.7710, recall=0.5892 y specificity=0.8619


################################################################################
### Explorando KNN con variables binarias
################################################################################

# En este metodo es preferible que los datos esten en una escala parecida
# ya que se trabaja con disimiladirades, por lo que verificamos que en efecto
# los datos esten en la misma escala. Veamos como se ven: 

X11()
ggpairs(data=datos, title="Datos")
summary(datos[-9])
sqrt(var(datos$insulin))

# Parece ser que la escala de los datos es muy distinta en algunos parametros,
# Por lo que conviene estandarizar los datos. Nosotros aplicaremos una transfor-
# macion 0 1

X11()
ggpairs(data=datos, title="Datos")
summary(datos[-9])
sqrt(var(datos$insulin))
# Parece ser que la escala de los datos es muy distinta en algunos parametros,
# Por lo que conviene estandarizar los datos. Nosotros aplicaremos una transfor-
# macion 0 1

# Calcular la media de cada columna
medias <- colMeans(datos[-9])
# Calcular la desviación estándar de cada columna
desviaciones <- apply(datos[-9], 2, sd)
# Normalizar los datos restando la media y dividiendo por la desviación estándar
datos_normalizados <- as.data.frame(scale(datos[-9], center = medias, scale = desviaciones))

x11()
ggpairs(data=datos_normalizados,title="Datos normalizados")
#Como vemos de la grafica anterior los datos parecen ya estar en la misma escala

datos_normalizados$diabetes <- datos$diabetes


####### ENTRENAMIENTO DEL MODELO 
Xmod8 <- model.matrix(diabetes~ ., data=datos_normalizados)[,-1]
Ymod8 <- datos_normalizados[,"diabetes"] 

# Este metodo necesita un valor del hiperparametro k, que representa el numero 
# de vecinos que se va a comparar. Vamos a tuneralo.

#Tuneo, se exploraran 60 valores de k
set.seed(123)
knn.cross <- tune.knn(x = Xmod8, y = Ymod8, k = 1:60,tunecontrol=tune.control(sampling = "cross"), cross=5)
summary(knn.cross)
plot(knn.cross)
# Mejor valor de k a usar 
knn.cross$best.parameters[[1]]
#El de k que minimiza el error fue 18

# regla a usar para nuevos datos, ya con la k tuneada
reskNN=knn(train=Xmod8, test=Xmod8, Ymod8, k = knn.cross$best.parameters[[1]], use.all = TRUE)
metrics_summary(obs = datos$diabetes, pred = reskNN, metrics_list=c("accuracy", "recall", "specificity"),type = 'classification')
# Los indices con todos los datos son accuracy= 0.7857, recall=0.5307 y specificity=0.9122


# Medici?n del poder predictivo, se debe incluir tuneo de k
modRHM_KNN=function(x, IndTrain, Dat){
  train= IndTrain[,x]
  test = (-train)
  Xmod8ttotal = model.matrix(diabetes~ ., data=Dat)[,-1]
  Xmod8t = Xmod8ttotal[train, ]
  Xmod8test = Xmod8ttotal[test, ]
  Ymod8t = Dat[train,"diabetes"] 
  knn.crosst <- tune.knn(x = Xmod8t, y = Ymod8t, k = 1:60,tunecontrol=tune.control(sampling = "cross"), cross=5)
  predb=knn(train=Xmod8t, test=Xmod8test, Ymod8t, k = knn.crosst$best.parameters[[1]], use.all = TRUE)
  resPod=metrics_summary(obs = Dat[test,"diabetes"], pred = predb, metrics_list=c("accuracy", "recall", "specificity"),type = 'classification')
  return(resPod[,2])
}

set.seed(123)
TCC.B.modKNN= sapply(1:B,modRHM_KNN, IndTrain=Partition, Dat=datos_normalizados)
(TCC.RHM.modKNN=rowMeans(TCC.B.modKNN))
# Los indices del poder de prediccion son accuracy=0.7610, recall=0.4461 y specificity=0.9184

# Los valores enteriores nos parecieron un poco bajos, vamos a ver que quedan con 
# los datos originales
set.seed(123)
TCC.B.modKNN_org= sapply(1:B,modRHM_KNN, IndTrain=Partition, Dat=datos)
(TCC.RHM.modKNN_org=rowMeans(TCC.B.modKNN_org))
# Los indices del poder de prediccion son accuracy=0.7517, recall=0.4853 y specificity=0.8850

# Probemos un modelo con una k fija, k=5, y con datos  sin transformar
knn_train <- function(x, train_data, test_data, k) {
  train_x <- train_data[, -9]
  train_y <- train_data$diabetes
  test_x <- test_data[, -9]
  pred <- knn(train = train_x, test = test_x, cl = train_y, k = k)
  res <- metrics_summary(obs = test_data$diabetes, pred = pred, metrics_list = c("accuracy", "recall", "specificity"), type = 'classification')
  return(res[, 2])
}
k <- 5  # Número de vecinos
TCC.B.knn <- sapply(1:B, function(x) {
  train <- Partition[, x]
  test <- -train
  knn_train(x, datos[train, ], datos[test, ], k)
})
(TCC.RHM.knn <- rowMeans(TCC.B.knn))

#Indices de poder predictivo para el K-NN son accuracy=0.7179487, recall=0.4692308 y specificity=0.8423077




#----------------------------------------------------------------------------------------------------------------------------------------------------

Explore el modelo de entrenamiento random forest con 200 árboles, pero tuneando el hiper-parámetro mtry.

# Convertir la variable de respuesta a factor si no lo es
datos$diabetes <- as.factor(datos$diabetes)

# Definir una función para realizar el ajuste y la evaluación del random forest
randomForestRHM <- function(x, IndTrain, Dat) {
  train = IndTrain[,x]
  test = (-train)
  
  # Tuneo del hiperparámetro mtry usando cross-validation
  tuneRF <- tuneRF(Dat[train, -9], Dat[train, 9], ntreeTry = 200, stepFactor = 1.5, improve = 0.01, trace = TRUE, plot = FALSE)
  best_mtry <- tuneRF[which.min(tuneRF[,2]), 1]
  
  # Ajuste del modelo con el mejor mtry encontrado
  rf_model <- randomForest(diabetes ~ ., data = Dat[train, ], ntree = 200, mtry = best_mtry)
  
  # Predicciones en el conjunto de prueba
  pred <- predict(rf_model, newdata = Dat[test, ])
  
  # Evaluación del modelo
  resPod <- metrics_summary(obs = Dat[test, "diabetes"], pred = pred, metrics_list = c("accuracy", "recall", "specificity"), type = 'classification')
  
  return(resPod[, 2])
}

# Calcular el poder predictivo del modelo Random Forest
TCC.B.rf <- sapply(1:B, randomForestRHM, IndTrain = Partition, Dat = datos)
(TCC.RHM.rf = rowMeans(TCC.B.rf))


#los indices de poder predictivo de random forest son accuracy=0.7861538, recall=0.5992308, specificity=0.8796154

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Los indices del valor predictivo de cada modelo son:
(TCC.RHM.mod1=rowMeans(TCC.B.mod1))
(TCC.RHM.mod2_int=rowMeans(TCC.B.mod2_int))
(TCC.RHM.mod3_cuad=rowMeans(TCC.B.mod3_cuad))
(TCC.RHM.mod1_sVar=rowMeans(TCC.B.mod1_sVar))
(TCC.RHM.mod1_sVar_int=rowMeans(TCC.B.mod1_sVar_int))
(TCC.RHM.mod1_sVar_cuad=rowMeans(TCC.B.mod1_sVar_cuad))
(TCC.RHM.modBIC=rowMeans(TCC.B.modBIC))
(TCC.RHM.mod_LDA=rowMeans(TCC.B.mod_LDA))
(TCC.RHM.mod_QDA=rowMeans(TCC.B.mod_QDA))
(TCC.RHM.modKNN=rowMeans(TCC.B.modKNN))
(TCC.RHM.rf = rowMeans(TCC.B.rf))





