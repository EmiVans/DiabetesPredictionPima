
# DiabetesPredictionPima
Fitting Supervised Learning Models for Diabetes Detection Using R

## Clasificación supervisada

La detección temprana y precisa de la diabetes es de suma importancia para mejorar la calidad de vida de los pacientes y reducir el riesgo de complicaciones graves asociadas con esta enfermedad. La diabetes no diagnosticada o mal controlada puede llevar a una serie de problemas de salud, incluyendo enfermedades cardiovasculares, daño renal, problemas de visión y neuropatía. Por lo tanto, desarrollar métodos efectivos para predecir la presencia de diabetes utilizando datos clínicos es un objetivo vital en la investigación médica y de salud pública.

En este estudio, utilizamos el conjunto de datos PimaIndiansDiabetes2 del paquete `mlbench` en R, que contiene información clínica sobre pacientes indígenas de Pima. El objetivo es usar las ocho variables clínicas observadas en las pacientes para predecir la presencia o no de diabetes (variable diabetes).

Realizamos dos análisis visuales sobre las variables predictoras, distinguiendo entre los grupos a clasificar: pacientes con y sin diabetes. Este análisis es crucial, ya que nos permite identificar de manera preliminar las variables que posiblemente tengan mayor influencia en el modelo predictivo.

![Imagenes](Plot 2.1 de la tarea 3 del seminario.jpeg)
*En general, como se muestra en la figura, las medias de las variables entre los grupos con y sin diabetes son similares. Sin embargo, se puede ver que las diferencias son más evidentes en las variables 'glucose', 'mass', 'age' y 'pregnant', donde un mayor valor tiende a correlacionarse con un mayor riesgo de diabetes. Por otro lado, las variables 'triceps' y 'pressure' no presentan diferencias significativas entre los grupos, sugiriendo que no están directamente relacionadas con la presencia de diabetes.*

El análisis de componentes principales en la figura 2 revela una correlación positiva del grupo rojo con todas las variables de estudio. Las correlaciones más fuertes se observan con 'pressure', 'glucose' e 'insulin', lo que confirma que mayores niveles de glucosa e insulina están asociados con un mayor riesgo de diabetes. Además, la presión arterial también muestra una correlación significativa con la presencia de diabetes en este análisis.

![Figura 2](Biplot%20ejer.%202.2%20de%20T.%203%20del%20seminario.jpeg)

La tabla a continuación proporciona un resumen detallado de los diversos modelos explorados en este estudio; los cuales son modelos lineales generalizados, el clasificador ingenuo, LDA, QDA, K-NN y random forest. Además, se muestran los índices de desempeño de los métodos de entrenamiento.

| Modelo explorado | Variables | Hiperparámetro | Accuracy | Recall | Specificity |
|------------------|-----------|----------------|----------|--------|-------------|
| Modelo lineal generalizado con distribución "binomial" y liga "logit" y con efectos principales | Todas las variables | Sin hiperparámetro | 0.7841 | 0.5676 | 0.8923 |
| Modelo lineal generalizado con interacciones y distribución "binomial" con liga "logit" | Todas las variables | Sin hiperparámetro | 0.7402 | 0.5438 | 0.8384 |
| Modelo lineal generalizado con variables al cuadrado y distribución "binomial" con liga "logit" | Todas las variables | Sin hiperparámetro | 0.7774 | 0.5869 | 0.8726 |
| Modelo lineal generalizado con efectos principales, distribución "binomial", liga "logit" y con selección lasso | pregnant, glucose, triceps, mass, pedigree, age | 0.0307 | 0.7764 | 0.52 | 0.9046 |
| Modelo lineal generalizado con distribución "binomial", liga "logit", con interacciones y selección lasso | glucose, mass, pedigree, pregnant, insulin, triceps, age | 0.00682 | 0.7766 | 0.5238 | 0.9030 |
| Modelo lineal generalizado con variables al cuadrado, distribución "binomial", liga "logit" y con selección lasso | glucose, mass, pedigree, age, pregnant | 0.0654 | 0.7769 | 0.5276 | 0.9015 |
| Modelo lineal generalizado con efectos principales, distribución "binomial", liga "probit" y con selección BIC | glucose, mass, age | Sin hiperparámetro | 0.7776 | 0.5561 | 0.8884 |
| Modelo naive classifier | Todas las variables | Sin hiperparámetro | 0.7780612 | 0.6538462 | 0.8396947 |
| QDA | Todas las variables | Sin hiperparámetro | 0.7710 | 0.5892 | 0.8619 |
| LDA | Todas las variables | Sin hiperparámetro | 0.7830 | 0.5707 | 0.8892 |
| KNN con los datos estandarizados | Todas las variables | k=18 | 0.7610 | 0.4461 | 0.9184 |
| Random Forest | Todas las variables | mtry=6, ntree=200, nd=1 | 0.7866 | 0.5976 | 0.8811 |

Viendo la tabla, se aprecia una tendencia consistente en todos los modelos analizados: la especificidad, que indica los verdaderos negativos, muestra resultados consistentemente superiores a 0.8, mientras que el recall, que representa la sensibilidad, tiende a ser menor, con valores casi siempre inferiores a 0.6. Esta disparidad puede explicarse por la desproporción en la distribución de la población del estudio: la cantidad de personas sin diabetes fue significativamente menor que aquellas con la enfermedad. Como resultado, los modelos pueden haber aprendido más efectivamente a identificar casos negativos, donde la enfermedad no está presente, mientras que la detección de casos positivos puede haber sido más desafiante debido a su menor representación en los datos de entrenamiento.

Además, se destaca que las variables 'glucose', 'age', 'mass', 'pregnant' y 'pedigree' son las que aparecen con mayor frecuencia en nuestros modelos. Esto sugiere que estas variables poseen un mayor impacto en la predicción de la presencia de diabetes, ya sea debido a su fuerte correlación con la enfermedad o a su importancia clínica en el diagnóstico y manejo de la misma.

Para seleccionar el modelo más adecuado, priorizamos aquel que lograra un equilibrio entre una buena especificidad y sensibilidad. Esto se debe a la importancia de evitar tanto los falsos positivos, que pueden generar ansiedad, como los falsos negativos, ya que es crucial detectar la diabetes a tiempo. Después de evaluar varias opciones, nuestro modelo elegido fue el random forest, ya que tuvo la segunda mejor sensibilidad y una especificidad aceptable.

Nuestro modelo considera todas las variables del conjunto de datos. Sin embargo, destacan la glucosa, edad e insulina, con valores de importancia de 37.1172, 21.8725 y 20.1775 respectivamente. Los hiperparámetros utilizados fueron `mtry=6`, lo que indica que en cada nodo, el algoritmo considera 6 características seleccionadas aleatoriamente para buscar el mejor punto de división. Además, se emplearon 200 árboles de decisión y un `nodesize` de 1 para nuestra variable categórica.

En resumen, el proceso general implica generar múltiples muestras de entrenamiento con reemplazo a partir del conjunto de datos PimaIndiansDiabetes2. Para cada muestra, se crea un árbol de decisión, y finalmente se utiliza la votación para realizar la predicción final. Este enfoque fue elegido por su poder predictivo, que alcanzó un valor de 0.7866667.

A pesar de que nuestro modelo alcanza una tasa global de clasificación del 78%, es importante destacar que esta cifra puede ser engañosa. Si bien la TCC global del modelo es aceptable, su tasa de sensibilidad es considerablemente baja. Esto significa que el modelo tiene dificultades para identificar correctamente los casos positivos de diabetes. Es fundamental tener en cuenta este aspecto al interpretar la eficacia del modelo en la práctica clínica.
