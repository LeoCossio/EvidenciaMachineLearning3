# Evidencia 3 Redes neuronales
# Leonardo Cossío Dinorín

# Linear_V1:
# Código que utiliza el modelo de Linear_SVC para clasificar
# noticias verdaderas y falsas.
# Este modelo muestra un claro sobreajuste en los datos de entrenamiento.

# Dataset obtenido de:
# https://github.com/lutzhamel/fake-news

# Librerías
import pandas as pd # Visualización y manipulación de datos
import matplotlib.pyplot as plt # Gráficas y visualización de datos
import numpy as np # Arrays y operaciones matemáticas

from sklearn.model_selection import train_test_split # Separación del dataset
from sklearn.feature_extraction.text import TfidfVectorizer # Conversión de texto a número
from sklearn.svm import LinearSVC # Clasificador
from sklearn.pipeline import Pipeline # Constructor de pipeline
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix, ConfusionMatrixDisplay # Métricas
from sklearn.model_selection import learning_curve


# Cargar datos
data = pd.read_csv("fake_or_real_news.csv")

# Preprocesamiento de los datos
data["fake"] = data["label"].apply(lambda x: 0 if x == "REAL" else 1) # One-Hot encoding
data = data.drop("label", axis=1) # Elimina la columna de label
X, y = data["text"], data["fake"] # Separa los datos de entrada y de salida

print("\n********** DIVISIÓN DEL DATASET *********************")
print(f"Dimensiones del dataset COMPLETO: {X.shape}")

# Dividir el dataset en train, validation y test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 20% test del dataset completo
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42) # 20% de train será para validation

print(f"Dimensiones del dataset de entrenamiento: {X_train.shape}")
print(f"Dimensiones del dataset de validación: {X_valid.shape}")
print(f"Dimensiones del dataset de prueba: {X_test.shape}")
print("*****************************************************\n")

# Crear el pipeline
pipeline = Pipeline([
    # Convierte el texto en una matriz numérica utilizando el 
    # valor de TF-IDF (Term Frequency-Inverse Document Frequency),
    # que refleja la importancia de cada término en un documento en 
    # relación con el conjunto de documentos.
    ('tfidf', TfidfVectorizer(stop_words="english", max_df=0.7)),  # TF-IDF
    ('l_svc', LinearSVC(random_state=42))  # Clasificador seleccionado
    # Por default tiene penalty="l2" y C=1.0
])

# Entrenamiento del modelo
pipeline.fit(X_train, y_train)

y_train_pred = pipeline.predict(X_train)
y_valid_pred = pipeline.predict(X_valid)
y_test_pred = pipeline.predict(X_test)

ac_train = accuracy_score(y_train, y_train_pred)
ac_valid = accuracy_score(y_valid, y_valid_pred)
ac_test = accuracy_score(y_test, y_test_pred)

# -------- EVALUACIÓN DEL MODELO ---------------------------------

#Evaluación en el conjunto de entrenamiento
train_f1 = f1_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)

print("\n***************** TRAIN SCORE ***********************")
print(f"Accuracy on train set: {ac_train:.2f}")
print(f"Recall on train set: {train_recall:.2f}")
print(f"Precision on train set: {train_precision:.2f}")
print(f"F1 score on train set: {train_f1:.2f}")
print("*******************************************************\n")

# Evaluación en el conjunto de validación
y_valid_pred = pipeline.predict(X_valid) # Predicciones
valid_accuracy = accuracy_score(y_valid, y_valid_pred)
valid_f1 = f1_score(y_valid, y_valid_pred)
valid_recall = recall_score(y_valid, y_valid_pred)
valid_precision = precision_score(y_valid, y_valid_pred)

print("\n***************** VALIDATION SCORE ********************")
print(f"Accuracy on validation set: {valid_accuracy:.2f}")
print(f"Recall on validation set: {valid_recall:.2f}")
print(f"Precision on validation set: {valid_precision:.2f}")
print(f"F1 score on validation set: {valid_f1:.2f}")
print("*******************************************************\n")

# Evaluación en el conjunto de prueba
y_test_pred = pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)

print("\n***************** TEST SCORE **************************")
print(f"Accuracy on test set: {test_accuracy:.2f}")
print(f"Recall on test set: {test_recall:.2f}")
print(f"Precision on test set: {test_precision:.2f}")
print(f"F1 score on test set: {test_f1:.2f}")
print("*******************************************************\n")

# --------------- Matrices de confusión --------------------------------

# Crear matrices de confusión
confusion_train = confusion_matrix(y_train, y_train_pred)
confusion_valid = confusion_matrix(y_valid, y_valid_pred)
confusion_test = confusion_matrix(y_test, y_test_pred)

# Graficar y guardar la matriz de confusión para el conjunto de entrenamiento
disp_train = ConfusionMatrixDisplay(confusion_matrix=confusion_train, display_labels=['Real', 'Fake'])
disp_train.plot(cmap='Blues')
plt.title('Matriz de Confusión - Train')
plt.savefig('matriz_m1_train.png')  # Guardar la imagen con el nombre especificado
plt.close()  # Cerrar la figura para liberar memoria

# Graficar y guardar la matriz de confusión para el conjunto de validación
disp_valid = ConfusionMatrixDisplay(confusion_matrix=confusion_valid, display_labels=['Real', 'Fake'])
disp_valid.plot(cmap='Blues')
plt.title('Matriz de Confusión - Validation')
plt.savefig('matriz_m1_valid.png')  # Guardar la imagen con el nombre especificado
plt.close()  # Cerrar la figura para liberar memoria

# Graficar y guardar la matriz de confusión para el conjunto de prueba
disp_test = ConfusionMatrixDisplay(confusion_matrix=confusion_test, display_labels=['Real', 'Fake'])
disp_test.plot(cmap='Blues')
plt.title('Matriz de Confusión - Test')
plt.savefig('matriz_m1_test.png')  # Guardar la imagen con el nombre especificado
plt.close()  # Cerrar la figura para liberar memoria

# --------------- Curva de aprendizaje ---------------------------------

# Definir un rango de tamaños del conjunto de entrenamiento
train_sizes, train_scores, valid_scores = learning_curve(
    pipeline, X_train, y_train, 
    train_sizes=np.linspace(0.1, 1.0, 10), 
    scoring='accuracy', 
    n_jobs=-1
)

# Calcular el promedio y la desviación estándar en cada punto
train_scores_mean = np.mean(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)

# Graficar las curvas de aprendizaje
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Accuracy en Entrenamiento', color='blue')
plt.plot(train_sizes, valid_scores_mean, label='Accuracy en Validación', color='green')

# Añadir bandas de desviación estándar
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color='blue')
plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.2, color='green')

plt.title('Curvas de Aprendizaje')
plt.xlabel('Tamaño del conjunto de entrenamiento')
plt.ylabel('Accuracy')
plt.ylim(0.0, 1.02)
plt.legend()
plt.grid()
plt.savefig('curva_m1.png')
plt.close()