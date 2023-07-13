import os.path

import pandas as pd
import numpy as np
import cv2
import sns as sns
from keras.callbacks import LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt, pyplot
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.utils import to_categorical

pokemon = pd.read_csv('pokemon.csv')
pokemon = pokemon.sort_values(by='Name')

image_folder = 'C:/Users/italo/PycharmProjects/redes_convolucionais/images/images'

# Para redimensionar todas as imagens para um tamanho fixo
image_size = (16, 16)

images = []
labels = []


def get_type1_name(name):
    string = str(pokemon[pokemon['Name'] == name]['Type1'].values)
    return string.strip("[]'")


datagen = ImageDataGenerator(
    rotation_range=45,
    zoom_range=0.30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest')


# Definir a função de scheduler para ajustar a taxa de aprendizado
def lr_scheduler(epoch):
    if epoch < 10:
        return 0.1
    elif epoch < 20:
        return 0.01
    else:
        return 0.001


images_in_folder = os.listdir(image_folder)
for filename in images_in_folder:
    img_name = filename.split('.')[0]
    img_extension = filename.split('.')[1]
    image = cv2.imread(os.path.join(image_folder, filename))
    # redimensionar para 16x16 para treinar mais rapido
    resized_image = cv2.resize(image, image_size)
    if resized_image is not None:
        for _ in range(2):
            random_img = datagen.random_transform(image)
            random_img = cv2.resize(random_img, image_size)

            # adicionar imagem e rótulo correspondente nos dados de treinamento
            images.append(random_img)
            label = get_type1_name(img_name)
            labels.append(label)


for _, row in pokemon.iterrows():
    image_name = row['Name']
    label = row['Type1']

    for ext in ['jpg', 'png']:
        image_path = f'{image_folder}/{image_name}.{ext}'  # .jpg ou .png, dependendo do formato das imagens

        if os.path.isfile(image_path):
            # Ler a imagem usando o OpenCV e redimensioná-la
            image = cv2.imread(image_path)
            image = cv2.resize(image, image_size)

            #adicionar imagem e rótulo correspondente nos dados de treinamento
            images.append(image)
            labels.append(label)


# Converter as listas em matrizes NumPy
images = np.array(images).astype('float32')
images /= 255.0
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)
labels = to_categorical(labels)

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)


# Definindo a arquitetura da rede neural
model = Sequential()

model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=train_images.shape[1:]))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(train_labels.shape[1], activation='softmax'))

# Definir o otimizador com uma taxa de aprendizado inicial
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)

# Definir o callback do scheduler para ajustar a taxa de aprendizado durante o treinamento
scheduler = LearningRateScheduler(lr_scheduler)

# Treinar a rede neural
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=2, batch_size=264, validation_data=(test_images, test_labels), callbacks=[scheduler, EarlyStopping(monitor='loss', min_delta=1e-10, patience=20, verbose=1)])

# Avaliar o desempenho
loss, accuracy = model.evaluate(test_images, test_labels)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# salvando o modelo em JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serializar weights para HDF5
model.save_weights("model.h5")
print("Saved model to disk")














# A partir daqui apenas gerando gráficos e checando as métricas de desempenho

# Fazer a previsão usando o modelo treinado
#
# predictions = []
#
# for image in test_images:
#     image = np.expand_dims(image, axis=0)
#
#     prediction = model.predict(image)
#     #predicted_label = encoder.inverse_transform(np.argmax(prediction, axis=1))
#
#     predicted_class = np.argmax(prediction)
#
#     predictions.append(predicted_class)
#
#
# # Calcular a matriz de confusão
# confusion = confusion_matrix(test_images, predictions)
#
#
# # Plotar a matriz de confusão
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix')
# plt.show()
#
