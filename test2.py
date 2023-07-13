# load json and create model
import cv2
import numpy as np
import pandas as pd
from keras.saving.legacy.model_config import model_from_json
from sklearn.preprocessing import LabelEncoder

INPUT_DIR = '../input/pokemon-images-and-types'

pokemon = pd.read_csv('pokemon.csv')
pokemon = pokemon.sort_values(by='Name')

y = pokemon['Type1'].unique()

print(y)


encoder = LabelEncoder()

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#score = loaded_model.evaluate(X, Y, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

new_image_path = 'C:/Users/italo/PycharmProjects/redes_convolucionais/images/pidgeot.png'
new_image = cv2.imread(new_image_path)
new_image = cv2.resize(new_image, (16, 16))
new_image = np.expand_dims(new_image, axis=0)

encoder.fit(y)

# Fazer a previsão usando o modelo treinado
prediction = loaded_model.predict(new_image)
predicted_label = encoder.inverse_transform(np.argmax(prediction, axis=1))

print('Predicted label:', predicted_label)