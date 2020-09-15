
import json
import tensorflow as tf
import numpy as np
import random

from flask import Flask, request

app = Flask(__name__)

#loading model
model = tf.keras.models.load_model('model.h5')

#creating feature function which we will use to display layers
feature_model = tf.keras.models.Model(
    model.input,
    [layer.output for layer in model.layers]
)

#loadind dataset again
_,(x_test, _) = tf.keras.datasets.mnist.load_data()

#normalising test case
x_test = x_test/255

#getting a random prediction
def get_predictions():
    index = np.random.choice(x_test.shape[0])
    image = x_test[index, :, :]
    image_arr = np.reshape(image, (1, 784))
    return feature_model.predict(image_arr), image


@app.route('/', methods = ['GET', 'POST']) #defining GET and POST request
def index():
    if request.method == 'POST':
        preds, image = get_predictions()
        final_preds = [i.tolist() for i in preds]
        return json.dumps({
            'prediction':final_preds,
            'image': image.tolist()
        })
    return "Welcom to the model server!"

if __name__ == '__main__':
    app.run()
