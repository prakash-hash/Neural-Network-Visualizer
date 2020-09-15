
import tensorflow as tf 
import random
import  streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageOps
import easygui

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

def get_predictions_user_image(image):
    image_arr = np.reshape(image, (1, 784))
    return feature_model.predict(image_arr)

# if __name__ == '__main__':
#     app.run()
    e97171
st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Nerual Network Visualizer')
st.markdown('<style>h1{color: #e97171;}</style>', unsafe_allow_html=True)
st.sidebar.markdown('<h1 style="color:#e97171">Input Image</h1>', unsafe_allow_html=True)

def binarize(image):
    image = cv2.imread(image)
    grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    # Convert RGB to BGR 
    grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
    _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    preprocessed_digits = []

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)

        # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
        cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)

        # Cropping out the digit from the image corresponding to the current contours in the for loop
        digit = thresh[y:y+h, x:x+w]

        # Resizing that digit to (18, 18)
        resized_digit = cv2.resize(digit, (18,18))

        # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
        padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)

        # Adding the preprocessed digit to the list of preprocessed digits
        preprocessed_digits.append(padded_digit)
    return preprocessed_digits[0]/255.0

get_random = True

def get_random():
    get_random = True
    preds, image = get_predictions()
    final_preds = [i.tolist() for i in preds]
    
    pred_dict = {
            'prediction':final_preds,
            'image': image.tolist()
        }
    
    preds = pred_dict['prediction']
    image = pred_dict['image']
    image = np.reshape(image, (28,28))
    st.sidebar.image(image, width=150)
    
    for layer, p in enumerate(preds):
        
        numbers = np.squeeze(np.array(p)) #squeezing to removing the single demension
        
        #creating layers
        plt.figure(figsize=(32,4))
        
        if layer == 2: # final output layer
            row = 1   
            col = 10
        else:          # dense layer
            row = 2
            col = 16
            
        for i, number in enumerate(numbers): 
            plt.subplot(row,col,i + 1) #subploting every node
            plt.imshow(number*np.ones((8, 8, 3)).astype('float32')) #creating 8x8 pixel dimension blocks
            plt.xticks([])
            plt.yticks([])
            
            if layer == 2:
                plt.xlabel(str(i), fontsize = 40) #labels for the output layer
        plt.subplots_adjust(wspace=0.05, hspace=0.05) #packing the nodes
        plt.tight_layout()
        st.text('Layer {}'.format(layer + 1))
        st.pyplot()
    st.markdown('## Final Prediction = {}'.format(np.argmax(preds[2])))

def get_random_u(file_upload):
    b_image = binarize(file_upload)
    st.sidebar.image(b_image, width = 150)
    preds = get_predictions_user_image((b_image))
    final_preds = [i.tolist() for i in preds]

    pred_dict = {
            'prediction':final_preds,
        }

    preds = pred_dict['prediction']

    for layer, p in enumerate(preds):

        numbers = np.squeeze(np.array(p)) #squeezing to removing the single demension

        #creating layers
        plt.figure(figsize=(32,4))

        if layer == 2: # final output layer
            row = 1   
            col = 10
        else:          # dense layer
            row = 2
            col = 16

        for i, number in enumerate(numbers): 
            plt.subplot(row,col,i + 1) #subploting every node
            plt.imshow(number*np.ones((8, 8, 3)).astype('float32')) #creating 8x8 pixel dimension blocks
            plt.xticks([])
            plt.yticks([])

            if layer == 2:
                plt.xlabel(str(i), fontsize = 40) #labels for the output layer
        plt.subplots_adjust(wspace=0.05, hspace=0.05) #packing the nodes
        plt.tight_layout()
        st.text('Layer {}'.format(layer + 1))
        st.pyplot()
    st.markdown('## Final Prediction = {}'.format(np.argmax(preds[2])))

file_upload = None
if st.button('Get random predictions'):
    get_random()
    file_upload = None

if st.button('Upload'):
    file_upload = easygui.fileopenbox()
    get_random_u(file_upload)
