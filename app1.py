
import io
import  streamlit as st
import json
import numpy as np
import requests 
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

st.set_option('deprecation.showfileUploaderEncoding', False)

URI = 'http://127.0.0.1:5000'

st.title('Nerual Network Visualizer')
st.sidebar.markdown('## Input Image')

if st.button('Get random predictions'):
    response = requests.post(URI, data={})
    response = json.loads(response.text)
    preds = response.get('prediction')
    image = response.get('image')
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
