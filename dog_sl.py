# -*- coding: utf-8 -*-
"""
Created on Tue May 17 21:38:41 2022

@author: steve
"""
import tensorflow as tf
import streamlit as st
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image
from keras.preprocessing.image import img_to_array
import streamlit as st
import os 
import numpy as np 
import cv2
import matplotlib.pyplot as plt

st.title('DOG RECOGNICION')
st.image('dog.jpg')
st.write('Performance du modele:')


mode = st.sidebar.radio(
     "choisissez un modele ? ",
     ('InceptionResNetV2', 'Xception','VGG16'))

options = st.sidebar.multiselect(
     'que voulez-vous afficher ?',
     ['performance', 'liste race', 'photo race'])



if mode == 'InceptionResNetV2':
    name_folder = ['Chihuahua', 'French_bulldog', 'German_shepherd', 'Great_Pyrenees', 'Labrador_retriever', 'Pomeranian', 'Rottweiler', 'Samoyed', 'Shetland_sheepdog', 'Tzu', 'Yorkshire_terrier', 'beagle', 'boxer', 'cocker_spaniel', 'golden_retriever', 'malinois', 'pug']
    if options  == 'liste race':
        st.write(name_folder)
    if options  == 'photo race':
        st.image('Race_image.png')
    
    uploaded_image = st.file_uploader("Choose a file",key=3)
    if uploaded_image:   
        if options  == 'performance':      
            st.image('fig25_InceptionResNetV2.png')        
        model_2 = tf.keras.models.load_model('my_model_25_InceptionResNetV2.h5')
        #liste_jerem = ['beagle', 'boxer', 'Chihuahua', 'cocker_spaniel', 'French_bulldog', 'German_shepherd', 'golden_retriever', 'Great_Pyrenees', 'Labrador_retriever', 'malinois']
        name_folder = ['Chihuahua', 'French_bulldog', 'German_shepherd', 'Great_Pyrenees', 'Labrador_retriever', 'Pomeranian', 'Rottweiler', 'Samoyed', 'Shetland_sheepdog', 'Tzu', 'Yorkshire_terrier', 'beagle', 'boxer', 'cocker_spaniel', 'golden_retriever', 'malinois', 'pug']
        if options  == 'liste race':
            st.write(name_folder)
        if options  == 'photo race':
            st.image('Race_image.png')
    #st.write(liste_jerem)
        pictures =  Image.open(uploaded_image)
        pictures.save('new_image.jpeg')
        Image.open("new_image.jpeg").save("new_image.bmp")
        img = tf.keras.preprocessing.image.load_img("new_image.bmp")#convert to bmp
        st.image(img)
    #prepro
    
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.keras.preprocessing.image.smart_resize(img, (150, 150))
        img = tf.reshape(img, (-1, 150, 150, 3))
        prediction = model_2.predict(img/255)
        st.write(np.argmax(prediction))
        face_cascade=cv2.CascadeClassifier('mydogdetector.xml')    
        img=cv2.imread("new_image.bmp")
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        font=cv2.FONT_HERSHEY_SIMPLEX
        faces=face_cascade.detectMultiScale(gray,1.345,5,75)
    
        st.write('La race du chien est surement :',name_folder[np.argmax(prediction)]) 
        i = 0
        for(x,y,w,h) in faces:        
            img=cv2.rectangle(img,(x,y),(x+w+20,y+h+20),(0,255,0),2)
            cv2.putText(img,name_folder[np.argmax(prediction)],(x+50,y-10),font,0.9,(0,255,0),2)
            i +=1
            if i > 1:
                break
        i=0
        p,l,m=cv2.split(img)
        img_1=cv2.merge([m,l,p])
        st.image(img_1)        
        
        
    
        st.write(np.argmax(prediction))
        st.write('La race du chien est surement :',name_folder[np.argmax(prediction)]) 
    
if mode == 'Xception':
    uploaded_image = st.file_uploader("Choose a file",key=2)
    
    if uploaded_image:
        if options  == 'performance':
            st.image('fig10_Xception.png')
        
        model_2 = tf.keras.models.load_model('my_model_10_Xception.h5')
        #liste_jerem = ['beagle', 'boxer', 'Chihuahua', 'cocker_spaniel', 'French_bulldog', 'German_shepherd', 'golden_retriever', 'Great_Pyrenees', 'Labrador_retriever', 'malinois']
        name_folder = ['Chihuahua', 'French_bulldog', 'German_shepherd', 'Great_Pyrenees', 'Labrador_retriever', 'Pomeranian', 'Rottweiler', 'Samoyed', 'Shetland_sheepdog', 'Tzu', 'Yorkshire_terrier', 'beagle', 'boxer', 'cocker_spaniel', 'golden_retriever', 'malinois', 'pug']
        if options  == 'liste race':
            st.write(name_folder)
        if options  == 'photo race':
            st.image('Race_image.png')
    #st.write(liste_jerem)
        pictures =  Image.open(uploaded_image)
        pictures.save('new_image.jpeg')
        Image.open("new_image.jpeg").save("new_image.bmp")
        img = tf.keras.preprocessing.image.load_img("new_image.bmp")#convert to bmp
        st.image(img)
    #prepro
    
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.keras.preprocessing.image.smart_resize(img, (150, 150))
        img = tf.reshape(img, (-1, 150, 150, 3))
        prediction = model_2.predict(img/255)
        st.write(np.argmax(prediction))
        face_cascade=cv2.CascadeClassifier('mydogdetector.xml')    
        img=cv2.imread("new_image.bmp")
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        font=cv2.FONT_HERSHEY_SIMPLEX
        faces=face_cascade.detectMultiScale(gray,1.345,5,75)
    
        st.write('La race du chien est surement :',name_folder[np.argmax(prediction)]) 
        i = 0
        for(x,y,w,h) in faces:        
            img=cv2.rectangle(img,(x,y),(x+w+20,y+h+20),(0,255,0),2)
            cv2.putText(img,name_folder[np.argmax(prediction)],(x+50,y-10),font,0.9,(0,255,0),2)
            i +=1
            if i > 2:
                break
        i=0
        p,l,m=cv2.split(img)
        img_1=cv2.merge([m,l,p])
        st.image(img_1)        
        
        
    
        st.write(np.argmax(prediction))
        st.write('La race du chien est surement :',name_folder[np.argmax(prediction)]) 
    
elif mode == 'VGG16':
    uploaded_image = st.file_uploader("Choose a file", key=1)

    if uploaded_image:
        if options  == 'performance':
            st.image('fig15_VGG16.png')
        
        model_2 = tf.keras.models.load_model('my_model_15_VGG16.h5')
        #liste_jerem = ['beagle', 'boxer', 'Chihuahua', 'cocker_spaniel', 'French_bulldog', 'German_shepherd', 'golden_retriever', 'Great_Pyrenees', 'Labrador_retriever', 'malinois']
        name_folder = ['Chihuahua', 'French_bulldog', 'German_shepherd', 'Great_Pyrenees', 'Labrador_retriever', 'Pomeranian', 'Rottweiler', 'Samoyed', 'Shetland_sheepdog', 'Tzu', 'Yorkshire_terrier', 'beagle', 'boxer', 'cocker_spaniel', 'golden_retriever', 'malinois', 'pug']
        if options  == 'liste race':
            st.write(name_folder)
        if options  == 'photo race':
            st.image('Race_image.png')
    #st.write(liste_jerem)
        pictures =  Image.open(uploaded_image)
        pictures.save('new_image.jpeg')
        Image.open("new_image.jpeg").save("new_image.bmp")
        img = tf.keras.preprocessing.image.load_img("new_image.bmp")#convert to bmp
        st.image(img)
    #prepro
    
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.keras.preprocessing.image.smart_resize(img, (150, 150))
        img = tf.reshape(img, (-1, 150, 150, 3))
        prediction = model_2.predict(img/255)
        st.write(np.argmax(prediction))
        face_cascade=cv2.CascadeClassifier('mydogdetector.xml')    
        img=cv2.imread("new_image.bmp")
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        font=cv2.FONT_HERSHEY_SIMPLEX
        faces=face_cascade.detectMultiScale(gray,1.345,5,75)
    
        st.write('La race du chien est surement :',name_folder[np.argmax(prediction)]) 
        i = 0
        for(x,y,w,h) in faces:        
            img=cv2.rectangle(img,(x,y),(x+w+20,y+h+20),(0,255,0),2)
            cv2.putText(img,name_folder[np.argmax(prediction)],(x+50,y-10),font,0.9,(0,255,0),2)
            i +=1
            if i > 2:
                break
        i=0
        p,l,m=cv2.split(img)
        img_1=cv2.merge([m,l,p])
        st.image(img_1)        
        
        
    
        st.write(np.argmax(prediction))
        st.write('La race du chien est surement :',name_folder[np.argmax(prediction)]) 
    
            