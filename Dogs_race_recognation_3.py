# -*- coding: utf-8 -*-
"""
Created on Thu May  5 10:05:35 2022

@author: steve
"""
import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, shutil
import re
from distutils.dir_util import copy_tree
from keras.preprocessing.image import  array_to_img, load_img
from  keras.preprocessing import image
from keras.models import Sequential
from keras.layers.pooling import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from keras import models, layers 
from tensorflow.keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import streamlit as st
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image
from keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50
# separation des images en 3 pour le test, le train et la validation:
    
original_data = '/Users/steve/Documents/DL/stanford_Dogs/dog'

#creation list des chemins des races

original_list = [] # list of the original path of each folder
for i in os.walk(original_data):
    original_list.append(i[0])

original_list =  original_list[1:]
#print(original_list) 

#rename folder original
original_list_rename = []
clean_dir = '/Users/steve/Documents/DL/stanford_Dogs/portion_data/dog/beagle'
len(clean_dir)
if not os.path.exists(clean_dir):
    for i in original_list:
        new_name = re.sub('^(.*-)',"", i)
        os.rename(i,new_name)
        original_list_rename.append(new_name)
    #print(new_name)
    #print('new list',original_list_rename)

#Creation des fichiers test, train et val + copy des fichiers
           
    
def split_val_train_test(dir_src,dir_tg,train_size,test_size):
    '''
    cette fonction permet de manière automatique de copier et diviser les photos d'un dataset en
    3 dossiers pour la validation, le train et le test. Les images sont differentes 
    dans les 3 dossiers. Cela permet de ne pas avoir les mêmes images dans le train et le test
    par exemple. 

    Parameters
    ----------
    dir_src : c'est le chemin source ou se trouve les dossiers à copier attention mettre
    le chemin absolut exemlple:'/Users/steve/Documents/DL/stanford_Dogs/dog'.
    dir_tg : c'est le chemin target la ou on va copier les dossiers'    
    train_size: taille du train
    test_size: taille du test
    Returns
    ------- 
    None. 
    '''     
    #nom de vos dossiers a creer:
    list_portion_data = ['test','train','val']
    
    for j in list_portion_data:
        #creation des dossiers val,train et test et copie des fichiers
        filePath = shutil.copytree(dir_src, '{}{}'.format(dir_tg,j))
        print('je cree le fichier : {}'.format(j))
    
        directory = '{}{}'.format(dir_tg,j)
        #suppression des fichiers inutes dans le train,test et val:
        for dirpath,_,filenames in os.walk(directory):
            list_files_train = []
            list_files_test = []
            list_files_val = []
            list_files = []
            print('je cree les listes des fichiers a supprimer')
            for f in filenames:            
                gg = os.path.abspath(os.path.join(dirpath, f))
                list_files.append(gg)
                
            size = round(len(list_files)*train_size) # train
            size2 = size + round(len(list_files)*test_size) # test
            size3 = size + size2 + round(len(list_files)*(1-(train_size+test_size))) # val
            
            #affichage dans la console
            if j == 'train':
                name_dir = 59
                print('taille image train {}:'.format(dirpath[name_dir:]),size)
            elif j == 'test':
                name_dir = 58                
                print('taille image test {}:'.format(dirpath[name_dir:]), (size2-size))
            else:
                name_dir = 57                
                print('taille image val {}:'.format(dirpath[name_dir:]), (size3-size2-size))
            
            list_files_train = list_files[:size]
            list_files_test = list_files[size:size2]
            list_files_val = list_files[size2:len(list_files)]
            #creation des list de files a supprimer
            erase_files_train = list_files_test + list_files_val 
            erase_files_test = list_files_val  + list_files_train
            erase_files_val = list_files_train + list_files_test 
            #suppression des fichier s'ils existent  
            if j == 'test':           
                for i in erase_files_test:
                    if os.path.exists(i):
                        os.remove(i)
                print('je supprime le test')
     
              
            elif j == 'train':            
                for i in erase_files_train:
                    if os.path.exists(i):
                        os.remove(i)
                print('je supprime le train')
             
            elif j == 'val':            
                for i in erase_files_val:
                    if os.path.exists(i):
                        os.remove(i)
                print('je supprime le val')
            else:
                print("c'est fini")
                
dir_src = '/Users/steve/Documents/DL/stanford_Dogs/dog'
dir_tg = '/Users/steve/Documents/DL/stanford_Dogs/portion_data2/'
train_size = 0.7
test_size =  0.2

if not os.path.exists('{}/test'.format(dir_tg)):
    split_val_train_test(dir_src,dir_tg,train_size,test_size)



def train_test_split(path_train,path_test):
    batch_size = 32

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
        fill_mode='nearest')

# this is the augmentation configuration we will use for testing:
# only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data

    train_generator = train_datagen.flow_from_directory(
        path_train,  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical',
        
        )  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        path_test,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical',
        )
    x_test, y_test = next(train_generator)
    return train_generator, validation_generator, x_test, y_test, train_datagen


path_train = '/Users/steve/Documents/DL/stanford_Dogs/portion_data/train'
path_test = '/Users/steve/Documents/DL/stanford_Dogs/portion_data/test'

train_generator, validation_generator, x_test, y_test, train_datagen =  train_test_split(path_train, path_test)


#afichage d'image generer par le datagen

def image_creation(train_dir,randoms=0):
    '''
    Cette fonction permet de retourner les images creer par l'image generateur de maniere aleatoire ou avec une option input

    Parameters
    ----------
    train_dir : C'est le dossier ou se trouve les classes.
    image_number : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    ''' 
    if randoms == 1: # mode aleatoire
        list_class = os.listdir(train_dir) # list des classes dans le repertoire train
        name_class = random.choice(list_class) # choix d'une race aleatoire       
        train_dir_class = '{}/{}'.format(train_dir,name_class) #creation du chemin de la classe
        image_number_len = len(train_dir_class) # nombre d'image dans le dossier
        image_number = random.randrange(0, image_number_len, 1) # image aleatoire d'une classe a plotter
        # creation de la liste des noms des images que l'on va plotter apres
        fnames = [os.path.join(train_dir_class,fname) for fname in os.listdir(train_dir_class)]
        img_path = fnames[image_number]
    #transformation de l'image en array et redimensionnement
        img = image.load_img(img_path,target_size=(150,150))
        x = image.img_to_array(img)
        x = x.reshape((1,)+ x.shape)
    #plottage des images
        i = 0
        for batch in train_datagen.flow(x,batch_size=1):
            plt.figure(i)
            imgplot = plt.imshow(image.array_to_img(batch[0]))
            plt.savefig('img{}'.format(i))
            i += 1
            if i % 4 == 0:
                break
        print("Numero de l'image: ",image_number)
        print("Nom de la classe: ",name_class)    
        plt.show()
    else:   #mode choix de la class
        list_class = os.listdir(train_dir)
        print('Voici la liste des classes:',list_class)
        #input demande a l'utilisateur la race a afficher
        name_class = input('quelle est la classe ?')
        while name_class not in list_class:
            name_class = input('quelle est la classe ?')
        train_dir_class = '{}/{}'.format(train_dir,name_class)
        image_number_len = len(train_dir_class)
        list_number = list(range(0,image_number_len+1,1))        
        image_number = int(input('Choisissez une photo entre 0 et {} : '.format(image_number_len)))        
        while image_number not in list_number:
            image_number = int(input('Choisissez une photo entre 0 et {} :'.format(image_number_len)))        
    # creation de la liste des noms des images que l'on va plotter apres
        fnames = [os.path.join(train_dir_class,fname) for fname in os.listdir(train_dir_class)]
        img_path = fnames[image_number]
    #transformation de l'image en array et redimensionnement
        img = image.load_img(img_path,target_size=(150,150))
        x = image.img_to_array(img)
        x = x.reshape((1,)+ x.shape)
    #plottage des images
        i = 0
        for batch in train_datagen.flow(x,batch_size=1):
            plt.figure(i)            
            imgplot = plt.imshow(image.array_to_img(batch[0]))
            plt.savefig('img{}'.format(i))
            i += 1
            if i % 4 == 0:
                break
        print(image_number)
        print(name_class)    
        plt.show()
    

path_train = '/Users/steve/Documents/DL/stanford_Dogs/portion_data/train'



image_creation(path_train,1)


def make_model(layer_1,layer_2,layer_3,layer_4, dropout_1, kernel_1,kernel_2,stride,dense_2,Nb_class):
    
    model = models.Sequential()
    model.add(layers.Conv2D(layer_1,(kernel_2,kernel_2),strides=(stride, stride), activation='relu', input_shape=(150,150,3)))
    model.add(layers.MaxPooling2D(2,2))
    
    model.add(layers.Conv2D(layer_2,(kernel_2, kernel_2),strides=(stride, stride), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    
    
    model.add(layers.Conv2D(layer_3,(kernel_1, kernel_1),strides=(stride, stride), activation='relu'))
    model.add(layers.MaxPooling2D(2,2)) 
     
    
    model.add(layers.Conv2D(layer_4,(kernel_1, kernel_1),strides=(stride, stride), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))  
    
    
    model.add(layers.Flatten())  
       
    model.add(layers.Dense(dense_2,activation=('relu'))) 
    model.add(layers.Dropout(dropout_1))                       
            
    model.add(layers.Dense(Nb_class,activation=('softmax')))

    model.summary()
    return model





layer_1 = 16
layer_2 = 32
layer_3 =64
layer_4 =128
dropout_1 = 0.4
kernel_1 = 2
kernel_2 = 5
stride = 1

dense_2 = 512
Nb_class = 17 
  
#model = make_model(layer_1,layer_2,layer_3,layer_4,dropout_1 ,kernel_1,kernel_2,stride ,dense_2,Nb_class)
epochs = 25

def fit_model(model,train_generator,validation_generator,epochs):
    early = tf.keras.callbacks.EarlyStopping( patience=10,
                                          min_delta=0.001,
                                          restore_best_weights=True)
    epochs= epochs

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    history = model.fit_generator(train_generator,  
                              epochs= epochs, 
                              validation_data=validation_generator,
                              callbacks=[early]) 
                              
    return history, model


#history, model = fit_model(model,train_generator,validation_generator,epochs)

#score = model.evaluate(x_test, y_test, verbose=0)
#print("Test loss:", score[0])
#print("Test accuracy:", score[1])

def plot_scrore(history):
# print(history.history.keys())
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss =  history.history['loss']
    val_loss =  history.history['val_loss']
    epochs = range(1,len(acc)+1)
    
    fig1 = plt.plot(epochs,acc,'p',label ='entrainement')
    fig1 =plt.plot(epochs,val_acc,'g',label ='validation')
    fig1 =plt.xlabel("Epochs")
    fig1 =plt.ylabel("Loss")
    fig1 = plt.legend()
    fig1 = plt.figure()
    
    fig2 = plt.plot(epochs,loss,'b',label ='entrainement loss')
    fig2 =plt.plot(epochs,val_acc,'r',label ='validation loss')
    fig2 =plt.title('training and validation loss')
    fig2 = plt.legend()
    fig2 = plt.figure()
    return fig1, fig2

#fig1, fig2 = plot_scrore(history)
#model.save('my_model_1.h5')

#PARTIE 2 DETECTION DE CHIEN AVEC DU TRANSFERT LEARNING

# load the InceptionResNetV2 architecture with imagenet weights as base
def plot_image_train(x,y): 
    a = train_generator.class_indices
    class_names = list(a.keys()) # storing class/breed names in a list
  
    def plot_images(img,labels):
        plt.figure(figsize=[15,10])
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.imshow(img[i])
            plt.title(class_names[np.argmax(labels[i])])
            plt.axis('off')
        
    race_image = plot_images(x,y)
    plt.savefig('race_image.png')

Race_image = plot_image_train(x_test, y_test)

def model_pretrained(Nb_class):
    base_model = tf.keras.applications.InceptionResNetV2(
					include_top=False,
					weights='imagenet',
					input_shape=(150,150,3)
					)

    base_model.trainable=False
# For freezing the layer we make use of layer.trainable = False
# means that its internal state will not change during training.
# model's trainable weights will not be updated during fit(),
# and also its state updates will not run.

    model = tf.keras.Sequential([
    		base_model,
		tf.keras.layers.BatchNormalization(renorm=True),
		tf.keras.layers.GlobalAveragePooling2D(),
		tf.keras.layers.Dense(512, activation='relu'),
		tf.keras.layers.Dense(256, activation='relu'),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(128, activation='relu'),
		tf.keras.layers.Dense(Nb_class, activation='softmax')
	])
    model.summary()
    
    return model

def plot_score_2(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss'] 
  
    # plot results
    # accuracy
    fig1 = plt.figure(figsize=(10, 16))
    plt.rcParams['figure.figsize'] = [16, 9]
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.grid'] = True
    plt.rcParams['figure.facecolor'] = 'white'
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.title(f'\nTraining and Validation Accuracy. \nTrain Accuracy: {str(acc[-1])}\nValidation Accuracy: {str(val_acc[-1])}')
        # loss
    fig2 =plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title(f'Training and Validation Loss. \nTrain Loss: {str(loss[-1])}\nValidation Loss: {str(val_loss[-1])}')
    plt.xlabel('epoch')
    plt.tight_layout(pad=3.0)    
    plt.savefig('fig25_InceptionResNetV2.png')
    return fig1, fig2 

model_2 = model_pretrained(Nb_class)

liste_modele = ['xception.Xception','InceptionResNetV2','EfficientNetB7','VGG16']

history_2, model_2 = fit_model(model_2,train_generator,validation_generator,epochs)


accuracy_score = model_2.evaluate(validation_generator)
print(accuracy_score)
print("Accuracy: {:.4f}%".format(accuracy_score[1] * 100)) 
print("Loss: ",accuracy_score[0])

fig1 = plot_score_2(history_2)

model_2.save('my_model_25_InceptionResNetV2.h5')

print(train_generator.class_indices)