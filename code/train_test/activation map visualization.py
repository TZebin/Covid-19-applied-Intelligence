#!/usr/bin/env python
# coding: utf-8

# In[1]:
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
import numpy as np
from tqdm import tqdm
import math

from tensorflow.keras import backend as K

covid_path = './dataset/covid/'
covid_files = os.listdir(covid_path)
#covid_files.remove('.DS_Store')

labels_covid = []
data_covid = []

for file in tqdm(covid_files):
    imagePath = covid_path + file
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    data_covid.append(image)
    labels_covid.append('covid')


pne_path = 'C:/Users/User/Desktop/keras-covid-19_Adrian/dataset/pneumonia/'
pne_files = os.listdir(pne_path)
#norm_files.remove('.DS_Store')

labels_pne = []

data_pne = []

for file in tqdm(pne_files):
    imagePath = pne_path + file
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    data_pne.append(image)
    labels_pne.append('pneum')


norm_path = 'C:/Users/User/Desktop/keras-covid-19_Adrian/dataset/normal/'
norm_files = os.listdir(norm_path)
#norm_files.remove('.DS_Store')

labels_norm = []

data_norm = []

for file in tqdm(norm_files):
    imagePath = norm_path + file
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    data_norm.append(image)
    labels_norm.append('normal')
    
data = data_covid + data_pne + data_norm

labels = labels_covid + labels_pne + labels_norm  
  

data = np.array(data) / 255.0
labels = np.array(labels)
# perform one-hot encoding on the labels

lb = LabelEncoder()
labels = lb.fit_transform(labels)
labels = to_categorical(labels); print(labels)    
print(lb.classes_)
print(lb.inverse_transform([0,1,2]))
    
   
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# initialize the training data augmentation object
trainAug = ImageDataGenerator(
	rotation_range=15,
	fill_mode="nearest")

# load the VGG16 network, ensuring the head FC layer sets are left
# off
baseModel = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))
baseModel.summary()
# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# #### Transfer Learning

#### Downloading the Pre-trained Model
#mobnet_model = keras.applications.mobilenet.MobileNet()



lr_opt =1e-3

epochs = 2
batch_size = 8
 

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=lr_opt, decay= lr_opt/epochs)


model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

model.summary()

# train the head of the network
# import keras
#lr=keras.callbacks.LearningRateScheduler(step_decay,verbose=1)
#checkpoint= keras.callbacks.ModelCheckpoint('./Checkpoint_normal', monitor='val_acc', 
#                                               verbose=0, save_best_only=True, 
#                                               save_weights_only=False, 
#                                               mode='auto',
#                                               period=1)

print("[INFO] training head...")

H = model.fit_generator(
                trainAug.flow(np.array(trainX), np.array(trainY), batch_size=batch_size),
                steps_per_epoch=len(trainX) // batch_size,
                validation_data=(np.array(testX), np.array(testY),),
                validation_steps=len(testX) // batch_size,
                epochs=epochs)


N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('C:/Users/User/Desktop/keras-covid-19_Adrian/plot_3_class.png')

# serialize the model to disk
print("[INFO] saving COVID-19 detector model...")
model.save("C:/Users/User/Desktop/keras-covid-19_Adrian/model_covid_vs_pneumonia_vs_normal.h5")

print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=batch_size)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report

print(classification_report(testY.argmax(axis=1), predIdxs,target_names=['covid', 'normal', 'pneum']))

# compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity
cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
# total = sum(sum(cm))
# acc = (cm[0, 0] + cm[1, 1]) / total
# sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
# specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

# show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
           
predIdxs = model.predict(np.array(testX), batch_size=8)
predIdxs = np.argmax(predIdxs, axis=1)


# In[ ]:


predIdxs


# In[ ]:


testY.argmax(axis=1)


# In[ ]:


import sklearn
print(sklearn.metrics.confusion_matrix(testY.argmax(axis=1), predIdxs))


# In[ ]:


print(classification_report(testY.argmax(axis=1), predIdxs,target_names=['covid', 'normal', 'pneum']))


# In[ ]:


model.save("/kaggle/working/model_covid_vs_pneumonia_vs_normal.h5")


# In[8]:


from tensorflow.keras.models import load_model
 
# load model
model = load_model('./model_pneumonia_vs_covid.h5')


# In[9]:


model.summary()


# ### Class Activation Map

# In[ ]:


# img_path = norm_path + norm_files[0]
# norm_img = cv2.imread(img_path)
# norm_img = cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB)
# norm_img = cv2.resize(norm_img, (224, 224))
# norm_img = np.expand_dims(norm_img,axis=0)


# In[10]:


def get_class_activation_map(ind,path,files) :
    
    img_path =  path + files[ind]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img,axis=0)
    
    predict = model.predict(img)
    target_class = np.argmax(predict[0])
    last_conv = model.get_layer('block5_conv3')
    grads = K.gradients(model.output[:,target_class],last_conv.output)[0]
    pooled_grads = K.mean(grads,axis=(0,1,2))
    iterate = K.function([model.input],[pooled_grads,last_conv.output[0]])
    pooled_grads_value,conv_layer_output = iterate([img])
    
    for i in range(512):
        conv_layer_output[:,:,i] *= pooled_grads_value[i]
    
    heatmap = np.mean(conv_layer_output,axis=-1)
    
    for x in range(heatmap.shape[0]):
        for y in range(heatmap.shape[1]):
            heatmap[x,y] = np.max(heatmap[x,y],0)
    heatmap = np.maximum(heatmap,0)
    heatmap /= np.max(heatmap)
    plt.imshow(heatmap)
#     output_path_heatmap = '/kaggle/working/output_images/' + files[ind] + 'heatmap.jpeg'
#     plt.imsave(output_path_heatmap,heatmap)
    
    img_gray = cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY)
    upsample = cv2.resize(heatmap, (224,224))
    # plt.imshow(upsample,alpha=0.5)
    # plt.imshow(img_gray)
    #plt.imshow(upsample * img_gray)
    output_path_gradcam = 'C:/Users/User/Desktop/keras-covid-19_Adrian/output' + files[ind] + 'gradcam.jpeg'
    plt.imsave(output_path_gradcam,upsample * img_gray)
    
    #plt.show()
    
    #return img


# In[20]:


path = pne_path
files = pne_files


# In[26]:


get_class_activation_map(5,path,files)


# In[25]:


path + files[5]



# import shutil
# shutil.rmtree('/kaggle/working/output_images/')
os.remo



np.arange(0,11)


# In[ ]:


# img_path = covid_path + covid_files[1]
# covid_img = cv2.imread(img_path)
# covid_img = cv2.cvtColor(covid_img, cv2.COLOR_BGR2RGB)
# covid_img = cv2.resize(covid_img, (224, 224))
# covid_img = np.expand_dims(covid_img,axis=0)


# In[ ]:


#norm_img.shape


# In[ ]:


#covid_img.shape


# In[ ]:


#predict = model.predict(norm_img)
# print(decode_predictions(predict,top=3))
# target_class = np.argmax(predict[0])


# In[ ]:


predict = model.predict(norm_img)


# In[ ]:


predict


# In[ ]:


# target_class = np.argmax(predict[0])
# target_class


# In[ ]:


#print("Target Class is covid")


# In[ ]:


#last_conv = model.get_layer('block5_conv3')


# In[ ]:


#grads = K.gradients(model.output[:,0],last_conv.output)[0]


# In[ ]:


# pooled_grads = K.mean(grads,axis=(0,1,2))
# iterate = K.function([model.input],[pooled_grads,last_conv.output[0]])
# pooled_grads_value,conv_layer_output = iterate([covid_img])


# In[ ]:


# for i in range(512):
#     conv_layer_output[:,:,i] *= pooled_grads_value[i]
# heatmap = np.mean(conv_layer_output,axis=-1)


# In[ ]:


# for x in range(heatmap.shape[0]):
#     for y in range(heatmap.shape[1]):
#         heatmap[x,y] = np.max(heatmap[x,y],0)


# In[ ]:


# heatmap = np.maximum(heatmap,0)
# heatmap /= np.max(heatmap)
# plt.imshow(heatmap)


# In[ ]:


#img_gray = cv2.cvtColor(norm_img[0], cv2.COLOR_BGR2GRAY)


# In[ ]:


upsample = cv2.resize(heatmap, (224,224))
# plt.imshow(upsample,alpha=0.5)
# plt.imshow(img_gray)
plt.imshow(upsample * img_gray)
plt.show()


# In[ ]:


upsample = cv2.resize(heatmap, (224,224))
# plt.imshow(upsample,alpha=0.5)
# plt.imshow(img_gray)
plt.imshow(upsample * img_gray)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





