####Was machden die Bilder hier ?
#import tensorflow as tf
import torch
import pandas as pd
import numpy
import glob


IMAGE_PATH=r"Data\images\normalizedImg/"
imagelist=glob.listdir(IMAGE_PATH)
images=pd.DataFrame([])
names=[]

#images=np.zeros((len(imagelist),1440000))
#for j in range(0, len(imagelist)):
 #   i=imagelist[j]
  #  print(i)

   # im=np.asarray(PIL.Image.open(IMAGE_PATH+i)).flatten()
   # names.append(i)
   # images[j]=im
    #images=pd.concat((images, pd.DataFrame(im)), ignore_index=True)
#Images=pd.DataFrame(images)
#prognosis=[]
#for i in range(0, len(names)):
 #   p=np.array(train[train.ImageFile==names[i]].Prognosis)[0]
  #  if p=="MILD":
   #     prognosis.append(0)
   # else:
    #    prognosis.append(1)
#so, the prognosis is the outcome for the image analysis...one means severe, zero means mild

##try a random forest classifier, and a neural network

#Xtrain, Xtest,ytrain ,ytest=tts(Images, prognosis)
#regressor=RFR(n_estimators=500, n_jobs=4, verbose=2)

#regressor.fit(Xtrain, ytrain)
#res=regressor.predict(Xtest).round()

#pred_acc=1-np.mean(abs(res-ytest))
#regressor2=MLPC(hidden_layer_sizes=(300,len(Xtrain.columns),2), verbose=2, max_iter=10000, tol=0.000001)

#regressor2.fit(Xtrain, ytrain)
#res2=regressor2.predict(Xtest).round()

#pred_acc2=1-np.mean(abs(res2-ytest))

#hm...not enough Ram...maybe things like ppartial_fit could be a nice idea
#Naja oder, weniger retundanz, dann haut das mit dem RAM auch hin;D
# ich glaube kaum, dass wir 16 GB Bilder haben oder NP arrays usw


#im=tf.keras.preprocessing.image.load_img(path_train+j+"/"+fname)
# # ###same for VR
# # and maybe a random selection of regressors?
# #what about keras/tensorflow?
# ##reworking impuiatition, clipping to a certain range...

