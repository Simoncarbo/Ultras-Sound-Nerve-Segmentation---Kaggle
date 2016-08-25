# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 15:20:56 2016

@author: sicarbonnell

temporary main
"""
import numpy as np
import numpy.random as random
import cv2
import skimage.transform as tf

from Utils.import_data import load_train_data
import Utils.processing as proc

#train_imgs, train_masks, train_index = load_train_data()

nb_rows, nb_columns = 160+20,224+20  # 420 x 580 -> (%2) 210 x 290 (%3) 140 x 193.333 (%4) 105 x 145 (%5) 84 x 116 (%6) 70 x 96.66

image = train_imgs[0,0]
image = cv2.resize(image, (nb_columns, nb_rows), interpolation=cv2.INTER_CUBIC) # INVERSE ORDER! cols,rows

#print(np.max(image))

def_image = proc.transform(image)
print(np.max(def_image))
print(np.min(def_image))

image = image[10:-10,10:-10]

#print(np.max(def_image))

image = cv2.resize(image, (nb_columns*4, nb_rows*4), interpolation=cv2.INTER_AREA)
def_image = cv2.resize(def_image, (nb_columns*4, nb_rows*4), interpolation=cv2.INTER_AREA)

cv2.imshow('original',image)
cv2.imshow('def',def_image.astype(np.uint8))


#cv2.imwrite('normal_small.png',image)#,(image*255/np.max(image)).astype(int))
#cv2.imwrite('transformed.png',def_image)#(def_image*255/np.max(def_image)).astype(int))


# plot different sizes
#for nb_rows, nb_columns in [(320,448),(160,224),(80,112),(40,56),(20,28),(10,14),(5,7)]:
#    image = train_imgs[0,0]
#    image = cv2.resize(image, (nb_columns, nb_rows), interpolation=cv2.INTER_CUBIC) # INVERSE ORDER! cols,rows
#    cv2.imwrite('10_1_'+str(nb_rows)+'x'+str(nb_columns)+'.png',image)