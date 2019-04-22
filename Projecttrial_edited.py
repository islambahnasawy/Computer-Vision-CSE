#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
#from matplotlib import pyplot as plt


# In[2]:


def plot_side_by_side(first, second, input_name, output_name, img_type):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 25))
    if img_type == 0:
        # grey
        ax1.imshow(cv2.cvtColor(first, cv2.COLOR_GRAY2RGB))
        ax2.imshow(cv2.cvtColor(second, cv2.COLOR_GRAY2RGB))
    else:
        ax1.imshow(cv2.cvtColor(first, cv2.COLOR_BGR2RGB))
        ax2.imshow(cv2.cvtColor(second, cv2.COLOR_BGR2RGB))
    ax1.set_title(input_name)
    ax2.set_title(output_name)
    plt.show()


# In[3]:


def cropim(image,xx,yy):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(image, 10, 250)
    (_,cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    idx = 0
    symbols = []

    #croping input ticket & appending in symbols array
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w>50 and h>50:
            new_img=image[y:y+h,x:x+w]
            new_img = cv2.resize(new_img,(xx,yy))
            symbols.append(new_img)
            #idx+=1
            #cv2.imwrite('output/' + str(idx) + '.png', new_img)
    return symbols


# In[12]:


image = cv2.imread("data1.jpg",1)
symbols = cropim(image,225,225)

idx = 0
for symbol in symbols:


    #image = cv2.imread('data.jpg',1)
    img_gray = cv2.cvtColor(symbol, cv2.COLOR_BGR2GRAY)
    template1 = cv2.imread('t2.jpg',1)
    template = cv2.imread('t2.jpg',0)
    w, h = template.shape[::-1]
    flag=0


    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(symbol, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        flag=1
        #print('yes')

    if flag == 1:
        print('Shape is detected')
        cv2.imshow('img',symbol)
        cv2.imshow('template',template1)
        cv2.waitKey(0)

        

    else:
        print('not detected')

#plot_side_by_side(template1,img_rgb,"template","detected shape",1)


# In[ ]:





# In[ ]:




