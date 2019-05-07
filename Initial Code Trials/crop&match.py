import cv2 
import numpy as np
'''image = cv2.imread("data1.jpg")
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(image, 10, 250)
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
idx = 0
symbols = []

#croping input ticket & appending in symbols array
for c in cnts:
	x,y,w,h = cv2.boundingRect(c)
	if w>45 and h>45:
		new_img=image[y:y+h,x:x+w]
		symbols.append(new_img)
		#idx+=1
		#cv2.imwrite('output/' + str(idx) + '.png', new_img)

#showing symbols just for testing
for symbol in symbols:
	idx+=1
	cv2.imshow('img' + str(idx),symbol)
cv2.waitKey(0)
'''


def cropim(image,xx,yy):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(image, 10, 250)
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

image = cv2.imread("ex1.jpg")
kernel = np.ones((5,5),np.uint8)
image = cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel)
template = cv2.imread("template3.png",0)
w, h = template.shape[::-1]
symbols = cropim(image,w,h)
flag=0

idx = 0
for symbol in symbols:
	idx+=1
	img_gray = cv2.cvtColor(symbol, cv2.COLOR_BGR2GRAY)
	res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
	threshold = 0.6
	loc = np.where( res >= threshold)
	for pt in zip(*loc[::-1]):
	    cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
	    flag=1
	    cv2.imshow("",symbol)
	#cv2.imshow('img' + str(idx),symbol)
print(flag)
cv2.waitKey(0)