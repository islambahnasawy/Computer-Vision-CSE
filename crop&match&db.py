import cv2 
import numpy as np
import os.path

templates = []
types = []

def create_database():
	BASE_PATH = 'db/symbols/'
	for dirname, dirnames, filenames in os.walk(BASE_PATH):
		for subdirname in dirnames:
		    subject_path = os.path.join(dirname, subdirname)
		    for filename in os.listdir(subject_path):
		        abs_path = "%s/%s" % (subject_path, filename)
		        templates.append(abs_path)
		        types.append(subdirname)


#function crops input image to extract symbols
def crop_ticket(ticket):
    gray = cv2.cvtColor(ticket,cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(ticket, 10, 250)
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    idx = 0
    symbols = []

    #croping input ticket & appending in symbols array
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w>50 and h>50:
            symbol = ticket[y:y+h,x:x+w]
            #new_img = cv2.resize(new_img,(xx,yy))
            symbols.append(symbol)
            #idx+=1
            #cv2.imwrite('output/' + str(idx) + '.png', new_img)
    return symbols


def match_symbol(symbol):
	flag = 0
	img_gray = cv2.cvtColor(symbol, cv2.COLOR_BGR2GRAY)
	width, height = img_gray.shape[::-1]
	for j in range(0, len(templates)-1, 1):
		symbol_type = ''
		template = cv2.imread(templates[j],0)
		template = cv2.resize(template,(width,height))
		res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
		threshold = 1
		for i in range(100, 0, -1):
			threshold = (i/100)
			loc = np.where( res >= threshold)
			for pt in zip(*loc[::-1]):
			    #cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
			    flag=1
			    #cv2.imshow("im"+str(idx)+str(threshold),symbol)
			#cv2.imshow('img' + str(idx),symbol)
			if flag == 1:
				symbol_type = types[j]
				break
		if flag == 1:
			return symbol,threshold,symbol_type
		else:
			return 0,0,0



create_database()
#reading and openning input ticket to remove white noise
ticket = cv2.imread("ex1.jpg")
kernel = np.ones((5,5),np.uint8)
ticket = cv2.morphologyEx(ticket,cv2.MORPH_OPEN,kernel)

symbols = crop_ticket(ticket)
idx = 0
for symbol in symbols:
	idx += 1
	cv2.imshow("image"+str(idx),symbol)
	
	# template,threshold,symbol_type = match_symbol(symbol)
	# if threshold == 0:
	# 	print (threshold)
	# else:
	# 	print (idx," ",threshold," ",symbol_type)
	# 	cv2.imshow('template' + str(idx),template)

cv2.waitKey(0)