#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request,jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import os.path


# In[1]:


#A function used to plot two image side by side

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


# In[4]:


#This Function takes a folder's path or directory and creates a data base, and returns a list of images' paths and list of the folder in which the images are.

def create_database(path):
    BASE_PATH = path
    templates = []
    types = []
    for dirname, dirnames, filenames in os.walk(BASE_PATH):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                abs_path = "%s/%s" % (subject_path, filename)
                templates.append(abs_path)
                types.append(subdirname)
    return templates, types


# In[5]:


#This faunction Crops shapes in an image within determined (x,y) and return a list of images.

def cropim(image):
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
            #new_img = cv2.resize(new_img,(xx,yy))
            symbols.append(new_img)
            #idx+=1
            #cv2.imwrite('output/' + str(idx) + '.png', new_img)
    return symbols


# In[7]:


#this function counts the connected components in an image and return how many of them are there. 
#(it helps in our algorithm to optimize time consumed and matching quality)

def count_connected(image):
    ret, labels, stats, centroids =  cv2.connectedComponentsWithStats(image,8,cv2.CV_32S)
    count = 1
    font = cv2.FONT_HERSHEY_PLAIN
    for i in centroids:
        if int(i[0]) == 110:
            continue
        #cv2.putText(image,str(count),(int(i[0])-4,int(i[1]-7)),font,1,(255,255,255),lineType=cv2.LINE_AA)
        count += 1
    return count


# In[8]:


#This function works as a dictionary for the output instructions we should send to the user. 
def get_inastruction(s_types,class_x):
    instruction = ''
    if (class_x == 'ClassC'):
        if (s_types == 'dc_donot'):
            instruction = 'Do Not Dryclean.'
        elif (s_types == 'handw'):
            instruction = 'Garment may be laundered through the use of water, detergent or soap and gentle hand manipulation.'
        elif (s_types == 'ir_dontt'):
            instruction = 'Item may not be smoothed or finished with an iron.'
        elif (s_types == 'td_donot'):
            instruction = 'A machine dryer may not be used. Usually accompanied by an alternate drying method symbol.'
        elif (s_types == 'w_donot'):
            instruction = 'Do Not Wash.'
        elif (s_types == 'wl_donot'):
            instruction = 'Do Not Wring.'
            
    elif (class_x == 'ClassB'):

        if (s_types == 'dc_donot'):
            instruction = 'Do Not Dryclean.'
        elif (s_types == 'b_dont'):
            instruction = 'Do Not Bleach'
        elif (s_types == 'donot_dry'):
            instruction = 'Do Not Dry.'
        elif (s_types == 'ir_dontt'):
            instruction = 'Do Not Iron'
        elif (s_types == 'mw30'):
            instruction = 'Machine Wash, Cold'
        elif (s_types == 'mw40'):
            instruction = 'Machine Wash, Warm'
        elif (s_types == 'mw50'):
            instruction = 'Machine Wash, Hot'
        elif (s_types == 'mw60'):
            instruction = 'Machine Wash, Hot: 60 degree'
        elif (s_types == 'td_high'):
            instruction = 'Tumble Dry, Normal, High Heat.'
        elif (s_types == 'td_low'):
            instruction = 'Tumble Dry, Normal, Medium Heat.'
        elif (s_types == 'td_mid'):
            instruction = 'Tumble Dry, Normal, Low Heat.'
        elif (s_types == 'w_donot'):
            instruction = 'Do Not Wash.'
        elif (s_types == 'wl_donot'):
            instruction = 'Do Not Wring.'
            
    elif (class_x == 'ClassA'):

        
        if (s_types == 'b_dont'):
            instruction = 'Do Not Bleach'
        elif (s_types == 'donot_dry'):
            instruction = 'Do Not Dry.'
        elif (s_types == 'dry_flat'):
            instruction = 'Dry Flat.'
        elif (s_types == 'ir_tall1'):
            instruction = 'Iron, High Heat.'
        elif (s_types == 'ir_tall2'):
            instruction = 'Iron, Medium Heat.'
        elif (s_types == 'ir_tall3'):
            instruction = 'Iron, Low Heat.'
        elif (s_types == 'ir_tallempty'):
            instruction = 'Iron, Any Temperature, Steam or Dry.'
        elif (s_types == 'mw30'):
            instruction = 'Machine Wash, Cold.'
        elif (s_types == 'mw40'):
            instruction = 'Machine Wash, Warm.'
        elif (s_types == 'mw50'):
            instruction = 'Machine Wash, Hot.'
        elif (s_types == 'mw60'):
            instruction = 'Machine Wash, Hot: 60 degree.'
    
        elif (s_types == 'P'):
            instruction = 'Dryclean, Any Solvent Except Trichloroethylene.'
        elif (s_types == 'td_high'):
            instruction = 'Tumble Dry, Normal, High Heat.'
        elif (s_types == 'td_low'):
            instruction = 'Tumble Dry, Normal, Medium Heat.'
        elif (s_types == 'td_mid'):
            instruction = 'Tumble Dry, Normal, Low Heat.'
            
        elif (s_types == 'w_norm'):
            instruction = 'Machine Wash, Normal.'
    
    
    return instruction
    


# In[9]:


#we have made 3 classes of inputs base on the number of connected components (we have calculated it before using the function mentioned above)
templatesA, typesA = create_database('ClassA')
templatesB, typesB = create_database('ClassB')
templatesC, typesC = create_database('ClassC')


app = Flask(__name__)
cors = CORS(app)

# # @app.route('/upload')
# # def upload():
# #     return render_template('upload.html')
#
@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method=='POST':
        f=request.files['file']
        f.save(secure_filename(f.filename))
        
#here we read the input image
#img_rgb = cv2.imread('example5.jpg',1)
        img_rgb = cv2.imread(f.filename, 1)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        #here we apply threshold on the input image as a kind of pre-processing to facilitate the following operations.
        ret1,img_rgb = cv2.threshold(img_rgb,180,255,cv2.THRESH_BINARY)
        #here we crop the input image into small images.
        symbols = cropim(img_rgb)

        #we declare some global variables to be used below in implementation
        maxthresh = .09
        step = -0.1
        instruct = ''
        set_inst = {""}

        #A message is sent to user to inform him that the image is recieved correctly and processing started.
        print('Please wait for processing...!')

        #For each symbol detected by the crop fn, we calcualtes connected components to determine the class which the symbol belongs to.
        for symbol in symbols:
            _,w, h = symbol.shape[::-1]
            symbol = cv2.cvtColor(symbol, cv2.COLOR_BGR2GRAY)
            counter = count_connected(symbol)

        #for each class we modify the inputs to the nect step
            if (counter <=7):
                templatesg = templatesA
                typesg = typesA
                Classg = 'ClassA'
                maxthresh = 0.8
                step = -0.06
                #print('in class A')
            elif (counter >=7 and counter <=10):
                templatesg = templatesB
                typesg = typesB
                Classg = 'ClassB'
                maxthresh = 0.6
                step = -0.1
                #print('in class B')
            elif (counter > 10):
                templatesg = templatesC
                typesg = typesC
                Classg = 'ClassC'
                maxthresh = 0.6
                step = -0.1
        #for each symbol, and for each threshold we try to match the detected symbol with our templates in database.
            for thresh in np.arange(maxthresh, 0.2, step):

                for j in range(0, len(templatesg)-1, 1):
                    template1 = cv2.imread(templatesg[j],1)
                    template = cv2.imread(templatesg[j],0)
                    template = cv2.resize(template,(w,h))

                    flag=0
                    flag2 =0
                    res = cv2.matchTemplate(symbol,template,cv2.TM_CCOEFF_NORMED)
                    loc = np.where( res >= thresh)

                    for pt in zip(*loc[::-1]):
                        cv2.rectangle(symbol, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
                        #if symbol is matched with any of database templates, set the flags = 1 
                        flag=1
                        flag2=1


                    if flag == 1:
                        #if flag = 1, therefore the symbol is matched, so then we get the instruction from the dictionay and add it to the set of instruction that would be send to the user as an output.
                        instruct = get_inastruction(typesg[j],Classg)
                        set_inst.add(instruct)

                        break

                    #else:
                        #print('not detected')

                if flag2 == 1:
                    break

        finalResult=''
        for inst in set_inst:
            finalResult =finalResult + inst + "/"
        print(finalResult)
        print('done man')

        #//////////////////////////////////////////////////
        return jsonify(
        success=True,
        message=finalResult
         )
#if __name__ =='__main__':


# In[ ]:





# In[ ]:




