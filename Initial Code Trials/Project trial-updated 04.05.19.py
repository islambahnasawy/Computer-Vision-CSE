from flask import Flask, render_template, request,jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import os.path


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


def cropim(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(image, 10, 250)
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    idx = 0
    symbols = []

    # croping input ticket & appending in symbols array
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w > 50 and h > 50:
            new_img = image[y:y + h, x:x + w]
            # new_img = cv2.resize(new_img,(xx,yy))
            symbols.append(new_img)
            # idx+=1
            # cv2.imwrite('output/' + str(idx) + '.png', new_img)
    return symbols


# In[6]:


def cornerHarris(image, blockSize, ksize, k=0.04):
    gray = np.float32(image)

    I_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize)
    I_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize)

    Ixx = I_x ** 2
    Ixy = I_y * I_x
    Iyy = I_y ** 2
    offset = int(blockSize / 2)
    rows = image.shape[0]
    cols = image.shape[1]
    corners = np.array([])
    corners.resize(gray.shape)
    idx = 0
    for y in range(offset, rows - offset):
        for x in range(offset, cols - offset):
            Sxx = np.sum(Ixx[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Syy = np.sum(Iyy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Sxy = np.sum(Ixy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            # Find determinant and trace
            M = np.array([[Sxx, Sxy], [Sxy, Syy]])
            det = np.linalg.det(M)
            trace = np.matrix.trace(M)
            r = det - k * (trace ** 2)
            corners[y, x] = r
            idx += 1
    return idx


# In[7]:


def count_connected(image):
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(image, 8, cv2.CV_32S)
    count = 1
    font = cv2.FONT_HERSHEY_PLAIN
    for i in centroids:
        if int(i[0]) == 110:
            continue
        # cv2.putText(image,str(count),(int(i[0])-4,int(i[1]-7)),font,1,(255,255,255),lineType=cv2.LINE_AA)
        count += 1
    return count


# In[8]:


def get_inastruction(s_types, class_x):
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


templatesA, typesA = create_database('ClassA')
templatesB, typesB = create_database('ClassB')
templatesC, typesC = create_database('ClassC')

# print(templatesC)
# print(templatesB)
# print(templatesC)

app = Flask(__name__)
cors = CORS(app)

# # @app.route('/upload')
# # def upload():
# #     return render_template('upload.html')
#

#create flask api to connect with android
@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method=='POST':
        f=request.files['file']
        f.save(secure_filename(f.filename))

        # //////////////////////////////////////////////

        img_rgb = cv2.imread(f.filename, 1)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        # kernel = np.ones((2,2),np.uint8)
        # img_rgb = cv2.morphologyEx(img_rgb,cv2.MORPH_OPEN,kernel)
        ret1, img_rgb = cv2.threshold(img_rgb, 180, 255, cv2.THRESH_BINARY)
        symbols = cropim(img_rgb)

        # threshold = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]
        # for thresh in threshold:

        maxthresh = .09
        step = -0.1
        instruct = ''
        set_inst = {""}

        print('Please wait for processing...!')
        for symbol in symbols:
            _, w, h = symbol.shape[::-1]
            symbol = cv2.cvtColor(symbol, cv2.COLOR_BGR2GRAY)

            counter = count_connected(symbol)
            # print('counter = ', counter)

            # gray = np.float32(symbol)
            # out = cornerHarris(gray,3,3,0.07)
            # gray = cv2.medianBlur(symbol, 5)
            # rows = gray.shape[0]

            if (counter <= 7):
                templatesg = templatesA
                typesg = typesA
                Classg = 'ClassA'
                maxthresh = 0.8
                step = -0.06
                # print('in class A')
            elif (counter >= 7 and counter <= 10):
                templatesg = templatesB
                typesg = typesB
                Classg = 'ClassB'
                maxthresh = 0.6
                step = -0.1
                # print('in class B')
            elif (counter > 10):
                templatesg = templatesC
                typesg = typesC
                Classg = 'ClassC'
                maxthresh = 0.6
                step = -0.1
                # print('in class C')
                # threshold = 0.4
            # print('in symbols')
            # plot_side_by_side(symbol,symbol,"template","detected shape",1)
            for thresh in np.arange(maxthresh, 0.2, step):
                # print('in thresh')
                for j in range(0, len(templatesg) - 1, 1):
                    template1 = cv2.imread(templatesg[j], 1)
                    template = cv2.imread(templatesg[j], 0)
                    template = cv2.resize(template, (w, h))
                    # w, h = template.shape[::-1]
                    flag = 0
                    flag2 = 0
                    res = cv2.matchTemplate(symbol, template, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(res >= thresh)
                    # print('in template')

                    for pt in zip(*loc[::-1]):
                        cv2.rectangle(symbol, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
                        flag = 1
                        flag2 = 1
                        # print('yes')

                    if flag == 1:
                        # print('Shape is detected')
                        # plot_side_by_side(template1,symbol,"template","detected shape",1)
                        # print(typesg[j])
                        instruct = get_inastruction(typesg[j], Classg)
                        set_inst.add(instruct)
                        # print(instruct)

                        break

                # threshold = 0.4

                # else:
                # print('not detected')

                if flag2 == 1:
                    break
        finalResult=''
        for inst in set_inst:
            finalResult =finalResult +"/n"+ inst
            # print(inst)
        # print(set_inst)
        print(finalResult)
        print('done man')

        #//////////////////////////////////////////////////
        return jsonify(
        success=True,
        message='success'
         )
#if __name__ =='__main__':

app.run(host='192.168.1.9')




