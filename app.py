
#import beeharrytest

#print(beeharrytest.predictionlist)

#print(*beeharrytest.predictionlist, sep = "")


#digit = ''.join(str(e) for e in beeharrytest.predictionlist)

from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import requests
import json
import subprocess
import sys
from flask_cors import CORS
import cv2
import numpy as np
import base64
import string

app = Flask(__name__)
CORS(app)
api = Api(app)


@app.route('/digit', methods = ['POST'])
def digitfunction():
    url = request.args.get("url")
    if url == 'SBM':
        r = requests.get("https://icecube-eu-832.icedrive.io/thumbnail?p=kNxhFfQcouDh6KNq67jWmC2sMICL255vYSkWMlPS1R%2F7%2B%2FXNygk2W%2FYfx8hAhbR1dQh2qtJkCWvaN1BOLlsXBLhAQLS%2F%2BFknPLLT5L6uYEczr9kgmBb2KX2dtPZcCZzh&w=1280&h=1280&m=cropped")
    elif url == 'MCB':
        r = requests.get("https://icecube-eu-832.icedrive.io/thumbnail?p=kNxhFfQcouDh6KNq67jWmC2sMICL255vYSkWMlPS1R%2F7%2B%2FXNygk2W%2FYfx8hAhbR1dQh2qtJkCWvaN1BOLlsXBLhAQLS%2F%2BFknPLLT5L6uYEczr9kgmBb2KX2dtPZcCZzh&w=1280&h=1280&m=cropped")
    elif url == 'MAU_BANK':
        r = requests.get("https://icecube-eu-832.icedrive.io/thumbnail?p=kNxhFfQcouDh6KNq67jWmC2sMICL255vYSkWMlPS1R%2F7%2B%2FXNygk2W%2FYfx8hAhbR1dQh2qtJkCWvaN1BOLlsXBLhAQLS%2F%2BFknPLLT5L6uYEczr9kgmBb2KX2dtPZcCZzh&w=1280&h=1280&m=cropped")
    elif url == 'ABSA':
        r = requests.get("https://icecube-eu-301.icedrive.io/thumbnail?p=XwZ1OupXQ9sDozb3KR5vuBNAUaZEDnFiEKK3vbh7FLmDobdStkhKbt6KLuloF6MO8lqgoWPq04qi%2Fw1sYIO06ZJ2QoAhBdDaSj710tVr7KsmZmQ7De%2BsXL1%2FOYChPsjT&w=1280&h=1280&m=cropped")
    elif url == 'AfrAsia':
        r = requests.get("https://icecube-eu-832.icedrive.io/thumbnail?p=kNxhFfQcouDh6KNq67jWmC2sMICL255vYSkWMlPS1R%2F7%2B%2FXNygk2W%2FYfx8hAhbR1dQh2qtJkCWvaN1BOLlsXBLhAQLS%2F%2BFknPLLT5L6uYEczr9kgmBb2KX2dtPZcCZzh&w=1280&h=1280&m=cropped")
    elif url == 'HSBC':
        r = requests.get("https://icecube-eu-832.icedrive.io/thumbnail?p=kNxhFfQcouDh6KNq67jWmC2sMICL255vYSkWMlPS1R%2F7%2B%2FXNygk2W%2FYfx8hAhbR1dQh2qtJkCWvaN1BOLlsXBLhAQLS%2F%2BFknPLLT5L6uYEczr9kgmBb2KX2dtPZcCZzh&w=1280&h=1280&m=cropped")
    elif url == 'BANK_ONE':
        r = requests.get("https://icecube-eu-832.icedrive.io/thumbnail?p=kNxhFfQcouDh6KNq67jWmC2sMICL255vYSkWMlPS1R%2F7%2B%2FXNygk2W%2FYfx8hAhbR1dQh2qtJkCWvaN1BOLlsXBLhAQLS%2F%2BFknPLLT5L6uYEczr9kgmBb2KX2dtPZcCZzh&w=1280&h=1280&m=cropped")
    elif url == 'BARCLAYS':
        r = requests.get("https://icecube-eu-832.icedrive.io/thumbnail?p=kNxhFfQcouDh6KNq67jWmC2sMICL255vYSkWMlPS1R%2F7%2B%2FXNygk2W%2FYfx8hAhbR1dQh2qtJkCWvaN1BOLlsXBLhAQLS%2F%2BFknPLLT5L6uYEczr9kgmBb2KX2dtPZcCZzh&w=1280&h=1280&m=cropped")
    elif url == 'BARODA':
        r = requests.get("https://icecube-eu-832.icedrive.io/thumbnail?p=kNxhFfQcouDh6KNq67jWmC2sMICL255vYSkWMlPS1R%2F7%2B%2FXNygk2W%2FYfx8hAhbR1dQh2qtJkCWvaN1BOLlsXBLhAQLS%2F%2BFknPLLT5L6uYEczr9kgmBb2KX2dtPZcCZzh&w=1280&h=1280&m=cropped")
    elif url == 'STANDARD_CHARTED':
        r = requests.get("https://icecube-eu-832.icedrive.io/thumbnail?p=kNxhFfQcouDh6KNq67jWmC2sMICL255vYSkWMlPS1R%2F7%2B%2FXNygk2W%2FYfx8hAhbR1dQh2qtJkCWvaN1BOLlsXBLhAQLS%2F%2BFknPLLT5L6uYEczr9kgmBb2KX2dtPZcCZzh&w=1280&h=1280&m=cropped")
    elif url == 'HBL':
        r = requests.get("https://icecube-eu-832.icedrive.io/thumbnail?p=kNxhFfQcouDh6KNq67jWmC2sMICL255vYSkWMlPS1R%2F7%2B%2FXNygk2W%2FYfx8hAhbR1dQh2qtJkCWvaN1BOLlsXBLhAQLS%2F%2BFknPLLT5L6uYEczr9kgmBb2KX2dtPZcCZzh&w=1280&h=1280&m=cropped")
    elif url == 'BCP':
        r = requests.get("https://icecube-eu-401.icedrive.io/thumbnail?p=f%2Bu8KcHtZws%2BEAp25OQ1bhOM1oRCZwNJZmcN5obTk7x0thbmAmv1tTiNi7%2BYTliRxkCP8K4Jjo77Tz2yXgSJ3UyzsBJdx5CzHdB4vk78h995FOH4EC0JGwp%2Fb76Pwr9s&w=1280&h=1280&m=cropped")
    retJson = {}
    with open("C:/Users/User/Desktop/yt_object_detection/Tensorflow/workspace/images/test/temp.jpg","wb") as f: 
        f.write(r.content)
    print("start")    
    beeharrytest.test()
    test2.test2func()
    print("end")
   
    
    retJson1 = {
        'digit': beeharrytest.x1,
        'Amount': test2.x2
        }
    """
    with open('text.json','w') as f1:
        json.dump(retJson1,f1)
    
    
    
    with open("text.json") as f:
        retJson = json.load(f)
    """    
    return retJson1

def data_uri_to_cv2_img(uri):
    nparr = np.fromstring(base64.b64decode(uri), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img



@app.route('/base64test', methods = ['GET','POST'])
def digitfunction2():
    if request.method == "POST":
        url2 = request.form['front']
        #print(url2)
        img = data_uri_to_cv2_img(url2)
        cv2.imwrite('C:/Users/User/Desktop/yt_object_detection/Tensorflow/workspace/images/test/temp.jpg', img)
        
        print("start")    
        beeharrytest.test()
        test2.test2func()
        print("end")
       
        retJson1 = {
        'digit': beeharrytest.x1,
        'Amount': test2.x2
        }
        """
        
        retJson1 = {
        'digit': 'dfsda',
        'Amount': 'ghsdg'
        }
        """
        print(retJson1)
        return retJson1
       
    else:
        
        return "123"
    


class Classify(Resource):
    def post(self):
        postedData = request.get_json(force=True)
        url = postedData["url"]
        r = requests.get(url)
        retJson = {}
        with open("C:/Users/User/Desktop/yt_object_detection/Tensorflow/workspace/images/test/temp.jpg","wb") as f:
            f.write(r.content)
            # To remove
            #import beeharrytest
            
            # Pass Image 
            proc = subprocess.Popen('beeharrytest.py', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
            ret = proc.communicate()[0]
            proc.wait()
            with open("text.json") as f:
                retJson = json.load(f)
        return retJson
 
api.add_resource(Classify,'/classify')   
if __name__ == "__main__":
    app.run(host='0.0.0.0')

