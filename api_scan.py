# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 08:21:58 2021

@author: TeamOcr
"""
from flask import Flask, request
from flask_restful import Api
from flask_cors import CORS
import cv2
import numpy as np
import base64

import handDetection
import printedDoc
import BanknameD
import amountDigitrecog
import date_detection_model

import printedCheque
"""
You need to import all scripts


"""
app = Flask(__name__)
CORS(app)
api = Api(app)


def data_uri_to_cv2_img(uri):
    nparr = np.fromstring(base64.b64decode(uri), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def readb64(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img


@app.route('/scanning', methods = ['GET','POST'])
def digitfunction2():
    if request.method == "POST":
        url2 = request.form['front']
    
        img = data_uri_to_cv2_img(url2)
        cv2.imwrite('C:\\TFODCourse-main\\images\\temp.jpg', img)
        
        print("start scanning")  
        print("test"
        printedDoc.recognition()
        handDetection.handDetection()
        amountDigitrecog.digitamountRecog()
        BanknameD.NameRecog()
        date_detection_model.dateRecog()
        
        retJson = {
        'bank_name' : printedDoc.logoG,
        'company_name': BanknameD.name_written,
        'Date': date_detection_model.dateg,
        'amount': handDetection.amounthand,
        'rs':amountDigitrecog.amountdigit,
        'micr': printedDoc.mcrG,
        'address': printedDoc.AddressG
        }
        print("End scanning")
       
        print(retJson)
        return retJson
       
    else:
        
        return "Get method is trigger"
    
@app.route('/printed', methods = ['GET','POST'])
def printedFunction():
    if request.method == "POST":
        url2 = request.form['front']
    
        img = data_uri_to_cv2_img(url2)
        cv2.imwrite('C:\\Users\\Neha\\TFODCourse-main\\images\\printed.jpg', img)
        
        print("start scanning")  

        printedCheque.printedRecog()
        
        retJson = {
        'bank_name' : printedCheque.Plogo,
        'company_name': printedCheque.Pname,
        'Date': printedCheque.Pdate,
        'amount': printedCheque.Pamount,
        'rs': printedCheque.Pdigit,
        'micr': printedCheque.Pmcr,
        'address': printedCheque.Paddress
        }
        print("End scanning")
       
        print(retJson)
        return retJson
       
    else:
        
        return "Get method is trigger"
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)

