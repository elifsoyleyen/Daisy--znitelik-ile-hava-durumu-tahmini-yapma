# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 16:28:31 2021

@author: elif
"""


import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import random 
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from keras import Sequential
from keras.layers import  Dense 
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from PyQt5.QtWidgets import QApplication, QTableView
from PyQt5.QtCore import QAbstractTableModel, Qt
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QRadioButton, QLabel, QGridLayout,QDesktopWidget, QWidget,QTableWidget,QTableView,QTableWidgetItem,QHeaderView,QGraphicsScene,QGraphicsPixmapItem,QFileDialog
import matplotlib.pyplot as plt
from goruntu import Ui_Dialog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import cv2
from skimage.feature import daisy
import pickle
from  keras.utils import np_utils
from keras.layers import Convolution2D, MaxPooling2D
from sklearn.preprocessing import StandardScaler
from yellowbrick.classifier import ROCAUC
from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from skimage import color
import os,shutil
from keras.layers import Dense, Dropout, Activation, Flatten
import scikitplot.metrics as splt
from sklearn.model_selection import KFold
import imutils
import sklearn.metrics as metrics
from sklearn.metrics import r2_score, mean_squared_error
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QPushButton


class MainWindow(QWidget,Ui_Dialog):
   
    def __init__(self):
      
        QtWidgets.QMainWindow.__init__(self)  
        self.setupUi(self)
        self.pushButton_2.clicked.connect(self.ayirma)
        self.pushButton.clicked.connect(self.loadetme)
        self.pushButton_6.clicked.connect(self.makineogrenmesi)
        self.pushButton_4.clicked.connect(self.daisy)
        self.pushButton.clicked.connect(self.egitme)
        self.pushButton_3.clicked.connect(self.klasorsec)
        self.pushButton_5.clicked.connect(self.resimsec)
        self.pushButton_7.clicked.connect(self.tahminet)
        self.pushButton_8.clicked.connect(self.arttir)
        self.comboBox.addItem("0.2")
        self.comboBox.addItem("0.3")
        self.comboBox.addItem("0.4")
        self.comboBox_4.addItem("Sift")
        self.comboBox_4.addItem("Surf")
        self.comboBox_5.addItem("Rgb")
        self.comboBox_5.addItem("Hsv")
        self.comboBox_5.addItem("Cie")
        self.comboBox_3.addItem("Hold-Out")
        self.comboBox_3.addItem("K-Fold")
        self.comboBox_6.addItem("1")
        self.comboBox_6.addItem("2")
        self.comboBox_6.addItem("3")
        self.comboBox_6.addItem("4")
        self.comboBox_6.addItem("5")
        self.comboBox_2.addItem("2")
        self.comboBox_2.addItem("5")
        self.path=""
      
        
        
    def klasorsec(self):
        file = str(QFileDialog.getExistingDirectory(self, "klasörü seç"))
        self.path=file.replace("C:/Users/elif/Desktop/goruntufinalodev/","./")+"/"
        self.directories=os.listdir(self.path)  
       
        
    def resimsec(self):
        dosyaadi = QFileDialog.getOpenFileName()
        self.resim = dosyaadi[0]
        print(self.resim)
        self.image = cv2.imread(self.resim.replace("C:/Users/elif/Desktop/goruntufinalodev/",""))
        self.img = cv2.resize(self.image, (160,120))
        self.dosya = 'savedImage.jpg'
        cv2.imwrite(self.dosya, self.img)
        self.pixmap = QPixmap("./"+self.dosya)
        self.label_34.setPixmap(self.pixmap)

    def arttir(self):
         file = str(QFileDialog.getExistingDirectory(self, "klasörü seç"))
         path=file.replace("C:/Users/elif/Desktop/goruntufinalodev/","./")+"/"
         directories=os.listdir(path)  
         for klasorid,directory in enumerate(directories):
             print (directory)
             files=os.listdir(path+directory)
             print (files)
             for file_name in files:
                 for derece in (30,90,120):
                     resim = cv2.imread(path+directory+"/"+file_name)
                     if(klasorid==0):
                         klasörismi=str("cloudy/")
                     elif (klasorid==1):
                         klasörismi=str("rain/")
                     elif (klasorid==2):
                         klasörismi=str("shine/")
                     elif (klasorid==3):
                         klasörismi=str("sunrise/")
                     rotate = imutils.rotate_bound(resim,derece)
                     cv2.imwrite("./veriarttirim/"+klasörismi+file_name+str(derece)+".jpg",rotate)
                     print("./veriarttirim/"+klasörismi+file_name+str(derece)+".jpg")
                     
    def tahminet(self):
        havadurum=["Cloudy","Rain","Shine","Sunrise"]
        image =cv2.imread(self.filename)
        saved_model = load_model("den.h5")
        if self.label_26.text()=="Sift Algoritması ile RGB Uzayında Hazırlandı..":
            descs=[]
            X=[]
            print('burdayım')
            sift = cv2.xfeatures2d.SIFT_create(nfeatures=10)
            keypoints_sift,descriptors=sift.detectAndCompute(image,None)
            image = cv2.drawKeypoints(image, keypoints_sift, None)
            image=color.rgb2gray(image)
            yükseklik=image.shape[0]
            genislik=image.shape[1] 
            n = 50
            for kp in keypoints_sift:
                xekseni,yekseni=int(kp.pt[0]),int(kp.pt[1])
                if xekseni-int(n/2)>0 and xekseni+(n/2)<genislik and yekseni-int(n/2)>0 and yekseni+int(n/2)<yükseklik:
                    print('girdi')
                    region = image[yekseni-int(n/2):yekseni+int(n/2),xekseni-int(n/2):xekseni+int(n/2)]
                    desc, descs_img = daisy(region,rings=2, histograms=6,
                           step=33,radius=7,visualize=True) 
                    desc=desc.reshape(desc.shape[0],desc.shape[1]*desc.shape[2])
                    desc=cv2.resize(desc, (50, 50))
                    desc=desc.flatten()
                    descs.append(desc) 
                    X.append(desc)
                    
            X_data=pd.DataFrame(X)
            X_data = X_data.values.reshape(X_data.shape[0], 50, 50,1)
            X_data=  X_data.astype('float32') 
            X_data /= 255 
            self.textEdit_11.setText(str(havadurum[saved_model.predict_classes(([X_data]))[0]]))
            print('bitti')
        elif self.label_26.text()=="Sift Algoritması ile HSV Uzayında Hazırlandı..":
            descs=[]
            X=[]
            image=cv2.cvtColor(image,cv2.COLOR_BGR2HSV) 
            print('burdayım')
            sift = cv2.xfeatures2d.SIFT_create(nfeatures=50)
            keypoints_sift,descriptors=sift.detectAndCompute(image,None)
            image = cv2.drawKeypoints(image, keypoints_sift, None)
            yükseklik=image.shape[0]
            genislik=image.shape[1] 
            n = 50
            print(keypoints_sift)
            for kp in keypoints_sift:
                xekseni,yekseni=int(kp.pt[0]),int(kp.pt[1])
                if xekseni-int(n/2)>0 and xekseni+(n/2)<genislik and yekseni-int(n/2)>0 and yekseni+int(n/2)<yükseklik:
                    region = image[yekseni-int(n/2):yekseni+int(n/2),xekseni-int(n/2):xekseni+int(n/2)]
                    h,s,v1=cv2.split(region)
                    desc, descs_img = daisy(v1,rings=2, histograms=6,
                           step=33,radius=7,visualize=True) 
                    desc=desc.reshape(desc.shape[0],desc.shape[1]*desc.shape[2])
                    desc=cv2.resize(desc, (50, 50))
                    desc=desc.flatten()
                    descs.append(desc) 
                    X.append(desc) 
            X_data=pd.DataFrame(X)
            X_data = X_data.values.reshape(X_data.shape[0], 50, 50,1)
            X_data=  X_data.astype('float32') 
            X_data /= 255 
            print(X_data)
            self.textEdit_11.setText(str(havadurum[saved_model.predict_classes(([X_data]))[0]])) 
            print('bitti')
                  
        elif self.label_26.text()=="Sift Algoritması ile CIE Uzayında Hazırlandı..":
            descs=[]
            X=[]
            image=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)   
            sift = cv2.xfeatures2d.SIFT_create(nfeatures=50)
            keypoints_sift,descriptors=sift.detectAndCompute(image,None)
            image = cv2.drawKeypoints(image, keypoints_sift, None)
            yükseklik=image.shape[0] #yükseklik
            genislik=image.shape[1] #genişlik
            n = 50
            for kp in keypoints_sift:       
                xekseni,yekseni=int(kp.pt[0]),int(kp.pt[1])
                if xekseni-int(n/2)>0 and xekseni+(n/2)<genislik and yekseni-int(n/2)>0 and yekseni+int(n/2)<yükseklik:
                    region = image[yekseni-int(n/2):yekseni+int(n/2),xekseni-int(n/2):xekseni+int(n/2)]                         
                    l,a,b1=cv2.split(region)
                    desc, descs_img = daisy(b1,rings=2, histograms=6,
                           step=33,radius=7,visualize=True) 
                    desc=desc.reshape(desc.shape[0],desc.shape[1]*desc.shape[2])
                    desc=cv2.resize(desc, (50, 50))
                    desc=desc.flatten()
                    descs.append(desc)
                    X.append(desc)
           
            X_data=pd.DataFrame(X)
            X_data = X_data.values.reshape(X_data.shape[0], 50, 50,1)
            X_data=  X_data.astype('float32') 
            X_data /= 255 
            self.textEdit_11.setText(str(havadurum[saved_model.predict_classes(([X_data]))[0]]))
            print('bitti')
           
        elif self.label_26.text()=="Surf Algoritması ile RGB Uzayında Hazırlandı..": 
            descs=[]
            X=[]
            surf = cv2.xfeatures2d.SURF_create()
            keypoints_surf, descriptors = surf.detectAndCompute(image, None)
            image = cv2.drawKeypoints(image, keypoints_surf, None)
            yükseklik=image.shape[0] 
            genislik=image.shape[1] 
            n = int(self.lineEdit.text())
            for kp in keypoints_surf:       
                xekseni,yekseni=int(kp.pt[0]),int(kp.pt[1])
                if xekseni-int(n/2)>0 and xekseni+(n/2)<genislik and yekseni-int(n/2)>0 and yekseni+int(n/2)<yükseklik:
                    region = image[yekseni-int(n/2):yekseni+int(n/2),xekseni-int(n/2):xekseni+int(n/2)]  
                    gray_image=color.rgb2gray(region)
                    desc, descs_img = daisy(gray_image,rings=2, histograms=6,
                           step=33,radius=7,visualize=True) 
                    desc=desc.reshape(desc.shape[0],desc.shape[1]*desc.shape[2])
                    desc=cv2.resize(desc, (50, 50))
                    desc=desc.flatten()
                    descs.append(desc) 
                X.append(desc)
         
            X_data=pd.DataFrame(X)
            X_data = X_data.values.reshape(X_data.shape[0], 50, 50,1)
            X_data=  X_data.astype('float32') 
            X_data /= 255 
            self.textEdit_11.setText(str(havadurum[saved_model.predict_classes(([X_data]))[0]]))
            print('bitti')
       
        elif self.label_26.text()=="Surf Algoritması ile HSV Uzayında Hazırlandı.. ":  
             descs=[]
             X=[]
             image=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)   
             surf = cv2.xfeatures2d.SURF_create()
             keypoints_surf, descriptors = surf.detectAndCompute(image, None)
             image = cv2.drawKeypoints(image, keypoints_surf, None)
             yükseklik=image.shape[0] #yükseklik
             genislik=image.shape[1] #genişlik
             n = int(self.lineEdit.text())
             for kp in keypoints_surf:       
                 xekseni,yekseni=int(kp.pt[0]),int(kp.pt[1])
                 if xekseni-int(n/2)>0 and xekseni+(n/2)<genislik and yekseni-int(n/2)>0 and yekseni+int(n/2)<yükseklik:
                     region = image[yekseni-int(n/2):yekseni+int(n/2),xekseni-int(n/2):xekseni+int(n/2)]  
                     h,s,v1=cv2.split(region)
                     desc, descs_img = daisy(v1,rings=2, histograms=6,
                            step=33,radius=7,visualize=True) 
                     desc=desc.reshape(desc.shape[0],desc.shape[1]*desc.shape[2])
                     desc=cv2.resize(desc, (50, 50))
                     desc=desc.flatten()
                     descs.append(desc) 
                     X.append(desc)
             
             X_data=pd.DataFrame(X)
             X_data = X_data.values.reshape(X_data.shape[0], 50, 50,1)
             X_data=  X_data.astype('float32') 
             X_data /= 255 
             self.textEdit_11.setText(str(havadurum[saved_model.predict_classes(([X_data]))[0]]))
             print('bitti')
            
        elif self.label_26.text()=="Surf Algoritması ile CIE Uzayında Hazırlandı.. ":
            descs=[]
            X=[]
            image=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
            surf = cv2.xfeatures2d.SURF_create()
            keypoints_surf, descriptors = surf.detectAndCompute(image, None)
            image = cv2.drawKeypoints(image, keypoints_surf, None)
            yükseklik=image.shape[0] #yükseklik
            genislik=image.shape[1] #genişlik
            n = 50
            for kp in keypoints_surf:       
                xekseni,yekseni=int(kp.pt[0]),int(kp.pt[1])
                if xekseni-int(n/2)>0 and xekseni+(n/2)<genislik and yekseni-int(n/2)>0 and yekseni+int(n/2)<yükseklik:
                    region = image[yekseni-int(n/2):yekseni+int(n/2),xekseni-int(n/2):xekseni+int(n/2)]  
                    l,a,b1=cv2.split(region)
                    desc, descs_img = daisy(b1,rings=2, histograms=6,
                           step=33,radius=7,visualize=True) 
                    desc=desc.reshape(desc.shape[0],desc.shape[1]*desc.shape[2])
                    desc=cv2.resize(desc, (50, 50))
                    desc=desc.flatten()
                    descs.append(desc) 
                    X.append(desc)
                    print('yaptım')
                    print(X)
            X_data=pd.DataFrame(X)
            X_data = X_data.values.reshape(X_data.shape[0], 50, 50,1)
            X_data=  X_data.astype('float32') 
            X_data /= 255 
            self.textEdit_11.setText(str(havadurum[saved_model.predict_classes(([X_data]))[0]]))
            print('bitti')
            
    def daisy(self):
        if self.path=="":           
           QMessageBox.about (self, "Uyarı" , "Klasör seçmeden bu işlemi yapamazsınız..!!" )
        else:
            self.test_size = self.comboBox.currentText()
            self.keypointcikarimi = self.comboBox_4.currentText()
            self.renkuzayi = self.comboBox_5.currentText()
            if self.keypointcikarimi=='Sift':
                if self.renkuzayi=='Hsv':
                    directories=os.listdir(self.path) 
                    descs=[]
                    y=[]
                    for label_no,directory in enumerate(directories):
                        files=os.listdir(self.path+directory)
                        print (files)
                        for file_name in files:
                            image = cv2.imread(self.path+directory+"/"+file_name)
                            image=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)   
                            sift = cv2.xfeatures2d.SIFT_create()
                            keypoints_sift,descriptors=sift.detectAndCompute(image,None)
                            image = cv2.drawKeypoints(image, keypoints_sift, None)
                            yükseklik=image.shape[0] #yükseklik
                            genislik=image.shape[1] #genişlik
                            n = int(self.lineEdit.text())
                            for kp in keypoints_sift:       
                                xekseni,yekseni=int(kp.pt[0]),int(kp.pt[1])
                                if xekseni-int(n/2)>0 and xekseni+(n/2)<genislik and yekseni-int(n/2)>0 and yekseni+int(n/2)<yükseklik:
                                    region = image[yekseni-int(n/2):yekseni+int(n/2),xekseni-int(n/2):xekseni+int(n/2)]                         
                                    h,s,v1=cv2.split(region)
                                    desc, descs_img = daisy(v1,rings=2, histograms=6,
                                           step=33,radius=7,visualize=True) 
                                    desc=desc.reshape(desc.shape[0],desc.shape[1]*desc.shape[2])
                                    desc=cv2.resize(desc, (50, 50))
                                    desc=desc.flatten()
                                    descs.append(desc) 
                                    y.append(label_no)   
                                    print (file_name, "islem tamam"
                                           ,label_no,desc.shape)     
                                    plt.imsave("./daisyresimler/sift/hsv/"+file_name, descs_img)
                    X=np.array(descs)
                    y=np.array(y)
                    print(X.shape)
                    print(y.shape)
                    pickle.dump(X,open('./sift/hsv/Xsifthsv.pkl', 'wb'))
                    pickle.dump(y,open('./sift/hsv/ysifthsv.pkl', 'wb'))  
                    print("islem tamam...")
                    QMessageBox.about (self, "Uyarı" , "İşlem Tamam" )
                if self.renkuzayi=='Rgb':
                    directories=os.listdir(self.path) 
                    descs=[]
                    y=[]
                    for label_no,directory in enumerate(directories):
                        files=os.listdir(self.path+directory)
                        print (files)
                        for file_name in files:
                            image = cv2.imread(self.path+directory+"/"+file_name)  
                            sift = cv2.xfeatures2d.SIFT_create()
                            keypoints_sift,descriptors=sift.detectAndCompute(image,None)
                            image = cv2.drawKeypoints(image, keypoints_sift, None)
                            image=color.rgb2gray(image)
                            yükseklik=image.shape[0]
                            genislik=image.shape[1] 
                            n = int(self.lineEdit.text())
                            for kp in keypoints_sift:
                                xekseni,yekseni=int(kp.pt[0]),int(kp.pt[1])
                                if xekseni-int(n/2)>0 and xekseni+(n/2)<genislik and yekseni-int(n/2)>0 and yekseni+int(n/2)<yükseklik:
                                    region = image[yekseni-int(n/2):yekseni+int(n/2),xekseni-int(n/2):xekseni+int(n/2)]
                                    desc, descs_img = daisy(region,rings=2, histograms=6,
                                           step=33,radius=7,visualize=True) 
                                    desc=desc.reshape(desc.shape[0],desc.shape[1]*desc.shape[2])
                                    desc=cv2.resize(desc, (50, 50))
                                    desc=desc.flatten()
                                    descs.append(desc) 
                                    y.append(label_no)   
                                    print (file_name, "islem tamam" 
                                           ,label_no,desc.shape)
                                    plt.imsave("./sift/rgb/"+file_name, descs_img)
                    X=np.array(descs)
                    y=np.array(y)
                    print("islem tamam...")
                    
                    pickle.dump(X, open('./sift/rgb/Xsiftrgb.pkl', 'wb'))
                    pickle.dump(y, open('./sift/rgb/ysiftrgb.pkl', 'wb'))
                    print(X)
                    print(y)
                if self.renkuzayi=='Cie':   
                    directories=os.listdir(self.path) 
                    descs=[]
                    y=[]
                    for label_no,directory in enumerate(directories):
                        files=os.listdir(self.path+directory)
                        print (files)
                        for file_name in files:
                            image = cv2.imread(self.path+directory+"/"+file_name)  
                            image=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
                            sift = cv2.xfeatures2d.SIFT_create()
                            keypoints_sift,descriptors=sift.detectAndCompute(image,None)
                            image = cv2.drawKeypoints(image, keypoints_sift, None)
                            yükseklik=image.shape[0] #yükseklik
                            genislik=image.shape[1] #genişlik
                            n = int(self.lineEdit.text())
                            for kp in keypoints_sift:       
                                xekseni,yekseni=int(kp.pt[0]),int(kp.pt[1])
                                if xekseni-int(n/2)>0 and xekseni+(n/2)<genislik and yekseni-int(n/2)>0 and yekseni+int(n/2)<yükseklik:
                                    region = image[yekseni-int(n/2):yekseni+int(n/2),xekseni-int(n/2):xekseni+int(n/2)]  
                                    l,a,b1=cv2.split(region)
                                    desc, descs_img = daisy(b1,rings=2, histograms=6,
                                           step=33,radius=7,visualize=True) 
                                    desc=desc.reshape(desc.shape[0],desc.shape[1]*desc.shape[2])
                                    desc=cv2.resize(desc, (50, 50))
                                    desc=desc.flatten()
                                    descs.append(desc) 
                                    y.append(label_no)   
                                    print (file_name, "islem tamam"
                                           ,label_no,desc.shape)
                                    plt.imsave("./sift/cie/"+file_name, descs_img)
                    X=np.array(descs)
                    y=np.array(y)
                    print("islem tamam...")
                    pickle.dump(X,open('./sift/cie/Xsiftcie.pkl', 'wb'))
                    pickle.dump(y,open('./sift/cie/ysiftcie.pkl', 'wb'))
                    
            if self.keypointcikarimi=='Surf':
                if self.renkuzayi=='Hsv':
                    directories=os.listdir(self.path) 
                    descs=[]
                    y=[]
                    for label_no,directory in enumerate(directories):
                        files=os.listdir(self.path+directory)
                        print (files)
                        for file_name in files:
                            image = cv2.imread(self.path+directory+"/"+file_name)
                            image=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)   
                            surf = cv2.xfeatures2d.SURF_create()
                            keypoints_surf, descriptors = surf.detectAndCompute(image, None)
                            image = cv2.drawKeypoints(image, keypoints_surf, None)
                            yükseklik=image.shape[0] #yükseklik
                            genislik=image.shape[1] #genişlik
                            n = int(self.lineEdit.text())
                            for kp in keypoints_surf:       
                                xekseni,yekseni=int(kp.pt[0]),int(kp.pt[1])
                                if xekseni-int(n/2)>0 and xekseni+(n/2)<genislik and yekseni-int(n/2)>0 and yekseni+int(n/2)<yükseklik:
                                    region = image[yekseni-int(n/2):yekseni+int(n/2),xekseni-int(n/2):xekseni+int(n/2)]  
                                    h,s,v1=cv2.split(region)
                                    desc, descs_img = daisy(v1,rings=2, histograms=6,
                                           step=33,radius=7,visualize=True) 
                                    desc=desc.reshape(desc.shape[0],desc.shape[1]*desc.shape[2])
                                    desc=cv2.resize(desc, (50, 50))
                                    desc=desc.flatten()
                                    descs.append(desc) 
                                    y.append(label_no)   
                                    print (file_name, "islem tamam"
                                           ,label_no,desc.shape)
                                    
                                    plt.imsave("./surf/hsv/"+file_name, descs_img)
                    X=np.array(descs)
                    y=np.array(y)
                    print("islem tamam...")
                    pickle.dump(X,open('./surf/hsv/Xsurfhsv.pkl', 'wb'))
                    pickle.dump(y,open('./surf/hsv/ysurfhsv.pkl', 'wb'))
                    
                if self.renkuzayi=='Rgb':
                    directories=os.listdir(self.path) 
                    descs=[]
                    y=[]
                    for label_no,directory in enumerate(directories):
                        files=os.listdir(self.path+directory)
                        print (files)
                        for file_name in files:
                            image = cv2.imread(self.path+directory+"/"+file_name)
                            surf = cv2.xfeatures2d.SURF_create()
                            keypoints_surf, descriptors = surf.detectAndCompute(image, None)
                            image = cv2.drawKeypoints(image, keypoints_surf, None)
                            yükseklik=image.shape[0] 
                            genislik=image.shape[1] 
                            n = int(self.lineEdit.text())
                            for kp in keypoints_surf:       
                                xekseni,yekseni=int(kp.pt[0]),int(kp.pt[1])
                                if xekseni-int(n/2)>0 and xekseni+(n/2)<genislik and yekseni-int(n/2)>0 and yekseni+int(n/2)<yükseklik:
                                    region = image[yekseni-int(n/2):yekseni+int(n/2),xekseni-int(n/2):xekseni+int(n/2)]  
                                    gray_image=color.rgb2gray(region)
                                    desc, descs_img = daisy(gray_image,rings=2, histograms=6,
                                           step=33,radius=7,visualize=True) 
                                    desc=desc.reshape(desc.shape[0],desc.shape[1]*desc.shape[2])
                                    desc=cv2.resize(desc, (50, 50))
                                    desc=desc.flatten()
                                    descs.append(desc) 
                                    y.append(label_no)   
                                    print (file_name, "islem tamam"
                                           ,label_no,desc.shape)
                                    plt.imsave("./surf/rgb/"+file_name, descs_img)
                    X=np.array(descs)
                    y=np.array(y)
                    print("islem tamam...")
                    pickle.dump(X,open('./surf/rgb/Xsurfrgb.pkl', 'wb'))
                    pickle.dump(y,open('./surf/rgb/ysurfrgb.pkl', 'wb'))
                if self.renkuzayi=='Cie':
                    directories=os.listdir(self.path) 
                    descs=[]
                    y=[]
                    for label_no,directory in enumerate(directories):
                        files=os.listdir(self.path+directory)
                        print (files)
                        for file_name in files:
                            image = cv2.imread(self.path+directory+"/"+file_name)
                            image=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
                            surf = cv2.xfeatures2d.SURF_create()
                            keypoints_surf, descriptors = surf.detectAndCompute(image, None)
                            image = cv2.drawKeypoints(image, keypoints_surf, None)
                            yükseklik=image.shape[0] #yükseklik
                            genislik=image.shape[1] #genişlik
                            n = int(self.lineEdit.text())
                            for kp in keypoints_surf:       
                                xekseni,yekseni=int(kp.pt[0]),int(kp.pt[1])
                                if xekseni-int(n/2)>0 and xekseni+(n/2)<genislik and yekseni-int(n/2)>0 and yekseni+int(n/2)<yükseklik:
                                    region = image[yekseni-int(n/2):yekseni+int(n/2),xekseni-int(n/2):xekseni+int(n/2)]  
                                    l,a,b1=cv2.split(region)
                                    desc, descs_img = daisy(b1,rings=2, histograms=6,
                                           step=33,radius=7,visualize=True) 
                                    desc=desc.reshape(desc.shape[0],desc.shape[1]*desc.shape[2])
                                    desc=cv2.resize(desc, (50, 50))
                                    desc=desc.flatten()
                                    descs.append(desc) 
                                    y.append(label_no)   
                                    print (file_name, "islem tamam"
                                           ,label_no,desc.shape)
                                    plt.imsave("./surf/cie/"+file_name, descs_img)
                          
                    X=np.array(descs)
                    y=np.array(y)
                    print("islem tamam...")
                    pickle.dump(X,open('./surf/cie/Xsurfcie.pkl', 'wb'))
                    pickle.dump(y,open('./surf/cie/ysurfcie.pkl', 'wb'))
            self.pushButton_2.setEnabled(True)
    def ayirma(self):
         self.test_size = self.comboBox.currentText()
         self.keypointcikarimi = self.comboBox_4.currentText()
         self.renkuzayi = self.comboBox_5.currentText()
         self.secim = self.comboBox_3.currentText()
         self.foldclass = self.comboBox_6.currentText() 
         self.kfoldndegeri = self.comboBox_2.currentText() 

         print(self.secim)
         if self.secim=='Hold-Out':
             if self.keypointcikarimi=='Sift':
                 if self.renkuzayi=='Rgb':
                     X=pickle.load(open('./sift/rgb/Xsiftrgb.pkl', 'rb'))
                     y=pickle.load(open('./sift/rgb/ysiftrgb.pkl', 'rb'))
                     X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=float(self.test_size), random_state=42, shuffle=True)
                     pickle.dump(X_train,open('./sift/rgb/X_train.pkl', 'wb'))    
                     pickle.dump(X_test,open('./sift/rgb/X_test.pkl', 'wb'))  
                     pickle.dump(y_train,open('./sift/rgb/y_train.pkl', 'wb'))  
                     pickle.dump(y_test,open('./sift/rgb/y_test.pkl', 'wb')) 
                     self.textEdit.setText(str(X_train))
                     self.textEdit_2.setText(str(X_test))
                     self.textEdit_3.setText(str(y_train))
                     self.textEdit_4.setText(str(y_test))
                     print (X_train.shape,X_test.shape)
                     print (y_train,y_test)
                     
                     
         elif self.secim=='K-Fold':
              if self.keypointcikarimi=='Sift':
                 if self.renkuzayi=='Rgb':
                      X=pickle.load(open('./sift/rgb/Xsiftrgb.pkl', 'rb'))
                      y=pickle.load(open('./sift/rgb/ysiftrgb.pkl', 'rb'))
                      adet=0
                      kf=KFold(n_splits=int(self.kfoldndegeri), random_state=1, shuffle=True)
                      for train_x, test_x in kf.split(X):
                          adet+=1
                          if(adet==int(self.foldclass)):
                              print("TRAIN:", train_x, "TEST:", test_x)
                              X_train, X_test = X[train_x], X[test_x]
                              y_train, y_test = y[train_x], y[test_x]
                              pickle.dump(X_train,open('./sift/rgb/X_train.pkl', 'wb'))    
                              pickle.dump(X_test,open('./sift/rgb/X_test.pkl', 'wb'))  
                              pickle.dump(y_train,open('./sift/rgb/y_train.pkl', 'wb'))  
                              pickle.dump(y_test,open('./sift/rgb/y_test.pkl', 'wb')) 
                              self.textEdit.setText(str(X_train))
                              self.textEdit_2.setText(str(X_test))
                              self.textEdit_3.setText(str(y_train))
                              self.textEdit_4.setText(str(y_test))
                              
                              
                          
         if self.secim=='Hold-Out':      
             if self.keypointcikarimi=='Surf':
                 if self.renkuzayi=='Rgb':
                     X=pickle.load(open('./surf/rgb/Xsurfrgb.pkl', 'rb'))
                     y=pickle.load(open('./surf/rgb/ysurfrgb.pkl', 'rb'))
                     X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=float(self.test_size), random_state=42, shuffle=True)
                     pickle.dump(X_train,open('./surf/rgb/X_train.pkl', 'wb'))    
                     pickle.dump(X_test,open('./surf/rgb/X_test.pkl', 'wb'))  
                     pickle.dump(y_train,open('./surf/rgb/y_train.pkl', 'wb'))  
                     pickle.dump(y_test,open('./surf/rgb/y_test.pkl', 'wb')) 
                     self.textEdit.setText(str(X_train))
                     self.textEdit_2.setText(str(X_test))
                     self.textEdit_3.setText(str(y_train))
                     self.textEdit_4.setText(str(y_test))
                     
                     
         elif self.secim=='K-Fold':
             if self.keypointcikarimi=='Surf':
                 if self.renkuzayi=='Rgb':
                     X=pickle.load(open('./surf/rgb/Xsurfrgb.pkl', 'rb'))
                     y=pickle.load(open('./surf/rgb/ysurfrgb.pkl', 'rb'))
                     adet=0
                     kf=KFold(n_splits=int(self.kfoldndegeri), random_state=1, shuffle=True)
                     for train_x, test_x in kf.split(X):
                          adet+=1
                          if(adet==int(self.foldclass)):
                             print("TRAIN:", train_x, "TEST:", test_x)
                             X_train, X_test = X[train_x], X[test_x]
                             y_train, y_test = y[train_x], y[test_x]
                             pickle.dump(X_train,open('./surf/rgb/X_train.pkl', 'wb'))    
                             pickle.dump(X_test,open('./surf/rgb/X_test.pkl', 'wb'))  
                             pickle.dump(y_train,open('./surf/rgb/y_train.pkl', 'wb'))  
                             pickle.dump(y_test,open('./surf/rgb/y_test.pkl', 'wb'))  
                             self.textEdit.setText(str(X_train))
                             self.textEdit_2.setText(str(X_test))
                             self.textEdit_3.setText(str(y_train))
                             self.textEdit_4.setText(str(y_test))
                             
                         
         if self.secim=='Hold-Out':      
             if self.keypointcikarimi=='Surf':
                 if self.renkuzayi=='Hsv':
                     X=pickle.load(open('./surf/hsv/Xsurfhsv.pkl', 'rb'))
                     self.y=pickle.load(open('./surf/hsv/ysurfhsv.pkl', 'rb'))  
                     X_train, X_test, y_train, y_test = train_test_split(X, self.y,  test_size=float(self.test_size), random_state=42, shuffle=True)
                     pickle.dump(X_train,open('./surf/hsv/X_train.pkl', 'wb'))    
                     pickle.dump(X_test,open('./surf/hsv/X_test.pkl', 'wb'))  
                     pickle.dump(y_train,open('./surf/hsv/y_train.pkl', 'wb'))  
                     pickle.dump(y_test,open('./surf/hsv/y_test.pkl', 'wb')) 
                     self.textEdit.setText(str(X_train))
                     self.textEdit_2.setText(str(X_test))
                     self.textEdit_3.setText(str(y_train))
                     self.textEdit_4.setText(str(y_test))
                     
                     
         elif self.secim=='K-Fold':
             if self.keypointcikarimi=='Surf':
                 if self.renkuzayi=='Hsv':
                     X=pickle.load(open('./surf/hsv/Xsurfhsv.pkl', 'rb'))
                     y=pickle.load(open('./surf/hsv/ysurfhsv.pkl', 'rb'))  
                     adet=0
                     kf=KFold(n_splits=int(self.kfoldndegeri), random_state=1, shuffle=True)
                     for train_x, test_x in kf.split(X):
                          adet+=1
                          if(adet==int(self.foldclass)):
                             print("TRAIN:", train_x, "TEST:", test_x)
                             X_train, X_test = X[train_x], X[test_x]
                             y_train, y_test = y[train_x], y[test_x]
                             pickle.dump(X_train,open('./surf/hsv/X_train.pkl', 'wb'))    
                             pickle.dump(X_test,open('./surf/hsv/X_test.pkl', 'wb'))  
                             pickle.dump(y_train,open('./surf/hsv/y_train.pkl', 'wb'))  
                             pickle.dump(y_test,open('./surf/hsv/y_test.pkl', 'wb'))  
                             self.textEdit.setText(str(X_train))
                             self.textEdit_2.setText(str(X_test))
                             self.textEdit_3.setText(str(y_train))
                             self.textEdit_4.setText(str(y_test))
                             
                         
         if self.secim=='Hold-Out':      
             if self.keypointcikarimi=='Sift':
                 if self.renkuzayi=='Hsv':
                     X=pickle.load(open('./sift/hsv/Xsifthsv.pkl', 'rb'))
                     self.y=pickle.load(open('./sift/hsv/ysifthsv.pkl', 'rb'))  
                     X_train, X_test, y_train, y_test = train_test_split(X, self.y,  test_size=float(self.test_size), random_state=42, shuffle=True)
                     pickle.dump(X_train,open('./sift/hsv/X_train.pkl', 'wb'))    
                     pickle.dump(X_test,open('./sift/hsv/X_test.pkl', 'wb'))  
                     pickle.dump(y_train,open('./sift/hsv/y_train.pkl', 'wb'))  
                     pickle.dump(y_test,open('./sift/hsv/y_test.pkl', 'wb')) 
                     self.textEdit.setText(str(X_train))
                     self.textEdit_2.setText(str(X_test))
                     self.textEdit_3.setText(str(y_train))
                     self.textEdit_4.setText(str(y_test))
                     
                     
         elif self.secim=='K-Fold':
             if self.keypointcikarimi=='Sift':
                 if self.renkuzayi=='Hsv':
                     X=pickle.load(open('./sift/hsv/Xsifthsv.pkl', 'rb'))
                     y=pickle.load(open('./sift/hsv/ysifthsv.pkl', 'rb'))   
                     adet=0
                     kf=KFold(n_splits=int(self.kfoldndegeri), random_state=1, shuffle=True)
                     for train_x, test_x in kf.split(X):
                          adet+=1
                          if(adet==int(self.foldclass)):
                             print("TRAIN:", train_x, "TEST:", test_x)
                             X_train, X_test = X[train_x], X[test_x]
                             y_train, y_test = y[train_x], y[test_x]
                             pickle.dump(X_train,open('./sift/hsv/X_train.pkl', 'wb'))    
                             pickle.dump(X_test,open('./sift/hsv/X_test.pkl', 'wb'))  
                             pickle.dump(y_train,open('./sift/hsv/y_train.pkl', 'wb'))  
                             pickle.dump(y_test,open('./sift/hsv/y_test.pkl', 'wb'))  
                             self.textEdit.setText(str(X_train))
                             self.textEdit_2.setText(str(X_test))
                             self.textEdit_3.setText(str(y_train))
                             self.textEdit_4.setText(str(y_test))
                         
                         
        
         if self.secim=='Hold-Out': 
            if self.keypointcikarimi=='Sift':
                if self.renkuzayi=='Cie':
                    X=pickle.load(open('./sift/cie/Xsiftcie.pkl', 'rb'))
                    self.y=pickle.load(open('./sift/cie/ysiftcie.pkl', 'rb'))         
                    X_train, X_test, y_train, y_test = train_test_split(X, self.y, test_size=float(self.test_size), random_state=42 ,shuffle=True)
                    pickle.dump(X_train,open('./sift/cie/X_train.pkl', 'wb'))    
                    pickle.dump(X_test,open('./sift/cie/X_test.pkl', 'wb'))  
                    pickle.dump(y_train,open('./sift/cie/y_train.pkl', 'wb'))  
                    pickle.dump(y_test,open('./sift/cie/y_test.pkl', 'wb')) 
                    self.textEdit.setText(str(X_train))
                    self.textEdit_2.setText(str(X_test))
                    self.textEdit_3.setText(str(y_train))
                    self.textEdit_4.setText(str(y_test))
                     
         elif self.secim=='K-Fold':
             if self.keypointcikarimi=='Sift':
                 if self.renkuzayi=='Cie':
                     X=pickle.load(open('./sift/cie/Xsiftcie.pkl', 'rb'))
                     y=pickle.load(open('./sift/cie/ysiftcie.pkl', 'rb'))  
                     adet=0
                     kf=KFold(n_splits=int(self.kfoldndegeri), random_state=1, shuffle=True)
                     for train_x, test_x in kf.split(X):
                          adet+=1
                          if(adet==int(self.foldclass)):
                             print("TRAIN:", train_x, "TEST:", test_x)
                             X_train, X_test = X[train_x], X[test_x]
                             y_train, y_test = y[train_x], y[test_x]
                             pickle.dump(X_train,open('./sift/cie/X_train.pkl', 'wb'))    
                             pickle.dump(X_test,open('./sift/cie/X_test.pkl', 'wb'))  
                             pickle.dump(y_train,open('./sift/cie/y_train.pkl', 'wb'))  
                             pickle.dump(y_test,open('./sift/cie/y_test.pkl', 'wb')) 
                             self.textEdit.setText(str(X_train))
                             self.textEdit_2.setText(str(X_test))
                             self.textEdit_3.setText(str(y_train))
                             self.textEdit_4.setText(str(y_test))
                             
                         
         if self.secim=='Hold-Out': 
             if self.keypointcikarimi=='Surf': 
                 if self.renkuzayi=='Cie':
                      X=pickle.load(open('./surf/cie/Xsurfcie.pkl', 'rb'))
                      self.y=pickle.load(open('./surf/cie/ysurfcie.pkl', 'rb'))         
                      X_train, X_test, y_train, y_test = train_test_split(X, self.y, test_size=float(self.test_size), random_state=42 ,shuffle=True)
                      pickle.dump(X_train,open('./surf/cie/X_train.pkl', 'wb'))    
                      pickle.dump(X_test,open('./surf/cie/X_test.pkl', 'wb'))  
                      pickle.dump(y_train,open('./surf/cie/y_train.pkl', 'wb'))  
                      pickle.dump(y_test,open('./surf/cie/y_test.pkl', 'wb')) 
                      self.textEdit.setText(str(X_train))
                      self.textEdit_2.setText(str(X_test))
                      self.textEdit_3.setText(str(y_train))
                      self.textEdit_4.setText(str(y_test))
                      
                      
         elif self.secim=='K-Fold':
             if self.keypointcikarimi=='Surf':
                 if self.renkuzayi=='Cie':
                     X=pickle.load(open('./surf/cie/Xsurfcie.pkl', 'rb'))
                     y=pickle.load(open('./surf/cie/ysurfcie.pkl', 'rb'))  
                     adet=0
                     kf=KFold(n_splits=int(self.kfoldndegeri), random_state=1, shuffle=True)
                     for train_x, test_x in kf.split(X):
                          adet+=1
                          if(adet==int(self.foldclass)):
                             print("TRAIN:", train_x, "TEST:", test_x)
                             X_train, X_test = X[train_x], X[test_x]
                             y_train, y_test = y[train_x], y[test_x]
                             pickle.dump(X_train,open('./surf/cie/X_train.pkl', 'wb'))    
                             pickle.dump(X_test,open('./surf/cie/X_test.pkl', 'wb'))  
                             pickle.dump(y_train,open('./surf/cie/y_train.pkl', 'wb'))  
                             pickle.dump(y_test,open('./surf/cie/y_test.pkl', 'wb')) 
                             self.textEdit.setText(str(X_train))
                             self.textEdit_2.setText(str(X_test))
                             self.textEdit_3.setText(str(y_train))
                             self.textEdit_4.setText(str(y_test))
                 
         self.loadetme()
        
    def loadetme(self):
         self.test_size = self.comboBox.currentText()
         self.keypointcikarimi = self.comboBox_4.currentText()
         self.renkuzayi = self.comboBox_5.currentText()
         if self.keypointcikarimi=='Sift':
             if self.renkuzayi=="Rgb":
                   self.X_train=pickle.load(open('./sift/rgb/X_train.pkl', 'rb'))
                   self.X_test=pickle.load(open('./sift/rgb/X_test.pkl', 'rb'))
                   self.y_train=pickle.load(open('./sift/rgb/y_train.pkl', 'rb'))
                   self.y_test=pickle.load(open('./sift/rgb/y_test.pkl', 'rb'))  
                   self.label_26.setText(str("Sift Algoritması ile RGB Uzayında Hazırlandı.."))
                   self.label_46.setText(str("Sift Algoritması ile RGB Uzayında Hazırlandı.."))
                   self.label_45.setText(str("Sift Algoritması ile RGB Uzayında Hazırlandı.."))
                   self.label_44.setText(str("Sift Algoritması ile RGB Uzayında Hazırlandı.."))
             if self.renkuzayi=="Hsv":
                   self.X_train=pickle.load(open('./sift/hsv/X_train.pkl', 'rb'))
                   self.X_test=pickle.load(open('./sift/hsv/X_test.pkl', 'rb'))
                   self.y_train=pickle.load(open('./sift/hsv/y_train.pkl', 'rb'))
                   self.y_test=pickle.load(open('./sift/hsv/y_test.pkl', 'rb'))     
                   self.label_26.setText(str("Sift Algoritması ile HSV Uzayında Hazırlandı.."))
                   self.label_44.setText(str("Sift Algoritması ile HSV Uzayında Hazırlandı.."))
                   self.label_45.setText(str("Sift Algoritması ile HSV Uzayında Hazırlandı.."))
                   self.label_46.setText(str("Sift Algoritması ile HSV Uzayında Hazırlandı.."))
             if self.renkuzayi=="Cie":
                  self.X_train=pickle.load(open('./sift/cie/X_train.pkl', 'rb'))
                  self.X_test=pickle.load(open('./sift/cie/X_test.pkl', 'rb'))
                  self.y_train=pickle.load(open('./sift/cie/y_train.pkl', 'rb'))
                  self.y_test=pickle.load(open('./sift/cie/y_test.pkl', 'rb')) 
                  self.label_26.setText(str("Sift Algoritması ile CIE Uzayında Hazırlandı.."))
                  self.label_45.setText(str("Sift Algoritması ile CIE Uzayında Hazırlandı.."))
                  self.label_44.setText(str("Sift Algoritması ile CIE Uzayında Hazırlandı.."))
                  self.label_46.setText(str("Sift Algoritması ile CIE Uzayında Hazırlandı.."))
         if self.keypointcikarimi=='Surf':
             if self.renkuzayi=="Rgb":
                   self.X_train=pickle.load(open('./surf/rgb/X_train.pkl', 'rb'))
                   self.X_test=pickle.load(open('./surf/rgb/X_test.pkl', 'rb'))
                   self.y_train=pickle.load(open('./surf/rgb/y_train.pkl', 'rb'))
                   self.y_test=pickle.load(open('./surf/rgb/y_test.pkl', 'rb')) 
                   self.label_26.setText(str("Surf Algoritması ile RGB Uzayında Hazırlandı.. "))
                   self.label_44.setText(str("Surf Algoritması ile RGB Uzayında Hazırlandı.. "))
                   self.label_45.setText(str("Surf Algoritması ile RGB Uzayında Hazırlandı.. "))
                   self.label_46.setText(str("Surf Algoritması ile RGB Uzayında Hazırlandı.. "))
             if self.renkuzayi=="Hsv":
                   self.X_train=pickle.load(open('./surf/hsv/X_train.pkl', 'rb'))
                   self.X_test=pickle.load(open('./surf/hsv/X_test.pkl', 'rb'))
                   self.y_train=pickle.load(open('./surf/hsv/y_train.pkl', 'rb'))
                   self.y_test=pickle.load(open('./surf/hsv/y_test.pkl', 'rb'))  
                   self.label_26.setText(str("Surf Algoritması ile HSV Uzayında Hazırlandı.. "))
                   self.label_44.setText(str("Surf Algoritması ile HSV Uzayında Hazırlandı.. "))
                   self.label_45.setText(str("Surf Algoritması ile HSV Uzayında Hazırlandı.. "))
                   self.label_46.setText(str("Surf Algoritması ile HSV Uzayında Hazırlandı.. "))
             if self.renkuzayi=="Cie":
                  self.X_train=pickle.load(open('./surf/cie/X_train.pkl', 'rb'))
                  self.X_test=pickle.load(open('./surf/cie/X_test.pkl', 'rb'))
                  self.y_train=pickle.load(open('./surf/cie/y_train.pkl', 'rb'))
                  self.y_test=pickle.load(open('./surf/cie/y_test.pkl', 'rb'))
                  self.label_26.setText(str("Surf Algoritması ile CIE Uzayında Hazırlandı.. "))
                  self.label_44.setText(str("Surf Algoritması ile CIE Uzayında Hazırlandı.. "))
                  self.label_45.setText(str("Surf Algoritması ile CIE Uzayında Hazırlandı.. "))
                  self.label_46.setText(str("Surf Algoritması ile CIE Uzayında Hazırlandı.. "))
                 
    def makineogrenmesi(self):
        self.loadetme()
        clf = LogisticRegression() 
        clf.fit(self.X_train, self.y_train)
        self.y_pred = clf.predict(self.X_test)
        self.textEdit_7.setText(str("Başarı oranı:{:0.2f}".format(accuracy_score(self.y_test, self.y_pred))))
        confMat = confusion_matrix(self.y_test, self.y_pred)
        print(confMat)
        TN = confMat[0,0]
        TP = confMat[1,1]
        FN = confMat[1,0]
        FP = confMat[0,1]
        sensivity = float(TP)/(TP+FN)
        specifity = float(TN)/(TN+FP)
        self.textEdit_8.setText(str('Sensivity:%.3f' %sensivity))
        self.textEdit_9.setText(str('Specifity:%.3f' %specifity))
        
        pred_prob1 = clf.predict_proba(self.X_test)
        fpr, tpr, thresh = roc_curve(self.y_test, pred_prob1[:,1], pos_label=1)
        plt.figure(figsize=(4.5,2.5))
        plt.plot(fpr, tpr,color='red')
        plt.plot([0,1], [0,1], linestyle='--', color='green')
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')
        plt.savefig('./roclog.png')
        self.pixmap = QPixmap("./roclog.png") 
        self.label_29.setPixmap(self.pixmap)
        

        from sklearn.metrics import plot_confusion_matrix
        plot_confusion_matrix(clf,self.X_test, self.y_test,cmap=plt.cm.Blues)
        plt.savefig("./abc.png")
        plt.show()
        self.pixmap = QPixmap("./abc.png") 
        self.label_36.setPixmap(self.pixmap)
        plt.show()

     
    def egitme(self):
         self.X_train = self.X_train.reshape(self.X_train.shape[0], 50, 50,1)
         self.X_test = self.X_test.reshape(self.X_test.shape[0], 50, 50,1)
         self.X_train = self.X_train.astype('float32')
         self.X_test = self.X_test.astype('float32')
         self.X_train /= 255
         self.X_test /= 255
         self.y_train = np_utils.to_categorical(self.y_train,4)
         self.y_test = np_utils.to_categorical(self.y_test, 4)
         model = Sequential()
         model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(50,50,1)))
         model.add(Convolution2D(32, 3, 3, activation='relu'))
         model.add(MaxPooling2D(pool_size=(2,2)))
         model.add(Dropout(0.25))
         model.add(Flatten())
         model.add(Dense(128, activation='relu'))
         model.add(Dropout(0.5))
         model.add(Dense(4, activation='softmax'))
         model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
         model.summary()
         epochsdeger = int(self.lineEdit_2.text())
         modelfit=model.fit(self.X_train, self.y_train, batch_size=1, epochs=epochsdeger, verbose=1, validation_data=(self.X_test, self.y_test))
         scores = model.evaluate(self.X_test, self.y_test, verbose=1) 
         model.save("den.h5")
         self.textEdit_5.setText("%{:.2f}".format(scores[1]))
         self.textEdit_6.setText("%{:.2f}".format(scores[0]))
         
        
         plt.subplot(2,1,1)
         plt.figure(figsize=(4.5,2.5))
         plt.plot(modelfit.history['accuracy']) #
         plt.plot(modelfit.history['val_accuracy'])
         plt.title('model accuracy')
         plt.ylabel('accuracy')
         plt.xlabel('epoch')
         plt.legend(['train', 'test'], loc='lower right')
         plt.savefig('./AccVal_acc.png')
         self.pixmap = QPixmap("./AccVal_acc.png") 
         self.label_7.setPixmap(self.pixmap)
         

         plt.subplot(1,2,1)
         plt.figure(figsize=(4.5,2.5))
         plt.plot(modelfit.history['loss']) 
         plt.plot(modelfit.history['val_loss'])
         plt.title('model loss')
         plt.ylabel('loss')
         plt.xlabel('epoch')
         plt.legend(['train', 'test'], loc='lower right')
         plt.savefig('./loss.png')
         self.pixmap = QPixmap("./loss.png") 
         self.label_32.setPixmap(self.pixmap)
         
         fpr =dict()
         tpr = dict()
         threshold =dict()
         
         self.y_pred = model.predict(self.X_test)
         self.y_test = np.argmax(self.y_test, axis=1)
         for i in range(4):
             fpr[i], tpr[i], threshold[i] = roc_curve(self.y_test, self.y_pred[:,i], pos_label=i)
         def roc_ciz(fpr, tpr):
             plt.figure(figsize=(4.5,2.5))
             plt.plot(fpr[0], tpr[0], color='pink', label='Cloudy')
             plt.plot(fpr[1], tpr[1], color='purple', label='Rain')
             plt.plot(fpr[2], tpr[2], color='red', label='Shine')
             plt.plot(fpr[3], tpr[3], color='green', label='Sunrise')
             #plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
             plt.xlabel('False Positive Rate')
             plt.ylabel('True Positive Rate')
             plt.title('ROC Curve')
             plt.legend()
             plt.savefig('./ROCE.png')
             plt.show()
         roc_ciz(fpr, tpr)
         self.pixmap = QPixmap("./ROCE.png") 
         self.label_37.setPixmap(self.pixmap)
         
         import seaborn as sns
         self.y_pred = np.argmax(self.y_pred, axis=1)
         cm = confusion_matrix(self.y_test, self.y_pred)
         print(cm)
         plt.figure(figsize=(4.5,2.5))
         ax= plt.subplot()
         sns.heatmap(cm, annot=True, ax = ax,cmap=plt.cm.Blues)
         ax.set_xlabel('Tahmin');ax.set_ylabel('Gerçek'); 
         ax.set_title('Confusion Matrix'); 
         ax.xaxis.set_ticklabels(['Cloudy', 'Rain','Shine','Sunrise'])
         ax.yaxis.set_ticklabels(['Cloudy', 'Rain','Shine','Sunrise'],rotation=45)
         plt.savefig("./CONFYAP.png")
         self.pixmap = QPixmap("./CONFYAP.png")
         self.label_16.setPixmap(self.pixmap)
    
         
         
         
      
        
         
         
    

             
    
            
             
        
            
       
       