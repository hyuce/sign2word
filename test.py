import tensorflow as tf 
import NetworkBuilder 
from DataSetGenerator import DataSetGenerator
import numpy as np 
import os
import cv2




def resizeAndPad(img, size):
        h, w = img.shape[:2]

        sh, sw = size
        
        # interpolation method
        if h >= sh or w >= sw:  # shrinking image
            interp = cv2.INTER_AREA
        else: # stretching image
            interp = cv2.INTER_CUBIC

        # aspect ratio of image
        aspect = w/h

        new_shape = list(img.shape)
        
        
        # padding
        if aspect >= 1: # horizontal image
            
            new_shape[0] = w
            new_shape[1] = w
            new_shape = tuple(new_shape)
            new_img=np.zeros(new_shape, dtype=np.uint8)
            h_offset=int((w-h)/2)
            new_img[h_offset:h_offset+h, :, :] = img.copy()

        elif aspect < 1: # vertical image
            
            new_shape[0] = h
            new_shape[1] = h
            new_shape = tuple(new_shape)
            new_img = np.zeros(new_shape,dtype=np.uint8)
            w_offset = int((h-w) / 2)
            new_img[:, w_offset:w_offset + w, :] = img.copy()
        else:
            new_img = img.copy()
        # scale and pad
        scaled_img = cv2.resize(new_img, size, interpolation=interp)
        return scaled_img



import tensorflow as tf 
from NetworkBuilder import NetworkBuilder 
from DataSetGenerator import DataSetGenerator
import datetime 
import numpy as np 
import os


model_save_path="Modelin kaydedildiği Klasör  Yol Adresi"


from numpy import zeros, newaxis
import time

cap = cv2.VideoCapture(0)
frame_counter=0


with tf.Session() as sess:

        saver = tf.train.import_meta_graph('model-xxxxx.meta isimli dosyanın kayıt ediliği dizinin path'')
        saver.restore(sess,tf.train.latest_checkpoint(model_save_path))
)
        input_img = tf.get_default_graph().get_tensor_by_name('Input/input:0')
        prediction = tf.get_default_graph().get_tensor_by_name('ModelV2/Activation_8/Softmax:0')
        target_labels = tf.get_default_graph().get_tensor_by_name('Target/Targets:0')
      
        frame_counter =0
        eski_Harf = 'aaaa'
        while True:
                
            ret, frame = cap.read()
            frame = cv2.flip(frame,1)
            
            frame_counter = frame_counter + 1
            cv2.rectangle(frame,(100,100),(420,420),(0,0,100),8)
            font = cv2.FONT_HERSHEY_TRIPLEX
            cv2.putText(frame,'El Alani',(120,80),font,1.5,(0,0,100),2,cv2.LINE_AA)
            cv2.imshow('frame',frame)
            elAlani = frame[100:420, 100:420]
            grayElAlani = cv2.cvtColor(elAlani, cv2.COLOR_BGR2GRAY)
            gaussianElAlani = cv2.GaussianBlur(grayElAlani,(5,5),0)
            ret1,thresholdingElAlani = cv2.threshold(gaussianElAlani,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            
        
            cv2.imshow('El Alani',thresholdingElAlani)

             

            
            
            img = thresholdingElAlani
            
           
            
            img.resize((320,320,3), refcheck=False)
            
            image = []
      
            img = resizeAndPad(img,(128,128))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.reshape(img, (img.shape[0], img.shape[1], 1))
            image.append(img)
            testImage=np.array(image,dtype=np.uint8)
            np.reshape(testImage, [128, 128, 1])

            Son_Tahmin = sess.run([prediction],feed_dict={input_img: testImage})
            
            tahminler = [Son_Tahmin[0][0][0],Son_Tahmin[0][0][1],Son_Tahmin[0][0][2],Son_Tahmin[0][0][3],Son_Tahmin[0][0][4]]
            
            if frame_counter % 20 == 0:
                                if max(tahminler)== Son_Tahmin[0][0][0] :
                                    print('A',end='')
                                    
                                if max(tahminler)== Son_Tahmin[0][0][1]:
                                    print('B',end='')
                                    
                                if max((tahminler)== Son_Tahmin[0][0][2])  :
                                    print('C',end='')
                                    
                                if max(tahminler)== (Son_Tahmin[0][0][3]) :
                                   print('D',end='')

                                if max(tahminler)== (Son_Tahmin[0][0][4]) :
                                   print('E',end='')
                                    

                                  


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.realease()
        cv2.destroyAllWindows()        
        


        

