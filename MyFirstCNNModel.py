import tensorflow as tf 
from NetworkBuilder import NetworkBuilder 
from DataSetGenerator import DataSetGenerator
import datetime 
import numpy as np 
import os

#placeholders for the input image

with tf.name_scope("Input") as scope:
    input_img = tf.placeholder(dtype='float', shape=[None, 128, 128, 1], name="input")

with tf.name_scope("Target") as scope:
    target_labels = tf.placeholder(dtype='float', shape=[None, 5], name="Targets")

with tf.name_scope("keep_prob_input"):
    keep_prob = tf.placeholder(dtype='float', name='keep_prob')

    

nb = NetworkBuilder()

with tf.name_scope("ModelV2") as scope:
    model = input_img
    model = nb.attach_conv_layer(model, 32, summary=True)
    model = nb.attach_relu_layer(model)
    model = nb.attach_conv_layer(model, 32, summary=True)
    model = nb.attach_relu_layer(model)
    model = nb.attach_pooling_layer(model)

    model = nb.attach_conv_layer(model, 64, summary=True)
    model = nb.attach_relu_layer(model)
    model = nb.attach_conv_layer(model, 64, summary=True)
    model = nb.attach_relu_layer(model)
    model = nb.attach_pooling_layer(model)

    model = nb.attach_conv_layer(model, 128, summary=True)
    model = nb.attach_relu_layer(model)
    model = nb.attach_conv_layer(model, 128, summary=True)
    model = nb.attach_relu_layer(model)
    model = nb.attach_pooling_layer(model)

    model = nb.flatten(model)
    model = nb.attach_dense_layer(model, 200, summary=True)
    model = nb.attach_sigmoid_layer(model)
    model = nb.attach_dense_layer(model, 32, summary=True)
    model = nb.attach_sigmoid_layer(model)
    model = nb.attach_dense_layer(model, 5) 
    prediction = nb.attach_softmax_layer(model)

    
    
with tf.name_scope("Optimization") as scope:
    global_step = tf.Variable(0, name='global_step', trainable=False)
    cost = tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=target_labels)
    cost = tf.reduce_mean(cost)
    tf.summary.scalar("cost", cost)

    optimizer = tf.train.AdamOptimizer().minimize(cost,global_step=global_step)

with tf.name_scope('accuracy') as scope:
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(target_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))






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
       # print("shaped")
        
        # padding
        if aspect >= 1: # horizontal image
            #new_shape = list(img.shape)
            new_shape[0] = w
            new_shape[1] = w
            new_shape = tuple(new_shape)
            new_img=np.zeros(new_shape, dtype=np.uint8)
            h_offset=int((w-h)/2)
            new_img[h_offset:h_offset+h, :, :] = img.copy()

        elif aspect < 1: # vertical image
            #new_shape = list(img.shape)
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



    
    
#***************************** train the model**************************************

dg = DataSetGenerator("C:\\Users\\Learner\\Desktop\\SIGN2WORD\\train")

epochs = 50
batchSize =10

saver = tf.train.Saver()   #create a train saver object
model_save_path="C:\\Users\\Learner\\Desktop\\SIGN2WORD\\saved model v2" # train windows üzerinde yapildi
model_name='model'

with tf.Session() as sess:

    summaryMerged = tf.summary.merge_all()

    filename = "C:\\Users\\Learner\\Desktop\\SIGN2WORD\\summary_log\\run"
    tf.global_variables_initializer().run()
    
  
    saver.restore(sess,tf.train.latest_checkpoint(model_save_path))

    writer = tf.summary.FileWriter(filename, sess.graph)

    


   

    for epoch in range(epochs):
        batches = dg.get_mini_batches(batchSize,(128,128), allchannel=False)
        for imgs ,labels in batches:
            imgs=np.divide(imgs, 255)
            error,sumOut, acu, steps,_ = sess.run([cost, summaryMerged, accuracy,global_step,optimizer],
                                            feed_dict={input_img: imgs, target_labels: labels})
            writer.add_summary(sumOut, steps)
            
            
            print("Döngü=", epoch," *** ", "Toplam Resim=", steps*batchSize," *** ", "Hata=", error, " *** ","Dogruluk=", acu)

            if steps % 100 == 0:
                print("Model Kayit ediliyor...")
                saver.save(sess, model_save_path+"\\"+model_name, global_step=steps)
                print("Model Kayit edildi")









    



