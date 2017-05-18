#!/usr/bin/python3

#import tensorflow as tf
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3) ## Configure the fraction here
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

import keras
import keras.backend as K
#K.set_session(sess)

import numpy as np
from subprocess import Popen, PIPE, DEVNULL
import struct
import sys
import os 
import time
import kaldiIO

#### Run this script from /home/tejas/experiments/en-US_429_512_256_512_913_retraining_after_pruning/pavans_1024x4

def writeUtteranceText(uid, featMat, fp):
    fp.write(uid + ' [\n')
    for row in featMat:
        row.tofile(fp, sep=' ')
        f.write('\n')
    fp.write(' ]\n')

def arithmetic_mean(featMat, model_list, weights):
    prediction = np.zeros((featMat.shape[0], model_list[0].output_shape[1]))
    for m, wt in zip(model_list, weights):
        prediction += wt * m.predict(featMat)
    am = prediction/len(model_list)
    return am

def geometric_mean(featMat, model_list, weights):
    prediction = np.ones((featMat.shape[0], model_list[0].output_shape[1]))
    for m, wt in zip(model_list, weights):
        prediction *= m.predict(featMat) ** wt
    gm = np.power(prediction, 1/np.ceil(sum(weights)))
    normalized_gm = gm.T / gm.sum(axis=1)
    return normalized_gm.T
    return gm


if __name__ == '__main__':

    test = sys.argv[1]
    location = '/home/tejas/experiments/en-US_429_512_256_512_913_retraining_after_pruning/pavans_1024x4' 
    data = '../en-US/data/test_' + test + '_i3feat' 
    encoding = sys.stdout.encoding 

    model_list = ['outdir_en-US_429_1024x5_1375/dnn.nnet.h15',
                  'outdir_en-US_429_1024x4_1375/dnn.nnet.h15',
                  'outdir_en-US_429_2048x4_1375/dnn.nnet.h15']

    weights = list(np.ones(len(model_list)))
    #weights = [0.45, 0.35, 0.2] 
    
    print ('Loading models ....')
    model_list = [keras.models.load_model('pavans_1024x4/' + m) for m in model_list]
    print ('Loading models, DONE')

    # Splice features
    p1 = Popen (['splice-feats','--print-args=false','--left-context=5','--right-context=5',
            'scp:' + data + '/feats.scp','ark:-'], stdout=PIPE)

    f = open (location + '/temp.ark', 'wb')
    st = time.time()
    while True:
        uid, featMat = kaldiIO.readUtterance(p1.stdout)
        if uid == None:
            print ('Reached the end of feats.scp')
            break
        print ('Processing utt id: ' + uid)
        #log_avg_prediction = np.log(arithmetic_mean(featMat, model_list, weights))
        log_avg_prediction = np.log(geometric_mean(featMat, model_list, weights))  
        #writeUtteranceText(uid, log_avg_prediction, f)
        kaldiIO.writeUtterance(uid, log_avg_prediction, f, encoding)
    et = time.time()
    print ('Time taken: ', et-st)
    f.close()
