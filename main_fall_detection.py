from keras.models import load_model
from keras.models import model_from_json
import numpy as np
import pickle
import sys
import os
#from PIL import Image
import datetime
import imageio
import cv2

L = 20

# flow has flow in both x and y dim
flow        = np.zeros(shape=(224,224,2*L,1), dtype=np.float64)
# flow in x dim
flow_x_all  = np.zeros(shape=(224,224,L), dtype=np.float64)
# flow in y dim
flow_y_all  = np.zeros(shape=(224,224,L), dtype=np.float64)

model_mobilenet_fall = load_model('model_mobilenet_fall.h5')
    
# Setup up the parameters to compute flow
flow_struct = cv2.DualTVL1OpticalFlow_create(   tau = 0.25,
                                                theta = 0.3,
                                                nscales = 5,
                                                warps = 5,
                                                epsilon = 0.01,
                                                innnerIterations = 10,
                                                outerIterations = 5,
                                                scaleStep = 0.8,
                                                gamma = 0.0,
                                                medianFiltering = 5,
                                                useInitialFlow = False 
                                                )
        
        
y_all = []                                            
iImage = 0
while(1):
    print('iImage = ' + str(iImage))
    
    I0 = cv2.imread('/tmp/figs/fig' + str(iImage) + '.jpg'  ,cv2.IMREAD_GRAYSCALE)
    I1 = cv2.imread('/tmp/figs/fig' + str(iImage+1) + '.jpg',cv2.IMREAD_GRAYSCALE)
            
    I0 = cv2.resize(I0,(224,224))
    I1 = cv2.resize(I1,(224,224))

    
    # Compute optical flow
    if 0:
        curr_flow = cv2.calcOpticalFlowFarneback(I0,I1, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    else:
        curr_flow = flow_struct.calc(I0.astype('uint8'),I1.astype('uint8'),None)
        
    curr_flow = 8*curr_flow
    
    # For the first L images just fill the buffer
    if cntImage < L:
        flow_x_all[:,:,cntImage] = curr_flow[:,:,0]
        flow_y_all[:,:,cntImage] = curr_flow[:,:,1]
        image_buffer.append(I0.astype('uint8'))
        
        cntImage = cntImage + 1
        continue

    # Put new data at the end of the array
    flow_x_all[:,:,0:L-1]   = flow_x_all[:,:,1:L].copy()
    flow_y_all[:,:,0:L-1]   = flow_y_all[:,:,1:L].copy()
    flow_x_all[:,:,L-1]     = curr_flow[:,:,0].copy()
    flow_y_all[:,:,L-1]     = curr_flow[:,:,1].copy()
    cntImage = cntImage + 1
    
    # Interleave x and y data in the final flow array
    tmp1 = np.arange(0,2*L,2)
    tmp2 = np.arange(1,2*L+1,2)
    flow[:,:,tmp1,0] = flow_x_all
    flow[:,:,tmp2,0] = flow_y_all
    
    # 0th dimension should be of dimension batch size for keras
    # Before this step flow has dimension 224x224x20x1
    # After this step flow has dimension 1x224x224x20
    flow_transpose = np.transpose(flow,(3,0,1,2))
    
    # Run NN on the input
    y = model_mobilenet_fall.predict(flow_transpose)
    y_all.append(y)
    if y == 1:
        publish_to_aws_sns(1)
    
