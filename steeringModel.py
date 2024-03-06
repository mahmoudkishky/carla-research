import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Dropout
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow import one_hot
from tensorflow.keras.layers import Concatenate, Add
import numpy as np
import scipy.stats as stats
import trainingConfigs
import sys


def cooccurance_matrix(num_class = 117, sigma = 1):
    
    s =  np.zeros((num_class,num_class), dtype = 'float32')
    
    for branch in trainingConfigs.BRANCHES:
        
        class_idx = sum(trainingConfigs.BRANCHE_CLASS_COUNT[0 : trainingConfigs.BRANCHES.index(branch)])
        
        for i in range(class_idx, class_idx + trainingConfigs.BRANCHE_CLASS_COUNT[trainingConfigs.BRANCHES.index(branch)]):
            
            x = np.linspace(0, trainingConfigs.BRANCHE_CLASS_COUNT[trainingConfigs.BRANCHES.index(branch)] - 1, num = trainingConfigs.BRANCHE_CLASS_COUNT[trainingConfigs.BRANCHES.index(branch)])
            y = stats.norm.pdf(x, i - class_idx, sigma)
            #y = stats.laplace.pdf(x, i - class_idx, sigma)
            y = [-1 if  el < 0.15 else el for el in y]
            s[i, class_idx : (class_idx + trainingConfigs.BRANCHE_CLASS_COUNT[trainingConfigs.BRANCHES.index(branch)])] = y
            #s[i,i] = 0

    return s

def cil_ising_loss(y_true, y_pred):
   
  s = cooccurance_matrix(num_class = sum(trainingConfigs.BRANCHE_CLASS_COUNT), sigma = 1)
  cce = tf.keras.losses.CategoricalCrossentropy()

  #mask_l = [0] * sum(trainingConfigs.BRANCHE_CLASS_COUNT)
  #mask_r = [0] * sum(trainingConfigs.BRANCHE_CLASS_COUNT)
  #mask_s = [0] * sum(trainingConfigs.BRANCHE_CLASS_COUNT)
  #mask_f = [0] * sum(trainingConfigs.BRANCHE_CLASS_COUNT)

  #mask_l[0 : sum(trainingConfigs.BRANCHE_CLASS_COUNT[0 : 1])]   = [1] * trainingConfigs.BRANCHE_CLASS_COUNT[0]
  #mask_r[sum(trainingConfigs.BRANCHE_CLASS_COUNT[0 : 1]) : sum(trainingConfigs.BRANCHE_CLASS_COUNT[0 : 2])]  = [1] * trainingConfigs.BRANCHE_CLASS_COUNT[1]
  #mask_s[sum(trainingConfigs.BRANCHE_CLASS_COUNT[0 : 2]) : sum(trainingConfigs.BRANCHE_CLASS_COUNT[0 : 3])]  = [1] * trainingConfigs.BRANCHE_CLASS_COUNT[2]
  #mask_f[sum(trainingConfigs.BRANCHE_CLASS_COUNT[0 : 3]) : sum(trainingConfigs.BRANCHE_CLASS_COUNT[0 : 4])]  = [1] * trainingConfigs.BRANCHE_CLASS_COUNT[3]

  #mask_l = np.array([mask_l,] * int(trainingConfigs.BATCH_SIZE / 4))
  #mask_r = np.array([mask_r,] * int(trainingConfigs.BATCH_SIZE / 4))
  #mask_s = np.array([mask_s,] * int(trainingConfigs.BATCH_SIZE / 4))
  #mask_f = np.array([mask_f,] * int(trainingConfigs.BATCH_SIZE / 4))

  #mask = np.vstack((mask_l, mask_r, mask_s, mask_f))

  #y_mask = y_pred * mask

  l1 =  (1 - trainingConfigs.COOCCURANCE_WEIGHT) * cce(y_true, y_pred)
  l2 = (trainingConfigs.COOCCURANCE_WEIGHT) * tf.reduce_mean(tf.linalg.matmul(tf.linalg.matmul(y_pred, s), tf.transpose(y_pred)))
  #l2 = (trainingConfigs.COOCCURANCE_WEIGHT) * tf.reduce_mean(tf.linalg.matmul(tf.linalg.matmul(y_pred, s), tf.transpose(y_pred)))
  #l2 = (trainingConfigs.COOCCURANCE_WEIGHT) * tf.reduce_mean(tf.linalg.diag_part(tf.linalg.matmul(tf.linalg.matmul(y_mask, s), tf.transpose(y_mask))))

  return tf.math.add(l1, l2)

def cil_loss(y_true, y_pred):

  mse = tf.keras.losses.MeanSquaredError()
  batch_nav_cmds = np.array(int(trainingConfigs.BATCH_SIZE / 4 ) * [trainingConfigs.BRANCHES.index('L')] + 
                 int(trainingConfigs.BATCH_SIZE / 4 ) * [trainingConfigs.BRANCHES.index('R')] +
                 int(trainingConfigs.BATCH_SIZE / 4 ) * [trainingConfigs.BRANCHES.index('S')] + 
                 int(trainingConfigs.BATCH_SIZE / 4 ) * [trainingConfigs.BRANCHES.index('F')])

  nav_cmds_mask = tf.convert_to_tensor(one_hot(batch_nav_cmds, depth = 4))
  y_mask = tf.reduce_sum(y_pred * nav_cmds_mask, axis = 1)

  return mse(y_true, y_mask)


def cil_class_loss(y_true, y_pred):

  cce = tf.keras.losses.CategoricalCrossentropy()

  mask_l = [0] * sum(trainingConfigs.BRANCHE_CLASS_COUNT)
  mask_r = [0] * sum(trainingConfigs.BRANCHE_CLASS_COUNT)
  mask_s = [0] * sum(trainingConfigs.BRANCHE_CLASS_COUNT)
  mask_f = [0] * sum(trainingConfigs.BRANCHE_CLASS_COUNT)

  mask_l[0 : sum(trainingConfigs.BRANCHE_CLASS_COUNT[0 : 1])]   = [1] * trainingConfigs.BRANCHE_CLASS_COUNT[0]
  mask_r[sum(trainingConfigs.BRANCHE_CLASS_COUNT[0 : 1]) : sum(trainingConfigs.BRANCHE_CLASS_COUNT[0 : 2])]  = [1] * trainingConfigs.BRANCHE_CLASS_COUNT[1]
  mask_s[sum(trainingConfigs.BRANCHE_CLASS_COUNT[0 : 2]) : sum(trainingConfigs.BRANCHE_CLASS_COUNT[0 : 3])]  = [1] * trainingConfigs.BRANCHE_CLASS_COUNT[2]
  mask_f[sum(trainingConfigs.BRANCHE_CLASS_COUNT[0 : 3]) : sum(trainingConfigs.BRANCHE_CLASS_COUNT[0 : 4])]  = [1] * trainingConfigs.BRANCHE_CLASS_COUNT[3]

  mask_l = np.array([mask_l,] * int(trainingConfigs.BATCH_SIZE / 4))
  mask_r = np.array([mask_r,] * int(trainingConfigs.BATCH_SIZE / 4))
  mask_s = np.array([mask_s,] * int(trainingConfigs.BATCH_SIZE / 4))
  mask_f = np.array([mask_f,] * int(trainingConfigs.BATCH_SIZE / 4))

  mask = np.vstack((mask_l, mask_r, mask_s, mask_f))

  y_mask = y_pred * mask


  #mask_r = [0] * sum(trainingConfigs.BRANCHE_CLASS_COUNT)
  #mask_f = [0] * sum(trainingConfigs.BRANCHE_CLASS_COUNT)

  #mask_r[0 : sum(trainingConfigs.BRANCHE_CLASS_COUNT[0 : 1])]   = [1] * trainingConfigs.BRANCHE_CLASS_COUNT[0]
  #mask_f[sum(trainingConfigs.BRANCHE_CLASS_COUNT[0 : 1]) : sum(trainingConfigs.BRANCHE_CLASS_COUNT[0 : 2])]  = [1] * trainingConfigs.BRANCHE_CLASS_COUNT[1]

  #mask_r = np.array([mask_r,] * int(trainingConfigs.BATCH_SIZE / 2))
  #mask_f = np.array([mask_f,] * int(trainingConfigs.BATCH_SIZE / 2))

  #mask = np.vstack((mask_r, mask_f))

  #y_mask = y_pred * mask

  #return cce(y_true, y_pred)
  return cce(y_true, y_mask)
 

def GatedLinearUnit(units, input):

  linear = Dense(units * 2)(input)
  linear = layers.LeakyReLU(0.3)(linear)
  linear = Dropout(0.5)(linear)

  linear = Dense(units)(linear)
  linear = layers.LeakyReLU(0.3)(linear)

  #gate = Dense(units, activation="sigmoid")(input)
  gate = Dense(units, activation="tanh")(input)

  return linear * gate
    

def GatedUnit(input, units):
  gate = Dense(256)(input)
  gate = layers.LeakyReLU(0.3)(gate)
  gate = Dropout(0.5)(gate)

  gate = Dense(256)(gate)
  gate = layers.LeakyReLU(0.3)(gate)
  gate = Dropout(0.5)(gate)

  gate = Dense(units, activation="tanh")(gate)
  return gate


def camera_stream(img):
    
  cnn = layers.Conv2D(32, (5, 5), (2,2), activation='relu', input_shape=(88, 200, 3), 
          kernel_initializer = tf.keras.initializers.GlorotNormal(), bias_initializer = tf.keras.initializers.Constant(0.1))(img)
  cnn = BatchNormalization()(cnn)

  cnn = layers.Conv2D(32, (3, 3), (1,1), activation='relu', 
          kernel_initializer = tf.keras.initializers.GlorotNormal(), bias_initializer = tf.keras.initializers.Constant(0.1))(cnn)
  cnn = BatchNormalization()(cnn)

  cnn = layers.Conv2D(64, (3, 3), (2,2), activation='relu',
            kernel_initializer = tf.keras.initializers.GlorotNormal(), bias_initializer = tf.keras.initializers.Constant(0.1) )(cnn)
  cnn = BatchNormalization()(cnn)

  cnn = layers.Conv2D(64, (3, 3), (1,1), activation='relu',
              kernel_initializer = tf.keras.initializers.GlorotNormal(), bias_initializer = tf.keras.initializers.Constant(0.1))(cnn)
  cnn = BatchNormalization()(cnn)

  cnn = layers.Conv2D(128, (3, 3), (2,2), activation='relu',
              kernel_initializer = tf.keras.initializers.GlorotNormal(), bias_initializer = tf.keras.initializers.Constant(0.1))(cnn)
  cnn = BatchNormalization()(cnn)

  cnn = layers.Conv2D(128, (3, 3), (1,1), activation='relu',
              kernel_initializer = tf.keras.initializers.GlorotNormal(), bias_initializer = tf.keras.initializers.Constant(0.1))(cnn)
  cnn = BatchNormalization()(cnn)


  cnn = layers.Conv2D(256, (3, 3), (1,1), activation='relu',
              kernel_initializer = tf.keras.initializers.GlorotNormal(), bias_initializer = tf.keras.initializers.Constant(0.1) )(cnn)
  cnn = BatchNormalization()(cnn)

  cnn = layers.Conv2D(256, (3, 3), (1,1), activation='relu',
              kernel_initializer = tf.keras.initializers.GlorotNormal(), bias_initializer = tf.keras.initializers.Constant(0.1))(cnn)
  cnn = BatchNormalization()(cnn)
    
  cnn = Flatten()(cnn)

  cnn = Dense(512)(cnn)
  cnn = layers.LeakyReLU(0.3)(cnn)
  cnn = Dropout(0.3)(cnn)
  cnn = Dense(512)(cnn)
  cnn = layers.LeakyReLU(0.3)(cnn)
  cnn = Dropout(0.3)(cnn)

  return cnn


def measurement_stream(meas):

  meas = Dense(128)(meas)
  meas = layers.LeakyReLU(0.3)(meas)
  meas = Dropout(0.5)(meas)

  meas = Dense(128)(meas)
  meas = layers.LeakyReLU(0.3)(meas)
  meas = Dropout(0.5)(meas)

  return meas


def get_model(isClassification = False, isCooccurance = False, isInterConnectBranches = False):

  branches = []
  interconnect = []
  br = []
  outputs = []

  front_input_img = Input(shape=(88, 200, 3))
  left_input_img = Input(shape=(88, 200, 3))
  right_input_img = Input(shape=(88, 200, 3))
  meas_yaw = Input(shape=(1))

  cnn1 = camera_stream(front_input_img)
  cnn2 = camera_stream(left_input_img)
  cnn3 = camera_stream(right_input_img)
  fc1  = measurement_stream(meas_yaw)

  cnn = tf.concat([cnn1, cnn2, cnn3, fc1], 1)

  cnn = Dense(512)(cnn)
  cnn = layers.LeakyReLU(0.3)(cnn)
  cnn = Dropout(0.5)(cnn)


  for i in range(len(trainingConfigs.BRANCHES)):

    branch = Dense(256)(cnn)
    branch = layers.LeakyReLU(0.3)(branch)
    branch = Dropout(0.5)(branch)

    branch = Dense(256)(branch)
    branch = layers.LeakyReLU(0.3)(branch)
    branch = Dropout(0.5)(branch)

    if isClassification:
      branch = Dense(trainingConfigs.BRANCHE_CLASS_COUNT[i], activation  = "softmax")(branch)
      branches.append(branch)
    else:
      if isInterConnectBranches:
        br.append(Dense(1)(branch))
        interconnect.append(GatedUnit(branch, trainingConfigs.INTERCONNECT_SIZE[i]))
      else:
        branches.append(Dense(1)(branch))

  if isInterConnectBranches:

    #outputs.append(Add()([br[0] , br[1] * interconnect[0]]))
    outputs.append(br[0])

    outputs.append(Add()([br[1] , br[0] * interconnect[1]]))
    #outputs.append(br[1])
    
    outputs.append(br[2])

    #outputs.append(br[3])
    outputs.append(Add()([br[3], tf.reduce_sum(Concatenate()([br[0], br[1]]) * interconnect[3], axis = 1, keepdims = True)]))   
    
    combined_outputs = Concatenate()([outputs[0], outputs[1], outputs[2], outputs[3]])
    
  else:
    combined_outputs = Concatenate()([branches[0], branches[1], branches[2], branches[3]])
    #combined_outputs = Concatenate()([branches[0], branches[1]])
  
  
  model = Model(inputs=[front_input_img, left_input_img, right_input_img, meas_yaw], outputs= [combined_outputs])
  opt  = Adam(learning_rate = trainingConfigs.LEARNING_RATE)
  
  if isCooccurance:
     model.compile(optimizer = opt, loss = cil_ising_loss, metrics=['accuracy'])
  elif isClassification:
    model.compile(optimizer = opt, loss = cil_class_loss, metrics=['accuracy'])
    #model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics=['accuracy'])
  else:
    model.compile(optimizer = opt, loss = cil_loss)
  
  return model
