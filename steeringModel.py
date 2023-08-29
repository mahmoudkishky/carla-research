import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Dropout
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Concatenate, Add
import numpy as np
import scipy.stats as stats
import trainingConfigs
import sys


def cooccurance_matrix(num_class = 117, sigma = 1):
    
    s = np.zeros((num_class,num_class), dtype = 'float32')
    
    for branch in trainingConfigs.BRANCHES:
        
        class_idx = sum(trainingConfigs.BRANCHE_CLASS_COUNT[0 : trainingConfigs.BRANCHES.index(branch)])
        
        for i in range(class_idx, class_idx + trainingConfigs.BRANCHE_CLASS_COUNT[trainingConfigs.BRANCHES.index(branch)]):
            
            x = np.linspace(0, trainingConfigs.BRANCHE_CLASS_COUNT[trainingConfigs.BRANCHES.index(branch)] - 1, num = trainingConfigs.BRANCHE_CLASS_COUNT[trainingConfigs.BRANCHES.index(branch)])
            y = stats.norm.pdf(x, i - class_idx, sigma)
            s[i, class_idx : (class_idx + trainingConfigs.BRANCHE_CLASS_COUNT[trainingConfigs.BRANCHES.index(branch)])] = y
            s[i,i] = 0

    return s

def cil_ising_loss(y_true, y_pred):
   
  s = cooccurance_matrix(num_class = sum(trainingConfigs.BRANCHE_CLASS_COUNT), sigma = 3)
  cce = tf.keras.losses.CategoricalCrossentropy()

  mask_l = [0] * sum(trainingConfigs.BRANCHE_CLASS_COUNT)
  mask_r = [0] * sum(trainingConfigs.BRANCHE_CLASS_COUNT)
  mask_s = [0] * sum(trainingConfigs.BRANCHE_CLASS_COUNT)
  mask_f = [0] * sum(trainingConfigs.BRANCHE_CLASS_COUNT)

  mask_l[0:33]   = [1] * trainingConfigs.BRANCHE_CLASS_COUNT[0]
  mask_r[33:72]  = [1] * trainingConfigs.BRANCHE_CLASS_COUNT[1]
  mask_s[72:92]  = [1] * trainingConfigs.BRANCHE_CLASS_COUNT[2]
  mask_f[92:117] = [1] * trainingConfigs.BRANCHE_CLASS_COUNT[3]

  mask_l = np.array([mask_l,] * int(trainingConfigs.BATCH_SIZE / 4))
  mask_r = np.array([mask_r,] * int(trainingConfigs.BATCH_SIZE / 4))
  mask_s = np.array([mask_s,] * int(trainingConfigs.BATCH_SIZE / 4))
  mask_f = np.array([mask_f,] * int(trainingConfigs.BATCH_SIZE / 4))

  mask = np.vstack((mask_l, mask_r, mask_s, mask_f))

  y_mask = y_pred * mask

  l1 =  (1 - trainingConfigs.COOCCURANCE_WEIGHT) * cce(y_true, y_mask)
  #l2 =  -1 * (trainingConfigs.COOCCURANCE_WEIGHT) * tf.reduce_mean(tf.linalg.diag_part(tf.linalg.matmul(tf.linalg.matmul(y_pred, s), tf.transpose(y_pred))))
  l2 =  -1 * (trainingConfigs.COOCCURANCE_WEIGHT) * tf.reduce_mean(tf.linalg.matmul(tf.linalg.matmul(y_pred, s), tf.transpose(y_pred)))
  
  return tf.math.add(l1, l2)

def cil_loss(y_true, y_pred):

  batch_nav_cmds = np.array(int(trainingConfigs.BATCH_SIZE / 4 ) * [trainingConfigs.BRANCHES.index('L')] + 
                 int(trainingConfigs.BATCH_SIZE / 4 ) * [trainingConfigs.BRANCHES.index('R')] +
                 int(trainingConfigs.BATCH_SIZE / 4 ) * [trainingConfigs.BRANCHES.index('S')] + 
                 int(trainingConfigs.BATCH_SIZE / 4 ) * [trainingConfigs.BRANCHES.index('F')])

  enc_nav_cmds = to_categorical(batch_nav_cmds, num_classes = 4)
  pred = tf.reduce_sum(tf.multiply(enc_nav_cmds, y_pred), axis = 1)
  loss = tf.reduce_mean(tf.square(tf.subtract(pred, y_true)))

  return loss


def cil_class_loss(y_true, y_pred):

  cce = tf.keras.losses.CategoricalCrossentropy()

  mask_l = [0] * sum(trainingConfigs.BRANCHE_CLASS_COUNT)
  mask_r = [0] * sum(trainingConfigs.BRANCHE_CLASS_COUNT)
  mask_s = [0] * sum(trainingConfigs.BRANCHE_CLASS_COUNT)
  mask_f = [0] * sum(trainingConfigs.BRANCHE_CLASS_COUNT)

  mask_l[0:33]   = [1] * trainingConfigs.BRANCHE_CLASS_COUNT[0]
  mask_r[33:72]  = [1] * trainingConfigs.BRANCHE_CLASS_COUNT[1]
  mask_s[72:92]  = [1] * trainingConfigs.BRANCHE_CLASS_COUNT[2]
  mask_f[92:117] = [1] * trainingConfigs.BRANCHE_CLASS_COUNT[3]

  mask_l = np.array([mask_l,] * int(trainingConfigs.BATCH_SIZE / 4))
  mask_r = np.array([mask_r,] * int(trainingConfigs.BATCH_SIZE / 4))
  mask_s = np.array([mask_s,] * int(trainingConfigs.BATCH_SIZE / 4))
  mask_f = np.array([mask_f,] * int(trainingConfigs.BATCH_SIZE / 4))

  mask = np.vstack((mask_l, mask_r, mask_s, mask_f))

  y_mask = y_pred * mask

  return cce(y_true, y_mask)
 

def GatedLinearUnit(units, input):

  linear = Dense(units * 2)(input)
  linear = Dropout(0.5)(linear)
  linear = Dense(units)(linear)

  sigmoid = Dense(units, activation="sigmoid")(input)

  return linear * sigmoid
    


def camera_stream(img):
    
  cnn = layers.Conv2D(32, (5, 5), (2,2), activation='relu', input_shape=(32, 32, 3))(img)
  cnn = BatchNormalization()(cnn)

  cnn = layers.Conv2D(32, (3, 3), (1,1), activation='relu')(cnn)
  cnn = BatchNormalization()(cnn)

  cnn = layers.Conv2D(64, (3, 3), (2,2), activation='relu', input_shape=(32, 32, 3))(cnn)
  cnn = BatchNormalization()(cnn)

  cnn = layers.Conv2D(64, (3, 3), (1,1), activation='relu')(cnn)
  cnn = BatchNormalization()(cnn)

  cnn = layers.Conv2D(128, (3, 3), (2,2), activation='relu', input_shape=(32, 32, 3))(cnn)
  cnn = BatchNormalization()(cnn)

  cnn = layers.Conv2D(128, (3, 3), (1,1), activation='relu')(cnn)
  cnn = BatchNormalization()(cnn)


  cnn = layers.Conv2D(256, (3, 3), (1,1), activation='relu', input_shape=(32, 32, 3))(cnn)
  cnn = BatchNormalization()(cnn)

  cnn = layers.Conv2D(256, (3, 3), (1,1), activation='relu')(cnn)
  cnn = BatchNormalization()(cnn)
    
  cnn = Flatten()(cnn)

  return cnn

def get_model(isClassification = False, isCooccurance = False, isInterConnectBranches = False):

  branches = []
  sub_branches = []
  br = []

  front_input_img = Input(shape=(88, 200, 3))
  left_input_img = Input(shape=(88, 200, 3))
  right_input_img = Input(shape=(88, 200, 3))

  cnn1 = camera_stream(front_input_img)
  cnn2 = camera_stream(left_input_img)
  cnn3 = camera_stream(right_input_img)

  cnn = tf.concat([cnn1, cnn2, cnn3], 1)

  if isInterConnectBranches:

    for i in range(4):

      branch = Dense(512, activation = "relu")(cnn)
      branch = Dropout(0.5)(branch)
      branch = Dense(512, activation = "relu")(branch)

      sub_branch = GatedLinearUnit(512, branch)
      sub_branches.append(sub_branch)
      branches.append(branch)
      


    br.append(Add()([branches[0], sub_branches[1], sub_branches[2], sub_branches[3]]))
    br.append(Add()([branches[1], sub_branches[0], sub_branches[2], sub_branches[3]]))
    br.append(Add()([branches[2], sub_branches[0], sub_branches[1], sub_branches[3]]))
    br.append(Add()([branches[3], sub_branches[0], sub_branches[1], sub_branches[2]]))

    for i in range(4):

      branches[i] = Dropout(0.5)(br[i])

      if isClassification:
        branches[i] = Dense(trainingConfigs.BRANCHE_CLASS_COUNT[i], activation  = "softmax")(branches[i])
      else:
        branches[i] = Dense(1)(branches[i])

      branches.append(branches[i])

  else:

    for i in range(4):
      branch = Dense(512, activation = "relu")(cnn)
      branch = Dropout(0.5)(branch)
      branch = Dense(512, activation = "relu")(branch)
      branch = Dropout(0.5)(branch)

      if isClassification:
        branch = Dense(trainingConfigs.BRANCHE_CLASS_COUNT[i], activation  = "softmax")(branch)
      else:
        branch = Dense(1)(branch)

      branches.append(branch)

  combined_outputs = Concatenate()([branches[0], branches[1], branches[2], branches[3]])

  model = Model(inputs=[front_input_img, left_input_img, right_input_img], outputs= [combined_outputs])
  #opt  = Adam(learning_rate = trainingConfigs.LEARNING_RATE, beta_1 = trainingConfigs.BETA1, beta_2 = trainingConfigs.BETA2)
  opt  = Adam(learning_rate = trainingConfigs.LEARNING_RATE)
  
  if isCooccurance:
     model.compile(optimizer = opt, loss = cil_ising_loss, metrics=['accuracy'])
  elif isClassification:
    model.compile(optimizer = opt, loss = cil_class_loss, metrics=['accuracy'])
  else:
    model.compile(optimizer = opt, loss = cil_loss)
  
  return model
