import numpy as np
import tensorflow as tf
from steeringModel import *
from dataGenerator import DataGenerator
from pandas import *
from sklearn.model_selection import train_test_split
import argparse
import os
import configs
import trainingConfigs
import json
import matplotlib.pyplot as plt

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def loadDataset(balance):

    data = {key: None for key in trainingConfigs.BRANCHES}
    traingIds = {key: None for key in trainingConfigs.BRANCHES}
    validationIds = {key: None for key in trainingConfigs.BRANCHES}

    q_low = -0.8
    q_hi = 0.8
    
    for branch in trainingConfigs.BRANCHES:

        data[branch] = read_csv(configs.BRANCHES_DATASET_PATH + branch + '/control.csv')
        
        dropped = data[branch][(data[branch]['steering_angle'] < q_low) | (data[branch]['steering_angle'] > q_hi)].sample(frac = 1.0, replace=False)
        data[branch] = data[branch].drop(list(dropped.index), axis = 0)
        
        if balance:

            # remove majority samples to ensure we have balanced learning
            dropped = data[branch][np.abs(data[branch]['steering_angle']) <= 0.05].sample(frac = 0.80, replace=False)
            data[branch] = data[branch].drop(list(dropped.index), axis = 0)
            
        dataIds = np.array(data[branch]['Id'])
        np.random.shuffle(dataIds)
        traingIds[branch], validationIds[branch] = train_test_split(dataIds, test_size = 0.15)

    return data, traingIds, validationIds

def AdamoptimizersScheduler(epoch, lr):
   if epoch % 5 == 0 and epoch != 0:
     return lr * 0.5
   else:
     return lr
   

def main():

    data, traingIds, validationIds = loadDataset(args.balance)

    # Generators
    training_generator = DataGenerator(data, traingIds, trainingConfigs.BATCH_SIZE, True, args.Class)
    validation_generator = DataGenerator(data, validationIds, trainingConfigs.BATCH_SIZE, False, args.Class)

    # Design model
    model = get_model(args.Class, args.co, args.connect)

    model.summary()

    with open("./outputs/steering_model.json", 'w') as outfile:
        json.dump(model.to_json(), outfile)

    if args.loadweights:
        print('loading weights ....')
        model.load_weights("./outputs/steering_model.keras",True)

    callback = tf.keras.callbacks.ModelCheckpoint("./outputs/cp-{epoch:04d}.keras", verbose = 1, save_weights_only = True, save_freq = 'epoch')
    
    # Train model on dataset
    model.fit(training_generator, epochs = trainingConfigs.EPOCHS, validation_data = validation_generator, callbacks = [callback])
    #model.fit(training_generator, epochs = trainingConfigs.EPOCHS, validation_data = validation_generator)

    model.save_weights("./outputs/steering_model.keras", True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Steering angle model trainer')
    parser.add_argument('--loadweights', dest='loadweights', action='store_true',default=False, help='Start from checkpoint')
    parser.add_argument('--balance', dest='balance', action='store_true', default=False, help='is balanced dataset')
    parser.add_argument('--Class', dest='Class', action='store_true', default=False, help='is classification model')
    parser.add_argument('--co', dest='co', action='store_true', default=False, help='is output layer statistic forced')
    parser.add_argument('--connect', dest='connect', action='store_true', default=False, help='branches share features')

    args = parser.parse_args()
    main()
