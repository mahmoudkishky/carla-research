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

    for branch in trainingConfigs.BRANCHES:

        data[branch] = read_csv(configs.BRANCHES_DATASET_PATH + branch + '/control.csv')

        if balance:

            # remove majority samples to ensure we have balanced learning
            steer = data[branch]['steering_angle']
            data[branch]['int_steer'] = np.array(configs.VEHICLE_MAX_STEER * steer, dtype = 'float32').astype(int)

            q_low = data[branch]['int_steer'].quantile(0.025)
            q_hi  = data[branch]['int_steer'].quantile(0.975)

            dropped = data[branch][(data[branch]['int_steer'] < q_low) | (data[branch]['int_steer'] > q_hi)].sample(frac = 1.0, replace=False)
            data[branch] = data[branch].drop(list(dropped.index), axis = 0)

            dropped = data[branch][(data[branch]['throttle'] <= 0.05) & (np.abs(data[branch]['int_steer']) <= 2)].sample(frac = 1.0, replace=False)
            data[branch] = data[branch].drop(list(dropped.index), axis = 0)
            

        dataIds = np.array(data[branch]['Id'])
        np.random.shuffle(dataIds)
        traingIds[branch], validationIds[branch] = train_test_split(dataIds, test_size = 0.1)

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

    #callback = tf.keras.callbacks.LearningRateScheduler(AdamoptimizersScheduler)

    # Train model on dataset
    #model.fit(training_generator, epochs = trainingConfigs.EPOCHS, validation_data = validation_generator, callbacks = [callback])
    model.fit(training_generator, epochs = trainingConfigs.EPOCHS, validation_data = validation_generator)

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
