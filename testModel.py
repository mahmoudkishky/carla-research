from dataCollector import dataCollector
import configs
import numpy as np
import random
import json
import steeringModel
import cv2
import tensorflow as tf
from keras.models import model_from_json
import os
import testingConfigs
import csv
import argparse
from steeringModel import GatedLinearUnit
import trainingConfigs

# To import a basic agent
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption
from agents.tools.misc import get_speed


os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Don't use the GPU @ carla client side while testing

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class testModel:

    def modelTestReadCamData(self, data, sensor):
        im = np.array(data.raw_data, dtype = np.dtype("uint8"))
        im = np.reshape(im, (configs.IMAGE_Y, configs.IMAGE_X, 4))
        im = im[:, :, :3]
        im = im[configs.IMAGE_CUT[0]: configs.IMAGE_CUT[1], :, :3]
        im = cv2.resize(im, dsize = (configs.IMAGE_RESIZE_X, configs.IMAGE_RESIZE_Y), interpolation = cv2.INTER_CUBIC)
        
        if(sensor == 'FrontCam'):
            self.front_img = im / 255.
        elif(sensor == 'LeftCam'):
            self.left_img = im / 255.
        elif(sensor == 'RightCam'):
            self.right_img = im / 255.


    def modelTestCollisionHandler(self, event):

        if(self.collector.collision == False):
            print('WARN: collion occured during episode')
            self.collector.collision = True

    def __init__(self):
        self.collector = dataCollector()
        self.collector.cameraReadData = self.modelTestReadCamData
        self.collector.collisionHandler = self.modelTestCollisionHandler


    
def cil_step(steerAgent, roadOption, isClass):
        
    global testEnv
    global model
    global prev_steer
    global drift
    global mse

    

    if(roadOption == RoadOption.LEFT):

        branch = 0
        s = 0
        e = sum(trainingConfigs.BRANCHE_CLASS_COUNT[0 : 1])

    elif(roadOption == RoadOption.RIGHT):

        branch = 1
        s = sum(trainingConfigs.BRANCHE_CLASS_COUNT[0 : 1])
        e = sum(trainingConfigs.BRANCHE_CLASS_COUNT[0 : 2])

    elif(roadOption == RoadOption.STRAIGHT):

        branch = 2
        s = sum(trainingConfigs.BRANCHE_CLASS_COUNT[0 : 2])
        e = sum(trainingConfigs.BRANCHE_CLASS_COUNT[0 : 3])

    else:

        branch = 3
        s = sum(trainingConfigs.BRANCHE_CLASS_COUNT[0 : 3])
        e = sum(trainingConfigs.BRANCHE_CLASS_COUNT[0 : 4])
        
    #############################################################
            
    w = (np.array(range(trainingConfigs.BRANCHE_CLASS_COUNT[branch])) * trainingConfigs.STEER_CLASS_STEP[branch]) + trainingConfigs.BRANCHE_MIN_ANGLE[branch]
    predicted_steer = model.predict([np.reshape(testEnv.front_img, (1,88,200,3)), np.reshape(testEnv.left_img, (1,88,200,3)), np.reshape(testEnv.right_img, (1,88,200,3)), np.array([(testEnv.collector.egoVehicle.get_transform().rotation.yaw) / 180.])], verbose = 0)


    if(isClass):

        steer = (np.sum(w * predicted_steer[0][s:e]))
        #steer = w[np.argmax(predicted_steer[0][s:e])]
         
    else:
        steer = predicted_steer[0][branch].tolist()


    mse += (steer - steerAgent)**2
    drift += (prev_steer - steer)**2

    return steer
    

def main():

    
    global testEnv
    global model
    global totol_distance
    global prev_steer
    global drift
    global mse

    testEnv = testModel()

    csv_file = open('./results.csv', 'w+')
    csv_writer = csv.writer(csv_file, delimiter=',')
    csv_writer.writerow(['Town', 'Weather', 'Source', 'Destination','Reach Destination', 'Collision', 'Distance', 'Drift', 'MSE'])
    csv_file.flush()
    
    for town in testingConfigs.TEST_TOWNS:

        for weather in testingConfigs.TEST_WEATHER_LIST:

            testEnv.collector.currentEpisodeIndex = 0

            for path in testingConfigs.TEST_PATH[testingConfigs.TEST_TOWNS.index(town)]:

                traffic_denisty = np.random.randint(configs.TRAFFIC_DENISTY_MIN, configs.TRAFFIC_DENISTY_MAX)
                pedestrian_denisty = np.random.randint(configs.PEDESTRIAN_DESNISTY_MIN, configs.PEDESTRIAN_DESNISTY_MAX)
                #success = testEnv.collector.setup_episode_settings(town, traffic_denisty, pedestrian_denisty, weather, True, path[0])
                success = testEnv.collector.setup_episode_settings(town, 0, 0, weather, True, path[0][0])

                with open('./outputs/steering_model.json', 'r') as jfile:
                    model = model_from_json(json.load(jfile))

                model.load_weights("./outputs/steering_model.keras")

                if(success):

                    print('path :', path)

                    agent = BasicAgent(testEnv.collector.egoVehicle, 25)
                    #agent.follow_speed_limits(True)
                    spawnpoints = testEnv.collector.world.get_map().get_spawn_points()
                    spectator = testEnv.collector.world.get_spectator()
                    agent.set_destination(spawnpoints[path[0][-1]].location)


                    #route = testingConfigs.TEST_ROUTE[testingConfigs.TEST_TOWNS.index(town)][testEnv.collector.currentEpisodeIndex]
                    #testEnv.collector.trafficManager.set_route(testEnv.collector.egoVehicle, route)

                    testEnv.collector.trafficManager.set_route(testEnv.collector.egoVehicle, path[1])


                    totol_distance = 0
                    drift = 0
                    mse = 0
                    prev_steer = 0

                    while True:

                        testEnv.collector.world.tick()
                        spectator.set_transform(testEnv.collector.frontCamera.get_transform())
                        velosity = testEnv.collector.egoVehicle.get_velocity()
                        speed = np.sqrt((velosity.x**2) + (velosity.y**2))
                        totol_distance +=  speed * configs.SIM_TICK
                        testEnv.collector.currentSimulationStep += 1

                        if agent.done() or testEnv.collector.collision or testEnv.collector.currentSimulationStep * configs.SIM_TICK >= 180.0:
                        
                            csv_writer.writerow([town, testingConfigs.TEST_WEATHER_LIST.index(weather), path[0][0], path[0][-1], str(agent.done()), str(testEnv.collector.collision), totol_distance, np.sqrt(drift / (testEnv.collector.currentSimulationStep)), mse / (testEnv.collector.currentSimulationStep)])
                            csv_file.flush()
                            testEnv.collector.cleanup()
                            print("##########################################################")

                            break

                        control = agent.run_step()
                        direction = agent.get_local_planner().get_incoming_waypoint_and_direction(steps = 0)[1]

                        #print('true: ' , control.steer)
                        control.steer = cil_step(control.steer, direction, args.Class)
                        #print('predicted : ' , control.steer)
                        testEnv.collector.egoVehicle.apply_control(control)
                        prev_steer = control.steer
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Steering angle model Testing')
    parser.add_argument('--Class', dest='Class', action='store_true', default=False, help='is classification model')
    args = parser.parse_args()
    main()



