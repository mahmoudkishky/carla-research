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

# To import a basic agent
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption
from agents.tools.misc import get_speed


os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Don't use the GPU @ carla client side while testing


class testModel:

    def modelTestReadCamData(self, data, sensor):
        im = np.array(data.raw_data, dtype = np.dtype("uint8"))
        im = np.reshape(im, (configs.IMAGE_Y, configs.IMAGE_X, 4))
        im = im[:, :, :3]
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


class CustomAgent(BasicAgent):
    def __init__(self, vehicle):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param target_speed: speed (in Km/h) at which the vehicle will move
        """
        self.prev_steer = 0

        super().__init__(vehicle)
    
    def run_step(self, debug=False):
        
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        Actions to take during each simulation step
        """
        global testEnv
        global model
        global drift
        global mse
        global isClass

        # Legacy code of the basic agent
        hazard_detected = False

        # Retrieve all relevant actors
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        vehicle_speed = get_speed(self._vehicle) / 3.6

        # Check for possible vehicle obstacles
        max_vehicle_distance = self._base_vehicle_threshold + vehicle_speed
        affected_by_vehicle, _, _ = self._vehicle_obstacle_detected(vehicle_list, max_vehicle_distance)
        if affected_by_vehicle:
            hazard_detected = True

        # Check if the vehicle is affected by a red traffic light
        max_tlight_distance = self._base_tlight_threshold + vehicle_speed
        affected_by_tlight, _ = self._affected_by_traffic_light(lights_list, max_tlight_distance)
        if affected_by_tlight:
            hazard_detected = True

        control = self._local_planner.run_step()


        ################# insert our code #########################
        
        road_option = testEnv.collector.trafficManager.get_next_action(testEnv.collector.egoVehicle)[0]
        if(road_option == "Left"):
            branch = 0
            s = 0
            e = 33
            w = np.array(range(-23,10)) 
        elif(road_option == "Right"):
            branch = 1
            s = 33
            e = 72
            w = np.array(range(-8,31))
        elif(road_option == "Straight"):
            branch = 2
            s = 72
            e = 92
            w = np.array(range(-9,11))
        elif(road_option == "LaneFollow"):
            branch = 3
            s = 92
            e = 117
            w = np.array(range(-12,13))

        #############################################################


        predicted_steer = model.predict([np.reshape(testEnv.front_img, (1,88,200,3)), np.reshape(testEnv.left_img, (1,88,200,3)), np.reshape(testEnv.right_img, (1,88,200,3))], verbose = 0)

        basicAgentSteer = control.steer

        if(isClass):
            control.steer = np.sum(w * predicted_steer[0][s:e]) / configs.VEHICLE_MAX_STEER
        else:
            control.steer = predicted_steer[0][branch].tolist()


        drift += (self.prev_steer - control.steer)**2
        mse += (control.steer - basicAgentSteer)**2
        
        self.prev_steer = control.steer

        if hazard_detected:
            control = self.add_emergency_stop(control)
        
        return control
    

def main():

    
    global testEnv
    global model
    global totol_distance
    global drift
    global mse
    global isClass
    global curveRadius

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
                success = testEnv.collector.setup_episode_settings(town, 0, 0, weather, True, path[0])

                with open('./outputs/steering_model.json', 'r') as jfile:
                    model = model_from_json(json.load(jfile))

                model.load_weights("./outputs/steering_model.keras")

                if(success):

                    print('path :', path)

                    agent = CustomAgent(testEnv.collector.egoVehicle)
                    spawnpoints = testEnv.collector.world.get_map().get_spawn_points()
                    spectator = testEnv.collector.world.get_spectator()
                    agent.set_destination(spawnpoints[path[-1]].location)

                    #path_tm = []
                    #for ind in path:
                        #path_tm.append(spawnpoints[ind].location)


                    route = testingConfigs.TEST_ROUTE[testingConfigs.TEST_TOWNS.index(town)][testEnv.collector.currentEpisodeIndex]
                    testEnv.collector.trafficManager.set_route(testEnv.collector.egoVehicle, route)


                    totol_distance = 0
                    drift = 0
                    mse = 0
                    isClass = args.Class

                    while True:

                        testEnv.collector.world.tick()
                        #testEnv.collector.trafficManager.set_path(testEnv.collector.egoVehicle, path_tm)
                        spectator.set_transform(testEnv.collector.frontCamera.get_transform())
                        velosity = testEnv.collector.egoVehicle.get_velocity()
                        speed = np.sqrt((velosity.x**2) + (velosity.y**2))
                        totol_distance +=  speed * configs.SIM_TICK
                        testEnv.collector.currentSimulationStep += 1

                        if agent.done() or testEnv.collector.collision or testEnv.collector.currentSimulationStep * configs.SIM_TICK >= 180.0:
                        
                            csv_writer.writerow([town, testingConfigs.TEST_WEATHER_LIST.index(weather), path[0], path[1], str(agent.done()), str(testEnv.collector.collision), totol_distance, np.sqrt(drift / (testEnv.collector.currentSimulationStep)), mse / (testEnv.collector.currentSimulationStep)])
                            csv_file.flush()
                            testEnv.collector.cleanup()
                            print("##########################################################")

                            break

                        testEnv.collector.egoVehicle.apply_control(agent.run_step())
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Steering angle model Testing')
    parser.add_argument('--Class', dest='Class', action='store_true', default=False, help='is classification model')
    args = parser.parse_args()
    main()



