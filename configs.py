import carla

DATASET_PATH = '/home/kishky/master/carla-research/dataset/'
#DATASET_PATH = '/media/kishky/My Passport/Master/carla_research/dataset/'

#BRANCHES_DATASET_PATH = '/media/kishky/My Passport/Master/carla_research/dataset_branches/'
BRANCHES_DATASET_PATH = '/home/kishky/master/carla-research/dataset_branches/'

EPISODE_SIMULATION_DURATION_SEC = 600    # 10 minutes long
EPISODE_START                   = 100
EPISODE_COUNT                   = 150    # number of episodes to be simulated

TRAFFIC_DENISTY_MIN             = 30
TRAFFIC_DENISTY_MAX             = 60

PEDESTRIAN_DESNISTY_MIN         = 50
PEDESTRIAN_DESNISTY_MAX         = 80

IMAGE_X                         = 800
IMAGE_Y                         = 600
IMAGE_CUT                       = [100, 500]

IMAGE_RESIZE_X                  = 200
IMAGE_RESIZE_Y                  = 88

LIDAR_CHANNELS                  = 32
LIDAR_RANGE                     = 15
LIDAR_FOV_U                     = 10
LIDAR_FOV_L                     = -30
LIDAR_POINTS_PER_SEC            = 150000
LIDAR_ROTATION_FREQ             = 10

SENSOR_TICK                     = 0.1   # 10 FPS
SIM_TICK                        = 0.05

TRAFFIC_LIGHT_GREEN_TIME        = 15.0
TRAFFIC_LIGHT_RED_TIME          = 5.0

VEHICLE_MAX_STEER = 70.0
VEHICLE_WHEEL_BASE = 2.89

# Steering wheel noise
APLLY_STEERING_NOISE_EVERY_SEC = 1
APPLY_NOISE_PROBABILITY = 0.5
APPLY_NOISE_MIN_T = 0.2
APPLY_NOISE_MAX_T = 0.4
NOISE_INTENSITY_MIN = 0.05
NOISE_INTENSITY_MAX = 0.1


PREDEFINED_WEATHER_LIST = [
carla.WeatherParameters.ClearNoon,
#carla.WeatherParameters.CloudyNoon,
#carla.WeatherParameters.WetNoon,
#carla.WeatherParameters.WetCloudyNoon,
#carla.WeatherParameters.MidRainyNoon,
#carla.WeatherParameters.HardRainNoon,
#carla.WeatherParameters.SoftRainNoon,
carla.WeatherParameters.ClearSunset,
#carla.WeatherParameters.CloudySunset,
#carla.WeatherParameters.WetSunset,
#carla.WeatherParameters.WetCloudySunset,
#carla.WeatherParameters.MidRainSunset,
#carla.WeatherParameters.HardRainSunset,
#carla.WeatherParameters.SoftRainSunset
]

SIMULATION_TOWN_LIST = [
'Town01',
#'Town02',
#'Town03',
#'Town04',
#'Town05',
#'Town06',
#'Town07'
]
