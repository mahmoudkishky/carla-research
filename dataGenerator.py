import numpy as np
import configs
import trainingConfigs
import cv2
from keras.utils import Sequence
from imgaug import augmenters as iaa
from tensorflow import one_hot
import matplotlib.pyplot as plt


# Use imgaug lib to create sequence of augmentors with diffierent probabilities 
def create_augmenter():

    st = lambda aug: iaa.Sometimes(0.4, aug)    # 40% of images to be augmented
    oc = lambda aug: iaa.Sometimes(0.3, aug)    # 30% of images to be augmented
    rl = lambda aug: iaa.Sometimes(0.1, aug)    # 10% of images to be augmented


    seq = iaa.Sequential([
        rl(iaa.GaussianBlur((0, 1.5))),                                            # blur images with a sigma between 0 and 1.5
        rl(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5)),  # add gaussian noise to images
        oc(iaa.Dropout((0.0, 0.10), per_channel=0.5)),                             # randomly remove up to X% of the pixels
        oc(iaa.CoarseDropout((0.0, 0.10), size_percent=(0.08, 0.2), per_channel=0.5)),

        # randomly remove up to X% of the pixels
        oc(iaa.Add((-40, 40), per_channel=0.5)),                                   # change brightness of images (by -X to Y of original value)
        st(iaa.Multiply((0.10, 2.5), per_channel=0.2)),                            # change brightness of images (X-Y% of original value)
        rl(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)),                # improve or worsen the contrast
    ], random_order=True)                                                          # order is shuffled each time with equal probability
    return seq


def assign_steering_label(steering_angle, branch):

    angle_norm = steering_angle - trainingConfigs.BRANCHE_MIN_ANGLE[trainingConfigs.BRANCHES.index(branch)]
    label = int(round(10 * angle_norm))

    return label

class DataGenerator(Sequence):

    'Generates data for Keras'
    def __init__(self, data, dataIds, batch_size = 32, isAugmentation = False ,isClassification = False):

        'Initialization'
        self.dim = (88, 200, 3)
        self.batch_size = batch_size
        self.dataIds = dataIds
        self.data = data
        self.on_epoch_end()
        self.augmenter = create_augmenter()
        self.isAugmentation = isAugmentation
        self.isClassification = isClassification

    def __len__(self):
        return int(np.floor(min(len(val) for val in self.dataIds.values()) / (self.batch_size / len(trainingConfigs.BRANCHES))))

    def __getitem__(self, index):

        'Generate one batch of data'
        batchIds = {key: None for key in trainingConfigs.BRANCHES}

        for branch in trainingConfigs.BRANCHES:
            batchIds[branch] = [self.dataIds[branch][k] for k in range(index * int(self.batch_size / len(trainingConfigs.BRANCHES)), (index + 1) * int(self.batch_size / len(trainingConfigs.BRANCHES)))]

        # Generate data-
        X, Y = self.__data_generation(batchIds)
        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        for branch in trainingConfigs.BRANCHES:
            np.random.shuffle(self.dataIds[branch])


    def __data_generation(self, tempDataIds):
        'Generates data containing batch_size samples'

        # Initialization
        X1 = np.empty((self.batch_size, *self.dim), dtype = np.dtype("uint8"))
        X2 = np.empty((self.batch_size, *self.dim), dtype = np.dtype("uint8"))
        X3 = np.empty((self.batch_size, *self.dim), dtype = np.dtype("uint8"))
        YAW = np.empty((self.batch_size), dtype = np.dtype("uint8"))
        
        if self.isClassification:
            Y = np.zeros((self.batch_size, sum(trainingConfigs.BRANCHE_CLASS_COUNT)), dtype = np.dtype("float32"))
        else:
            Y = np.empty((self.batch_size), dtype = np.dtype("float32"))
        
        # Generate data
        for branch in trainingConfigs.BRANCHES:

            branch_idx = trainingConfigs.BRANCHES.index(branch) * int(self.batch_size / len(trainingConfigs.BRANCHES))
            class_idx = sum(trainingConfigs.BRANCHE_CLASS_COUNT[0 : trainingConfigs.BRANCHES.index(branch)])
            
            for i in range(0, int(self.batch_size / len(trainingConfigs.BRANCHES))):
                
                X1[branch_idx + i,] = np.array(cv2.imread(configs.BRANCHES_DATASET_PATH + branch + '/FrontCam/' + str(tempDataIds[branch][i]) + '.png'))
                X2[branch_idx + i,] = np.array(cv2.imread(configs.BRANCHES_DATASET_PATH + branch + '/LeftCam/' + str(tempDataIds[branch][i]) + '.png'))
                X3[branch_idx + i,] = np.array(cv2.imread(configs.BRANCHES_DATASET_PATH + branch + '/RightCam/' + str(tempDataIds[branch][i]) + '.png'))
                YAW[branch_idx + i,] = self.data[branch].loc[tempDataIds[branch][i]]['yaw']

                if self.isClassification:

                    label = int(round(((self.data[branch].loc[tempDataIds[branch][i]]['steering_angle'] - trainingConfigs.BRANCHE_MIN_ANGLE[trainingConfigs.BRANCHES.index(branch)]) / trainingConfigs.STEER_CLASS_STEP[trainingConfigs.BRANCHES.index(branch)])))
                    Y[branch_idx + i].__setitem__(slice(class_idx, class_idx + trainingConfigs.BRANCHE_CLASS_COUNT[trainingConfigs.BRANCHES.index(branch)]), one_hot(label, depth = trainingConfigs.BRANCHE_CLASS_COUNT[trainingConfigs.BRANCHES.index(branch)]))

                else:
                    Y[branch_idx + i] = self.data[branch].loc[tempDataIds[branch][i]]['steering_angle']
            
        
        if self.isAugmentation:
            aug1 = self.augmenter.augment_images(np.array([X1[i] for i in range(self.batch_size)], dtype = np.uint8))
            aug2 = self.augmenter.augment_images(np.array([X2[i] for i in range(self.batch_size)], dtype = np.uint8))
            aug3 = self.augmenter.augment_images(np.array([X3[i] for i in range(self.batch_size)], dtype = np.uint8))
        else:
            aug1 = X1
            aug2 = X2
            aug3 = X3

        """
        #print('X:')
        #print(aug[0])
        plt.imshow(cv2.cvtColor(aug1[0], cv2.COLOR_BGR2RGB))
        plt.show()
        plt.figure()
        plt.imshow(cv2.cvtColor(aug2[0], cv2.COLOR_BGR2RGB))
        plt.show()
        plt.figure()
        plt.imshow(cv2.cvtColor(aug3[0], cv2.COLOR_BGR2RGB))
        plt.show()
        print('Y:')
        print(Y[0])
        """
        

        return [aug1 / 255., aug2 / 255., aug3 / 255., YAW / 180.], Y
