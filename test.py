import sys
import numpy as np
import keras
import os
import SimpleITK as sitk
from tensorflow.keras import layers
from tensorflow.keras import Model
os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
base_dir = '/data/wdm_Projects/X-ray-images-classification-with-Keras-TensorFlow/data'
#base_dir = '/home/ente/Schreibtisch/2018 - 11 - sorted data'
#base_dir = '/Users/mkunzmann/Desktop/2018 - 11 - sorted data'
#class_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax' ]
class_names = ['CT','Probe']
img_width = img_height = 512
color_channels = 3
target_width = target_height = 512
target_size =(30,256,256) #(40,40,40)#(100, 306, 386)
pooling_window = 2
conv_window = 3
kernel_size = 16
dropout=0.01
activation = "relu"
optimizer = 'Adam'
loss='categorical_crossentropy'
#metrics= ['acc']
metrics=['categorical_accuracy']
class_mode = 'categorical'
batch_size = 10
epochs = 150
verbose = 2
lr = 0.001
#layers & optimizer

def threscut(subject_data, threshold_min=1000, threshold_max=2000):
    subject_data[subject_data > threshold_max] = threshold_max
    subject_data[subject_data < threshold_min] = threshold_min

    return subject_data


def normalize_3D(img_3D, nor_min=1000, nor_max=2000):
    """ The shape of img_3D should be (depth, width, height)"""
    data_3D = img_3D - nor_min
    data_3D = data_3D / np.float32(nor_max-nor_min)
    #data_3D = img_3D - img_3D.min()
    #data_3D = data_3D / np.float32(img_3D.max())
    return np.asarray(data_3D, np.float32)

def generate_single_batch_test_data(data_path):
    while 1:
        #for i in os.listdir(data_path):
        class_ind = np.random.randint(0,1)
        if class_ind == 0:
            data_path_1 = data_path + '/CT'
            label = np.array([0,0])
            
        else:
            data_path_1 = data_path + '/Probe'
            label = np.array([0,0])
        i = np.random.randint(len(os.listdir(data_path_1)))
        img = sitk.ReadImage(os.path.join(data_path_1, sorted(os.listdir(data_path_1))[i]))
        img_data = sitk.GetArrayFromImage(img)
        img_data = img_data[:, :, :]
        img_data = np.float32(threscut(img_data))
        img_data = normalize_3D(img_data)
        [depth, height, width] = img_data.shape
        #depth_start = randint(0, depth)
        #width_start = randint(0, width)
        img_data = img_data[0:depth, 0:height, 0:width]

        img_data = np.expand_dims(img_data, axis=0)
        img_data = np.expand_dims(img_data, axis=4)
        #gc.collect()
        label = np.expand_dims(label, axis=0)
        yield img_data, label
def model():
	# Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for the three color channels: R, G, and B
#img_input = layers.Input(shape=(img_width, img_height, color_channels))
	img_input = layers.Input(shape=target_size + (1,))

# First convolution extracts 16 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
	x = layers.Conv3D(kernel_size, conv_window, activation=activation)(img_input)
	x = layers.MaxPooling3D(pooling_window)(x)

#x = layers.Conv3D(kernel_size, conv_window, activation=activation, padding='same')(x)

# Second convolution extracts 32 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
	x = layers.Conv3D(2*kernel_size, conv_window, activation=activation)(x)
	x = layers.MaxPooling3D(pooling_window)(x)

#x = layers.Conv3D(2*kernel_size, conv_window, activation=activation, padding='same')(x)

# Third convolution extracts 64 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
#x = layers.Conv3D(4*kernel_size, conv_window, activation=activation, padding='same')(x)
	x = layers.Conv3D(4*kernel_size, conv_window, activation=activation, padding='same')(x)
	x = layers.Conv3D(4*kernel_size, conv_window, activation=activation)(x)
	x = layers.MaxPooling3D(pooling_window)(x)

	x = layers.Conv3D(6*kernel_size, conv_window, activation=activation, padding='same')(x)
	x = layers.MaxPooling3D(pooling_window)(x)

	x = layers.Conv3D(6*kernel_size, conv_window, activation=activation, padding='same')(x)
	x = layers.Flatten()(x)

# Create a fully connected layer with ReLU activation and 512 hidden units
	x = layers.Dense(512, activation=activation)(x)

# Add Droptout Regularization
#x = layers.Dropout(dropout)(x)

# Create output layer with a single node and sigmoid activation
#output = layers.Dense(1, activation='sigmoid')(x)
	output = layers.Dense(len(class_names), activation = 'softmax') (x)


# Create model: input = input feature map
# output = input feature map + stacked convolution/maxpooling layers + fully connected layer + sigmoid output layer
	model = Model(img_input, output)
	return model

if __name__=="__main__":
	
	test_generator = generate_single_batch_test_data(base_dir+'/testing')
	model = model()
	model.load_weights(sys.argv[1])
	probabilities = model.predict_generator(test_generator, 4)
	print(probabilities)

