import os
from matplotlib import pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES']=str(1)
from tensorflow.keras import layers
from tensorflow.keras import Model
#from tensorflow.keras.layers.core import Lambda
from tensorflow.keras.layers import Lambda
#from tensorflow.keras.models import Sequential
from tensorflow.keras.backend import set_session
from tensorflow.python.framework import ops
#import keras.backend as K
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
# =============================================================================
# from tensorflow.keras import layers
# from tensorflow.keras import Model
# from tensorflow.keras.layers import Lambda
# import tensorflow.keras.backend as K
# =============================================================================

## hyper-parameter
target_size = (128, 128, 128)
kernel_size = 4
conv_window = 3
activation = "relu"
pooling_window = 2
dropout = 0.01
class_names = ['CT','Probe']

#optimizer = 'Adam'
def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return 100.*x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    #img_path = sys.argv[1]
    img_path = path
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer='block5_conv3'):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        new_model = VGG16(weights='imagenet')
    return new_model

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def grad_cam_3D(input_model, image, category_index, layer_name):
    '''
    model = Sequential()
    model.add(input_model)
    
    nb_classes = 2
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    model.add(Lambda(target_layer, output_shape = target_category_loss_output_shape))
    loss = K.sum(model.layers[-1].output)'''
    nb_classes = 2
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    x = Lambda(target_layer, output_shape = target_category_loss_output_shape)(input_model.output)
    model = Model(inputs=input_model.input, outputs=x)
#    model.summary()
    loss = K.sum(model.output)
    
    l = model.layers[layer_name]
    conv_output =  [l][0].output#[l for l in model.layers[0].layers if l.name is layer_name][0].output
    grads = normalize(_compute_gradients(loss, [conv_output])[0])
    gradient_function = K.function([model.layers[0].input, model.layers[1].input], [conv_output, grads])

    output, grads_val = gradient_function(image)
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1, 2))
    cam = np.ones(output.shape[0 : 3], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :,:, i]

    #cam = cv2.resize(cam, (256, 256))
    #cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    '''
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    '''
    return np.float32(cam), heatmap
#preprocessed_input = load_image(sys.argv[1])
    

def threscut(subject_data, threshold_min=0, threshold_max=1700):# 1000,2000
    subject_data[subject_data > threshold_max] = threshold_max
    subject_data[subject_data < threshold_min] = threshold_min

    return subject_data


def normalize_3D(img_3D, nor_min=0, nor_max=1700):
    """ The shape of img_3D should be (depth, width, height)"""
    data_3D = img_3D - nor_min
    if nor_max-nor_min==0:
        data_3D[data_3D==nor_max]=1
    else:
        data_3D = data_3D / np.float32(nor_max-nor_min) 
    #data_3D = img_3D - img_3D.min()
    #data_3D = data_3D / np.float32(img_3D.max())
    return np.asarray(data_3D, np.float32)

def hist_of_3d_array(array_3d):
    array_1d = np.reshape(array_3d, [np.size(array_3d)])
    num_bins=10
    hist = plt.hist(array_1d, num_bins, facecolor='blue', alpha=0.5)
    return(hist)

def data_augumentation_flip(array_3d):
    flip_index = np.random.randint(0,3)
    if flip_index == 0:
        array_3d = array_3d[::-1,:,:]
    if flip_index == 1:
        array_3d = array_3d[:,::-1,:]
    if flip_index == 2:
        array_3d = array_3d[:,:,::-1]
    return array_3d

def dilation(img_3d):
    d = sitk.GrayscaleDilateImageFilter()
    for i in range(8):
        img_3d = d.Execute(img_3d)
    return img_3d

def pre_process(img_3d):
    img_data = sitk.GetArrayFromImage(img_3d)
    img_data = threscut(img_data)
    img_data = normalize_3D(img_data)
    img = sitk.GetImageFromArray(img_data)
    img = dilation(img)
    return img

def class_model():
    # Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for the three color channels: R, G, and B
    #img_input = layers.Input(shape=(img_width, img_height, color_channels))
    img_input1 = layers.Input(shape=target_size + (1,))
    img_input2 = layers.Input(shape=target_size + (1,))
    img_input = layers.Subtract()([img_input2, img_input1])
    #img_input = layers.concatenate([img_input1, img_input2])
    #x = layers.concatenate([img_input, img_input1])
    #x = layers.concatenate([x, img_input2])
    # First convolution extracts 16 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
#    img_input = layers.Conv3D(kernel_size, conv_window, activation=activation, padding='same', strides=1)(img_input)
    net = layers.Conv3D(kernel_size, conv_window, activation=activation, padding='valid', strides=4)(img_input)
    #x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pooling_window)(net)
    #x = layers.BatchNormalization()(x)
    
    #x = layers.Conv3D(kernel_size, conv_window, activation=activation, padding='valid')(x)
    #branch_1 = x
    
    # Second convolution extracts 32 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = layers.Conv3D(2*kernel_size, conv_window, activation=activation, padding='same', strides=1)(x)
    #x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pooling_window)(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.Conv3D(2*kernel_size, conv_window, activation=activation, padding='same')(x)
    
    # Third convolution extracts 64 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    #x = layers.Conv3D(4*kernel_size, conv_window, activation=activation, padding='same')(x)
    #x = layers.Conv3D(4*kernel_size, conv_window, activation=activation, padding='valid', strides=2)(x)
    #x = layers.Conv3D(4*kernel_size, conv_window, activation=activation)(x)
    #x = layers.MaxPooling3D(pooling_window)(x)
    x = layers.Conv3D(6*kernel_size, conv_window, activation=activation, padding='same', strides=1)(x)
    #branch_2 = x
    #x = layers.MaxPooling3D(pooling_window)(x)
    x = layers.Conv3D(6*kernel_size, conv_window, activation=activation, padding='same')(x)
    x = layers.Conv3D(6*kernel_size, conv_window, activation=activation, padding='same', name='conv_3d')(x)
    #x = layers.BatchNormalization()(x)
    # Flatten feature map to a 1-dim tensor so we can add fully connected layers
    x = layers.Flatten(name='Flatten')(x)
    
    # Create a fully connected layer with ReLU activation and 512 hidden units
    x = layers.Dense(128, activation=activation, name='Dense_512')(x)
    
    # Add Droptout Regularization
    #x = layers.Dropout(dropout)(x)
    
    # Create output layer with a single node and sigmoid activation
    #output = layers.Dense(1, activation='sigmoid')(x)
    output = layers.Dense(len(class_names), activation = 'softmax', name='Dense_2') (x)
    
    
    # Create model: input = input feature map
    # output = input feature map + stacked convolution/maxpooling layers + fully connected layer + sigmoid output layer
    model = Model(inputs=[img_input1,img_input2], outputs=output)
    return model
    
if __name__=="__main__":
    
    import SimpleITK as sitk
    preprocessed_input = sitk.GetArrayFromImage(sitk.ReadImage("/data/wdm_Projects/X-ray-images-classification-with-Keras-TensorFlow/data/training/Probe_full_size/Probe_2.nii"))
    img_0 = sitk.ReadImage("/data/wdm_Projects/X-ray-images-classification-with-Keras-TensorFlow/data/testing/Probe_full_size/Probe_10.nii")
    #img_data = sitk.GetArrayFromImage(img)
#    img_data = preprocessed_input[:, :, :]
    img = pre_process(img_0)
    img_data = sitk.GetArrayFromImage(img)
    
    #img_data = normalize_3D(img_data)
    [depth, height, width] = img_data.shape
#    img = sitk.GetImageFromArray(img_data)
#    img = dilation(img)
#    img_data = sitk.GetArrayFromImage(img)

    data_sum = np.zeros((1,256,128,128,1))
    data_sum[0,:128,:,:,0] =img_data
    second = sitk.GetArrayFromImage(sitk.ReadImage("/data/wdm_Projects/X-ray-images-classification-with-Keras-TensorFlow/data/training/CT_full_size/CT_2.nii"))
    img = sitk.ReadImage("/data/wdm_Projects/X-ray-images-classification-with-Keras-TensorFlow/data/testing/CT_full_size/CT_10.nii")
    #img_data = sitk.GetArrayFromImage(img)
#    img_data = preprocessed_input[:, :, :]
#    img_data = threscut(img_data)
#    img_data = normalize_3D(img_data)
    img = pre_process(img)
    img_data = sitk.GetArrayFromImage(img)
    [depth, height, width] = img_data.shape
    #depth_start = randint(0, depth)
    #width_start = randint(0, width)
#    img = sitk.GetImageFromArray(img_data)
#    img = dilation(img)
#    img_data = sitk.GetArrayFromImage(img)
    
    data_sum[0,128:,:,:,0] = img_data
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))
    #model = VGG16(weights='imagenet')
# =============================================================================
#     gpu = '/gpu:' + str(1)
#     with tf.device(gpu):
# =============================================================================
        
    model = class_model()
    model.summary()
    model.load_weights('/data/wdm_Projects/X-ray-images-classification-with-Keras-TensorFlow/models/weights-improvement-02-16.12.h5')
    predictions = model.predict([ data_sum[:,128:,:,:,:], data_sum[:,:128,:,:,:]])
    #top_1 = decode_predictions(predictions)[0][0]
    print('Predicted class:')
    print(predictions)
    #print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))
    
    predicted_class = np.argmax(predictions)
    print(class_names[predicted_class])
    cam, heatmap = grad_cam_3D(model, image = [data_sum[:,:128,:,:,:], data_sum[:,128:,:,:,:]],
                               category_index=predicted_class, layer_name=2)
    plt.imshow(cam[0,...])
    print(cam.max()-cam.min())
    nor_cam = normalize_3D(cam, nor_min=np.min(cam), nor_max = np.max(cam))
#    nor_cam[nor_cam>0.5] = 2
#    nor_cam[nor_cam<=0.5] = 1
    nor_cam = nor_cam*10
# =============================================================================
    plt.imshow(nor_cam[60,...])
    HEAT = sitk.GetImageFromArray(nor_cam)
    HEAT.CopyInformation(img_0)
    sitk.WriteImage(HEAT, './test_Heat.nii.gz')
# =============================================================================
