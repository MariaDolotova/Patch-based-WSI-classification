'''
vgg19 model with transformed fully connected layers

'''

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.optimizers import Adam



def pretrained_model(img_shape, num_classes):
   
    
    
# LOAD VGG19
    input_tensor = Input(shape=img_shape)
    model = VGG19(weights='imagenet', 
                           include_top=False,
                           input_tensor=input_tensor)

#freeze first 12 layers
    for layer in model.layers[:12]:
        layer.trainable = False 
        print(layer)
    print('****************')
    for layer in model.layers[12:]:
        layer.trainable = True
        print(layer)

# CREATE A TOP MODEL
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(4096, activation='relu', name='fc1'))
    top_model.add(Dense(4096, activation='relu', name='fc2'))
    top_model.add(Dense(num_classes, activation='softmax', name='predictions'))


# CREATE AMODEL FROM VGG16 BY COPYING ALL THE LAYERS OF VGG19
    new_model = Sequential()
    for l in model.layers:
        new_model.add(l)


# CONCATENATE THE TWO MODELS
    new_model.add(top_model)

# COMPILE THE MODEL
    adam = Adam(lr=0.000000001)
    new_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
       
    return new_model


if __name__ == '__main__':
    model=pretrained_model((64, 64, 3), 3)
    model.summary()