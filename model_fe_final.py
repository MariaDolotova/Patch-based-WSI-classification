'''
model used for feature extraction

'''

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalMaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


    
def model_fe(img_shape=(64,64,3)):
      
    #loading the best vgg19 model obtained during training
    model=load_model('best_vgg19_model.h5') 
    model.summary()
    
    
    output_vgg19_conv = model.layers[-2].output 
    
    x=GlobalMaxPooling2D()(output_vgg19_conv)
      
    model_fe = Model(inputs=model.input, outputs=x)
    model_fe.summary()
    
    return model_fe


if __name__ == '__main__':
    model=model_fe((64,64,3))
    model.summary()
    