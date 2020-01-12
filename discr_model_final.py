'''
feedforward discriminative model

model is used in E step of EM algorithm (trained for 2 epochs at each iteration 
and then predicts probabilities for each patch)
    
'''

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def discriminative_model():
      
    loss_list=["binary_crossentropy",'binary_crossentropy', 'binary_crossentropy']
        
    inp = Input(shape=512)
        
    x=Dense(256, input_dim=512, activation='relu')(inp)
    x= Dropout(0.5)(x)
    
    x=Dense(256, activation='relu')(x)
    x= Dropout(0.5)(x)
    
    x=Dense(256, activation='relu')(x)
    x= Dropout(0.5)(x)
 
    prediction_0 = Dense(1, activation='sigmoid', name='output_0')(x)
    
    prediction_1 = Dense(1, activation='sigmoid', name='output_1')(x)
    
    prediction_2 = Dense(1, activation='sigmoid', name='output_2')(x)
    
    model = Model(inputs=inp, outputs=[prediction_0, prediction_1, prediction_2])
    model.compile(optimizer=Adam(lr=1e-6), loss=loss_list, metrics=['accuracy'], weighted_metrics=['accuracy'])
    

    return model


if __name__ == '__main__':
    model=discriminative_model()
    model.summary()
    x=model.get_weights()