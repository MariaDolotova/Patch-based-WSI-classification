'''
train vgg19 model with training dataset

'''

import numpy as np
import json
import os
from vgg19_final import pretrained_model
from sklearn.utils import class_weight
import _pickle as cPickle
from LoadData_final import DataGenerator, DataGenerator_weighted
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



PATH='path to your data(images)'


    
if __name__ == '__main__':

    
    PATCH_SIZE=64
# Parameters
    params = {'n_classes': 3,
             'n_channels': 1,
             'shuffle': False,
             'batch_size':25}


# list of all patches ids for training
    print(os.getpid())
    with open('list_train_patches_ids.json') as json_data:
        list_train_ids = json.load(json_data)
        json_data.close()

# list of labels for training patches
    with open('list_train_patches_labels.json') as json_data:
        list_train_labels = json.load(json_data)
        json_data.close()
        
# list of all patches ids for validation       
    with open('list_val_patches_ids.json') as json_data:
        list_val_ids = json.load(json_data)
        json_data.close()

# dictionary containes patches ids and corresponding labels      
    with open('patches_labels_dict.json') as json_data:
        labels = json.load(json_data)
        json_data.close()

# dictionary containes patches ids and corresponding patches coordinates        
    with open('patches_coords_dict.json') as json_data:
        coords = json.load(json_data)
        json_data.close()

 
        
    partition={'train' : list_train_ids, 'validation' : list_val_ids}



#preparing weights  
    classes=[0,1,2]
    class_weights = class_weight.compute_class_weight('balanced', np.unique(list_train_labels), list_train_labels)
 # data Generators  
    training_generator = DataGenerator_weighted('train', partition['train'], labels, coords, class_weights, **params)
    validation_generator = DataGenerator('val', partition['validation'], labels, coords, **params)
    
    
    #prepare early stopping callback functions
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2, min_delta=0.0001)
    mc = ModelCheckpoint('best_model_vgg_19.h5', monitor='val_loss', mode='min', save_best_only=True)

    #creating model
    model=pretrained_model((64,64,3), 3)
    model.summary()


    #fit model for several epochs
    history=model.fit_generator(generator=training_generator, 
                                validation_data=validation_generator, 
                                verbose=1, shuffle=False, 
                                epochs=20, use_multiprocessing=False,
                                workers=6, callbacks=[es, mc])

#saving model history
output = open('vgg19_balanced_lr_0.000000001_history.pkl', 'wb')
cPickle.dump(history.history, output, 2)
output.close()

    

    