'''
training of discriminative model for 2 epochs and obtaining predictions (E-step of EM algorithm)
'''
import numpy as np
import json
import os
from sklearn.utils import class_weight
import _pickle as cPickle
from LoadData_final import DataGeneratorDiscr
from discr_model_final import discriminative_model
from pMap_threshold_final import mapping_and_tresholding
import matplotlib.pyplot as plt


if __name__ == '__main__':
    PATCH_SIZE=64
# Parameters
    params = {'n_classes': 3,
             'n_channels': 1,
             'shuffle': False,
             'batch_size':25}

    print(os.getpid())
    with open('list_train_patches_ids.json') as json_data:
        list_train_ids = json.load(json_data)
        json_data.close()

    with open('list_train_patches_labels.json') as json_data:
        list_train_labels = json.load(json_data)
        json_data.close()
    
# dictionary, containes image_id and corresponding number of rows and columns in patches grid, number of patches and label   
    with open('list_train_info.json') as json_data: #image level
        list_train_info = json.load(json_data)
        json_data.close()
        
# list of dictionaries - each dictionary containes patch_id and patch_index in whole image grid      
    with open('list_train_patches_info.json') as json_data: #patch level
        list_train_patches_info = json.load(json_data)
        json_data.close()

        
    with open('list_val_patches_ids.json') as json_data:
        list_val_ids = json.load(json_data)
        json_data.close()
                
    with open('patches_labels_dict.json') as json_data:
        labels = json.load(json_data)
        json_data.close()
        
    with open('patches_coords_dict.json') as json_data:
        coords = json.load(json_data)
        json_data.close()
        
        
    partition={'train' : list_train_ids, 'validation' : list_val_ids}
    
    #features created with VGG19 model
    input = open('features_train_VGG19.pkl', 'rb')
    features = cPickle.load(input)
    input.close()
    
    #dictionary, contains patch_id and corresponding label in categorical view
    input = open('patches_labels_dict_cat.pkl', 'rb')
    patches_labels_dict_cat = cPickle.load(input)
    input.close()
    
    
    #data generators
    training_generator = DataGeneratorDiscr('train', partition['train'], patches_labels_dict_cat, coords, features, **params)
    test_generator=DataGeneratorDiscr('test', partition['train'], patches_labels_dict_cat, coords,  features, batch_size=1)
    
    nb_samples=len(list_train_ids)
    
 
    
    model=discriminative_model()
    

    count=0
    while count<100:

        #fit model for 2 epochs
        
        history=model.fit_generator(generator=training_generator, verbose=1, shuffle=False, epochs=2, use_multiprocessing=False,
                workers=6)
        model.save("discr_model/model_discr_fit_on_train_patches_"+str(count)+".h5")
        output = open('discr_model_hist/history_discr_model_'+str(count)+'.pkl', 'wb')
        cPickle.dump(history.history, output, 2)
        output.close()
       
        #predict

        predictions=model.predict_generator(test_generator, steps = nb_samples, use_multiprocessing=False,
                    workers=6, verbose=1)
        output = open('discr_pred/predictions_discr_model_'+str(count)+'.pkl', 'wb')
        cPickle.dump(predictions, output, 2)
        output.close()
        
        #mapping, thresholding and re-labeling
        
        mapping_and_tresholding(predictions, patches_labels_dict_cat, 
                            list_train_info, list_train_patches_info, count)
        #create updated data generator with updated labels before continue fitting
        training_generator = DataGeneratorDiscr('train', partition['train'], patches_labels_dict_cat, coords, features, **params)
        count=count+1