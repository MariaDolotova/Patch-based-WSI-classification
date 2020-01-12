'''
obtaining features with pretrained feature extraction model (pretrained and transformed vgg19)
'''

import json
import os
import _pickle as cPickle
from LoadData_final import DataGenerator 
from model_fe_final import model_fe

    
if __name__ == '__main__':
    

    print(os.getpid())
    with open('list_train_patches_ids_balanced.json') as json_data:
        list_train_ids = json.load(json_data)
        json_data.close()

        
    with open('list_val_patches_ids_balanced.json') as json_data:
        list_val_ids = json.load(json_data)
        json_data.close()

        
    with open('patches_labels_dict_balanced.json') as json_data:
        labels = json.load(json_data)
        json_data.close()
        
    with open('patches_coords_dict_balanced.json') as json_data:
        coords = json.load(json_data)
        json_data.close()

 
        
    partition={'train' : list_train_ids, 'validation' : list_val_ids}



    # data generators                                                                                                                                           
   
    data_generator=DataGenerator('test', partition['train'], labels, coords,  batch_size=1)

    # create model
    model=model_fe((64,64,3))

    #predict and save predictions
    nb_samples=len(list_train_ids)

    predictions=model.predict_generator(data_generator, steps = nb_samples, use_multiprocessing=False,
                    workers=6, verbose=1)
  

    output = open('features_train_VGG19.pkl', 'wb')
    cPickle.dump(predictions, output, 2)
    output.close()
    