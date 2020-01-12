from pMap_threshold import mapping_and_tresholding
import _pickle as cPickle  
import numpy as np
import json


with open('list_train_info.json') as json_data: #image level
    list_train_info = json.load(json_data)
    json_data.close()
        
with open('list_train_patches_info.json') as json_data: #patch level
    list_train_patches_info = json.load(json_data)
    json_data.close()

#loading predictions of best discriminative model
input = open('discr_pred/predictions_discr_model_23.pkl', 'rb')
predictions = cPickle.load(input)
input.close()

input = open('patches_labels_dict_cat.pkl', 'rb')
patches_labels_dict_cat = cPickle.load(input)
input.close()

mapping_and_tresholding(predictions, patches_labels_dict_cat, list_train_info, list_train_patches_info, 0)

discr_labels=[]
discr_features=[]
non_discr_patches=[]

m=0

#getting label histograms (sums of probabilities) for each image

for key in list_train_info:
    patches_len=list_train_info.get(key).get('patches')
    patches_label=list_train_info.get(key).get('label')

    features=np.empty(3)
    f0=0
    f1=0
    f2=0
    for l in range(patches_len):
        id=list_train_patches_info[m].get('patch_id')
        if (patches_labels_dict_cat.get(id)[0]!=0 or patches_labels_dict_cat.get(id)[1]!=0 or patches_labels_dict_cat.get(id)[2]!=0):
            f0+=predictions[0][m]
            f1+=predictions[1][m]
            f2+=predictions[2][m]
        else:
            non_discr_patches.append(id)
        m+=1 
    features[0]=f0
    features[1]=f1
    features[2]=f2
    discr_features.append(features)
    discr_labels.append(patches_label)

#saving sums of probabilities and corresponding image-level labels

output = open('discr_labels.pkl', 'wb')
cPickle.dump(discr_labels, output, 2)
output.close()

output = open('discr_features.pkl', 'wb')
cPickle.dump(discr_features, output, 2)
output.close()