'''
DataGenerators and additinal functions for patch generation 
    
'''

import numpy as np
import tensorflow.keras
from skimage import io
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt


PATH='path to your data'
PATCH_SIZE=64

# generate grid of patches
def generate_whole_image_patches(x, height1, width1, size=64):
    height, width, depth = x.shape
    patches = []
    for i in range(height // size):
        for j in range(width // size):
            patch = x[i*size:(i+1)*size, j*size:(j+1)*size, :]
            patches.append(patch)
    patches = np.array(patches)
    return patches


#generate patches with certain persent of tissue by using Otsu mask 
def generate_patches(img, size=64):

    height_new=img.shape[0]//size
    width_new=img.shape[1]//size
    img= img[0:height_new*size, 0:width_new*size,:]
    grayscale = rgb2gray(img)
    plt.imshow(grayscale)
    thresh = threshold_otsu(grayscale)
    binary = grayscale < thresh
    binary.astype(float)
    image_downscaled = downscale_local_mean(binary, (size, size))
    reshaped=np.reshape(image_downscaled, image_downscaled.shape[0]*image_downscaled.shape[1])  
    patches_ind = np.where(reshaped >= 0.7)  
    reshaped[patches_ind] = 1
    reshaped=np.reshape(reshaped,(image_downscaled.shape[0],image_downscaled.shape[1]))
    plt.imshow(reshaped, cmap=plt.cm.gray)
    patches = []
        
    for index in patches_ind[0]:
        y=index%reshaped.shape[1]
        x=index//reshaped.shape[1]
        patch = img[x*size:(x+1)*size, y*size:(y+1)*size, :]
        patches.append(patch)
      
    patches = np.array(patches)

    return patches, patches_ind, image_downscaled.shape[0], image_downscaled.shape[1]



#generates patches coordinates
def generate_patches_coords(img, size=64):
    
    height_new=img.shape[0]//size
    width_new=img.shape[1]//size
    img= img[0:height_new*size, 0:width_new*size,:]

    grayscale = rgb2gray(img)
    thresh = threshold_otsu(grayscale)
    binary = grayscale < thresh
    binary.astype(float)
    image_downscaled = downscale_local_mean(binary, (size, size))

    reshaped=np.reshape(image_downscaled, image_downscaled.shape[0]*image_downscaled.shape[1])
    
    patches_ind = np.where(reshaped >= 0.6)
    
    reshaped[patches_ind] = 1
    reshaped=np.reshape(reshaped,(image_downscaled.shape[0],image_downscaled.shape[1]))
   
    list_coords=[]
    
    for index in patches_ind[0]:
        patch_coords={}
        patch_coords['y']=int(index%reshaped.shape[1])
        patch_coords['x']=int(index//reshaped.shape[1])
        list_coords.append(patch_coords)

    return list_coords


# generates additional patches information required by algorithm
def generate_patches_info(patch_label, list_ids, img_id, img, size=64):
    
    height_new=img.shape[0]//size
    width_new=img.shape[1]//size
    img= img[0:height_new*size, 0:width_new*size,:]

    grayscale = rgb2gray(img)
    thresh = threshold_otsu(grayscale)
    binary = grayscale < thresh
    binary.astype(float)
    image_downscaled = downscale_local_mean(binary, (size, size))

    reshaped=np.reshape(image_downscaled, image_downscaled.shape[0]*image_downscaled.shape[1])
    
    patches_ind = np.where(reshaped >= 0.6)
    
    reshaped[patches_ind] = 1
    reshaped=np.reshape(reshaped,(image_downscaled.shape[0],image_downscaled.shape[1]))
   
    list_info=[]
    list_ids.get(img_id)['height']=int(reshaped.shape[0])
    list_ids.get(img_id)['width']=int(reshaped.shape[1])
    
    
    for index in patches_ind[0]:
        patch_info={}
        patch_info['y']=int(index%reshaped.shape[1])
        patch_info['x']=int(index//reshaped.shape[1])
        patch_info['index']=int(index)
        
        
        list_info.append(patch_info)
        
    list_ids.get(img_id)['patches']=int(len(patches_ind[0]))
    list_ids.get(img_id)['label']=int(patch_label)
    return list_info


#get one patch by coordinates
def get_patch(img, img_id, x, y, size=64):
    patch= img[x*size:(x+1)*size, y*size:(y+1)*size, :]
    return patch


#data genetaror for VGG19 training
class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, name, list_IDs, labels, coords, dim=(64,64,3), batch_size=32, n_channels=1,
                 n_classes=3, shuffle=False):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = False
        self.on_epoch_end()
        self.coords=coords
        self.dim=dim
        self.name=name
        self.classes=[0, 1, 2]

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp, index)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    

    def __data_generation(self, list_IDs_temp, index):
        'Generates data containing batch_size samples' 

        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)
        img_id_prev=''
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img_id=ID.split('_')
            img_id=str(img_id[0])+'_'+str(img_id[1])
            if img_id_prev!=img_id:
                img=io.imread(str(PATH)+str(img_id)+'.png')
                img_id_prev=img_id
            x_coord=self.coords[ID]['x']
            y_coord=self.coords[ID]['y']            
            sample=get_patch(img, img_id, x_coord, y_coord, 64)

    
            X[i,] = sample 
  
            y[i] = self.labels[ID]


        return X, tensorflow.keras.utils.to_categorical(y, num_classes=self.n_classes)
    
 
    
    
# data generator for discriminative model    
class DataGeneratorDiscr(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, name, list_IDs, labels, coords, data, dim=(512,), batch_size=32, n_channels=1,
                 n_classes=3, shuffle=False):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = False
        self.on_epoch_end()
        self.coords=coords
        self.name=name
        self.classes=[0, 1, 2]
        self.data=data
        self.dim=dim

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, index)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, list_IDs_temp, index):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, *self.dim))
        y0 = np.empty((self.batch_size), dtype=int)
        y1=np.empty((self.batch_size), dtype=int)
        y2=np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            sample_index=self.list_IDs.index(ID)
            sample=self.data[sample_index]
            X[i,] = sample  
            y0[i] = self.labels[ID][0]
            y1[i]=self.labels[ID][1]
            y2[i]=self.labels[ID][2]
     

        return X, {'output_0': y0, 'output_1': y1, 'output_2': y2} 

#data genetaror for VGG19 training(weighted)
class DataGenerator_weighted(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, name, list_IDs, labels, coords, weights,dim=(64,64,3), batch_size=32, n_channels=1,
                 n_classes=3, shuffle=False):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = False
        self.on_epoch_end()
        self.coords=coords
        self.dim=dim
        self.name=name
        self.classes=[0, 1, 2]
        self.weights=weights

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y, weights = self.__data_generation(list_IDs_temp, index)

        return X, y, weights

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            


    def __data_generation(self, list_IDs_temp, index):
        'Generates data containing batch_size samples' 
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)
        sample_weights = np.empty((self.batch_size), dtype=float)
        img_id_prev=''
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img_id=ID.split('_')
            img_id=str(img_id[0])+'_'+str(img_id[1])
            if img_id_prev!=img_id:
                img=io.imread(str(PATH)+str(img_id)+'.png')
                img_id_prev=img_id
            x_coord=self.coords[ID]['x']
            y_coord=self.coords[ID]['y']            
            sample=get_patch(img, img_id, x_coord, y_coord, 64)
   
            X[i,] = sample 
  
            y[i] = self.labels[ID]
            sample_weights[i]=self.weights[y[i]]


        return X, tensorflow.keras.utils.to_categorical(y, num_classes=self.n_classes), sample_weights
    