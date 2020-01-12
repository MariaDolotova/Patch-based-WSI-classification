import numpy as np
import scipy.ndimage as ndimage
from vispy.io import imsave




def mapping_and_tresholding(predictions, patches_labels_dict_cat, 
                            list_train_info, list_train_patches_info, count):

        h0=[]
        h1=[]
        h2=[]
        

        j=0
        m=0

#prepare lists of probabilities for each class
        for key in list_train_info:
            patches_len=list_train_info.get(key).get('patches')
            patches_label=list_train_info.get(key).get('label')
            for i in range(patches_len):
             if patches_label==0:
                    h0.append(predictions[0][j][0])
                    h0.append(predictions[1][j][0])                   
                    h0.append(predictions[2][j][0])
             elif patches_label==1:
                    h1.append(predictions[0][j][0])
                    h1.append(predictions[1][j][0])
                    h1.append(predictions[2][j][0])
            else:
                    h2.append(predictions[0][j][0])
                    h2.append(predictions[1][j][0])
                    h2.append(predictions[2][j][0])
            j=j+1
                
     #sorting and getting percentiles  
        h0.sort()
        h1.sort()
        h2.sort()
        thresh0 = np.percentile(h0, 70)
        thresh1 = np.percentile(h1, 70)
        thresh2 = np.percentile(h2, 70)

      
     #    creating probability maps for each biopsy based on predicted probabilities

        j=0
        
        for key in list_train_info:
                           
            patches_len=list_train_info.get(key).get('patches')
            height=list_train_info.get(key).get('height')
            width=list_train_info.get(key).get('width')
            patches_label=list_train_info.get(key).get('label')
            ar0=np.zeros(height*width)
            ar_im0=[]
            ar1=np.zeros(height*width)
            ar_im1=[]
            ar2=np.zeros(height*width)
            ar_im2=[]
            for i in range(patches_len):
                index=list_train_patches_info[j].get('patch_index')
                id=list_train_patches_info[j].get('patch_id')
                ar0[index]=predictions[0][j][0]
                ar_im0.append(predictions[0][j][0])
                ar1[index]=predictions[1][j][0]
                ar_im1.append(predictions[1][j][0])
                ar2[index]=predictions[2][j][0]
                ar_im2.append(predictions[2][j][0])
                j=j+1
            
            ar_im0.sort()
            thresh_im0 = np.percentile(ar_im0, 70)            
            ar_im1.sort()
            thresh_im1 = np.percentile(ar_im1, 70)            
            ar_im2.sort()
            thresh_im2 = np.percentile(ar_im2, 70)

           
            ar0=ar0.reshape(height, width)
            gaussian0=ndimage.gaussian_filter(ar0, sigma=1, order=0)
            gaussian0=gaussian0.reshape((height*width))

             
            ar1=ar1.reshape(height, width)
            gaussian1=ndimage.gaussian_filter(ar1, sigma=1, order=0)
            gaussian1=gaussian1.reshape((height*width))
            
            ar2=ar2.reshape(height, width)
            gaussian2=ndimage.gaussian_filter(ar2, sigma=1, order=0)
            gaussian2=gaussian2.reshape((height*width))
        
            color=np.zeros((height*width,3))
            
            for l in range(patches_len):
                index=list_train_patches_info[m].get('patch_index')
                id=list_train_patches_info[m].get('patch_id')

                if patches_label==0:
                    thresh=min(thresh0, thresh_im0)
                    current_label=patches_labels_dict_cat.get(id)[0]
                    
                    if (gaussian0[index]<=thresh and (gaussian1[index]<=0.5 and gaussian2[index]<=0.5)):
                        non_discr_0=True
                    else:
                        non_discr_0=False                        
                    
                    if non_discr_0==True:
                        if current_label==1:
                            patches_labels_dict_cat.get(id)[0]=0
                        color[index][0]=255
                        color[index][1]=255
                           
                    else:
                        if current_label==0:
                            patches_labels_dict_cat.get(id)[0]=1
                        color[index][0]=255
                            
                     
                elif patches_label==1:
                    thresh=min(thresh1, thresh_im1)
                    current_label=patches_labels_dict_cat.get(id)[1]
                    if (gaussian1[index]<=thresh and (gaussian0[index]<=0.5 and gaussian2[index]<=0.5)):
                        non_discr_0=True
                    else:
                        non_discr_0=False
                    
                    if non_discr_0==True:
                        if current_label==1:
                            patches_labels_dict_cat.get(id)[1]=0
                        color[index][0]=255
                        color[index][1]=255

                            
                    else:
                        if current_label==0:
                            patches_labels_dict_cat.get(id)[1]=1
                        color[index][1]=255

                                              
                else:
                    thresh=min(thresh2, thresh_im2)
                    current_label=patches_labels_dict_cat.get(id)[2]
                    if (gaussian2[index]<=thresh and (gaussian0[index]<=0.5 and gaussian1[index]<=0.5)):
                         non_discr_0=True
                    else:
                        non_discr_0=False
                                           
                    if non_discr_0==True:
                        if current_label==1:
                            patches_labels_dict_cat.get(id)[2]=0
                        color[index][0]=255
                        color[index][1]=255

                           
                    else:
                        if current_label==0:
                            patches_labels_dict_cat.get(id)[2]=1
                        color[index][2]=255 
                            

                m=m+1
            
            color=color.reshape((height,width,3))
            filename='maps/'+str(key)+'_color_map_label_'+str(patches_label)+'_iter_'+str(count)+'.png'
            imsave(filename, color)
                
                
            return 
        