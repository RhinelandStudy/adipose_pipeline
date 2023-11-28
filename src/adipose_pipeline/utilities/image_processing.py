
import numpy as np
import scipy.ndimage
from skimage.morphology import  remove_small_objects
#import cv2
from skimage.measure import label

def resample(image, spacing, new_spacing=[1, 1, 1],order=3,prefilter=True):
    # Determine current pixel spacing

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image,real_resize_factor,order=order,prefilter=prefilter)

    return image, new_spacing


def slice_largets_connected_componets(labels):
    """Calculate the largest connected component, all the labels are unified to one
    Args:
        labels: ndarray (int or float) label image or volume

    Returns:
        out :ndarray, the input array only with the largest connected component
"""
    mask = np.copy(labels)
    mask[labels > 0] = 1
    for slice in range(labels.shape[0]):
        slice_mask=np.copy(mask[slice,:,:])
        connected_labels, num = label(slice_mask, neighbors=4, return_num=True, connectivity=2)
        #0 is background data, so check with out zero
        #largestCC = np.argmax(np.bincount(connected_labels.flat)[1:])
        slice_mask[connected_labels != 1] = 0
        mask[slice,:,:] =np.copy(slice_mask)

    mask=np.array(mask, dtype=np.int8)
    return labels * mask

def largets_connected_componets(labels):
    """Calculate the largest connected component, all the labels are unified to one
    Args:
        labels: ndarray (int or float) label image or volume
        neighbors : {4, 8}, int, optional
        Whether to use 4- or 8-“connectivity”. In 3D, 4-“connectivity” means connected pixels have to share face,
        whereas with 8-“connectivity”,
        they have to share only edge or vertex. Deprecated, use ``connectivity`` instead.


    Returns:
        out :ndarray, the input array only with the largest connected component
"""
    mask = np.copy(labels)
    mask[labels > 0] = 1
    connected_labels, num = label(mask, return_num=True)
    #0 is background data, so check with out zero
    #largestCC = np.argmax(np.bincount(connected_labels.flat)[1:])
    if num !=1 :
        unique, counts = np.unique(connected_labels, return_counts=True)
        largest=np.argmax(counts[1:]) + 1 #0 is background data, so check with out zero
        mask[connected_labels != largest] = 0

    mask = np.array(mask, dtype=np.int8)

    return labels * mask

def remove_small(labels,remove_size):
    """Remove objects smaller than the remove size, all the labels are unified to one
    Args:
        labels: ndarray (int or float) label image or volume
        remove_size: int, the smallest allowed connected component

    Returns:
        out :ndarray, the input array with smallest connected components removed
"""
    label_mask = np.copy(labels)
    label_mask[labels > 0] = 1
    label_mask = np.array(label_mask, dtype=bool)
    label_mask = remove_small_objects(label_mask,remove_size, in_place=True)
    label_mask = np.array(label_mask, dtype=np.int8)

    return labels * label_mask


def swap_axes(data,plane):
    if plane == 'axial':
        return  data
    elif plane == 'frontal':
        data = np.swapaxes(data, 1, 0)
        return data
    elif plane == 'sagital':
        data = np.swapaxes(data, 2, 0)
        return data

def check_size(data,patch_size):

    x_low=int(np.floor(-1*(data.shape[1]-patch_size[0])/2))
    x_high=int(np.ceil(-1*(data.shape[1]-patch_size[0])/2))



    y_low=int(np.floor(-1*(data.shape[2]-patch_size[1])/2))
    y_high=int(np.ceil(-1*(data.shape[2]-patch_size[1])/2))

    new_arr=np.zeros((data.shape[0],patch_size[0],patch_size[1]))

    new_arr[:,x_low:patch_size[0]-x_high,y_low:patch_size[1]-y_high]=data[:,:,:]

    return new_arr


def organize_data(data,labels,plane,patch_size,lookup,old_spacing,new_spacing):

    #check data order for training depth, height, width
    old_spacing=np.array(old_spacing)
    new_spacing=np.array(new_spacing)
    if (new_spacing[0]-0.5 <= old_spacing[0] <= new_spacing[0]+0.5) and (new_spacing[1]-0.5 <= old_spacing[1] <= new_spacing[1]+0.5) and (new_spacing[2]-0.5 <= old_spacing[2] <= new_spacing[2]+0.5):
        pass
    else:
        print('Changing image Dimensions')
        data,data_spacing=resample(data,old_spacing,new_spacing=new_spacing)
        labels, data_spacing = resample(labels, old_spacing, new_spacing=new_spacing,order=0,prefilter=True)
        labels=np.array(labels,dtype=np.int8)
        print('New Dimensions %s' %data_spacing)

    data=swap_axes(data,plane)
    labels=swap_axes(labels,plane)
    if lookup:
        new_labels=np.zeros(labels.shape)
        for key in lookup.keys():
            new_labels[labels == int(key)]=int(lookup[key])
        labels=np.copy(new_labels)
    #check data size
    data=check_size(data,patch_size)
    labels=check_size(labels,patch_size)

    return data,labels

def change_data_plane(arr, plane='axial',return_index=False):
    if plane == 'axial':
        if return_index:
            return arr,0, arr.shape[0]
        else:
            return arr

    elif plane == 'frontal':
            if len(arr.shape) == 4:
                new_arr = np.zeros((arr.shape[1], arr.shape[1], arr.shape[2],arr.shape[3]))
                for slice in range(arr.shape[3]):
                    aux_arr=arr[:,:,:,slice]
                    aux_arr = np.swapaxes(aux_arr, 1, 0)
                    idx_low = int((new_arr.shape[1] / 2) - (aux_arr.shape[1] / 2))
                    idx_high = int((new_arr.shape[1] / 2) + (aux_arr.shape[1] / 2))
                    new_arr[:, idx_low:idx_high, :,slice] = aux_arr
                if return_index:
                    return new_arr,idx_low,idx_high
                else:
                    return new_arr
            else:
                new_arr = np.zeros((arr.shape[1], arr.shape[1], arr.shape[2]))
                arr = np.swapaxes(arr, 1, 0)
                idx_low = int((new_arr.shape[1] / 2) - (arr.shape[1] / 2))
                idx_high = int((new_arr.shape[1] / 2) + (arr.shape[1] / 2))
                new_arr[:, idx_low:idx_high, :] = arr
                if return_index:
                    return new_arr,idx_low,idx_high
                else:
                    return new_arr
    elif plane == 'sagital':
            if len(arr.shape)== 4:
                new_arr = np.zeros((arr.shape[2], arr.shape[1], arr.shape[2],arr.shape[3]))
                for slice in range(arr.shape[3]):
                    aux_arr=arr[:,:,:,slice]
                    aux_arr = np.swapaxes(aux_arr, 2, 0)

                    idx_low = int((new_arr.shape[2] / 2) - (aux_arr.shape[2] / 2))
                    idx_high = int((new_arr.shape[2] / 2) + (aux_arr.shape[2] / 2))

                    new_arr[:, :, idx_low:idx_high,slice] = aux_arr[:]
                if return_index:
                    return new_arr,idx_low,idx_high
                else:
                    return new_arr
            else:
                new_arr = np.zeros((arr.shape[2], arr.shape[1], arr.shape[2]))
                arr = np.swapaxes(arr, 2, 0)

                idx_low = int((new_arr.shape[2] / 2) - (arr.shape[2] / 2))
                idx_high = int((new_arr.shape[2] / 2) + (arr.shape[2] / 2))

                new_arr[:, :, idx_low:idx_high] = arr
                if return_index:
                    return new_arr,idx_low,idx_high
                else:
                    return new_arr

def find_labels(arr):
    idx=(np.where(arr > 0))
    min_idx=np.min(idx[0])
    max_idx=np.max(idx[0])
    return max_idx,min_idx
