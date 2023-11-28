from __future__ import division

import nipype.pipeline.engine as pe
from nipype import SelectFiles
import nipype.interfaces.utility as util
from nipype import IdentityInterface, DataSink

import os
from .configoptions import multiviewModel,singleViewModels,localizationModels,control_images

import numpy as np
from keras.models import load_model
import adipose_pipeline.loss as loss
from adipose_pipeline.utilities import own_itk as oitk
from adipose_pipeline.utilities.misc import locate_file, locate_dir
from adipose_pipeline.utilities.image_processing import change_data_plane,largets_connected_componets,swap_axes
from adipose_pipeline.utilities.visualization_misc import multiview_plotting
from skimage.measure import perimeter
import pandas as pd


def clean_segmentations(label_map):

    new_label_map=np.copy(label_map)
    new_label_map= largets_connected_componets(new_label_map)
    return new_label_map


def test_multiplane(params,data):
    """Segmentation network for the probability maps of frontal,axial and sagittal
    Args:
        params: train parameters of the network
        data: ndarray (int or float) containing 15 probability maps

    Returns:
        out :ndarray, prediction array of 5 classes
"""
    # ============  Path Configuration ==================================

    model_name = params['modelParams']['ModelName']
    #model_path = params['modelParams']['SavePath'] + model_name + '/'
    model_path = params['modelParams']['SavePath']

    # ============  Model Configuration ==================================
    n_ch = params['modelParams']['nChannels']
    nb_classes = params['modelParams']['nClasses']
    batch_size = params['modelParams']['BatchSize']
    MedBalFactor=params['modelParams']['MedFrequency']
    loss_type=params['modelParams']['Loss_Function']
    sigma=params['modelParams']['GradientSigma']
    print('-' * 30)
    print('model path')
    print(model_path + '/' + model_name + '_best_weights.h5')

    model = load_model(model_path + '/' + model_name + '_best_weights.h5',
                       custom_objects={'logistic_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'weighted_logistic_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'weighted_gradient_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'mixed_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'dice_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'dice_coef': loss.dice_coef,
                                       'dice_coef_0': loss.dice_coef_0,
                                       'dice_coef_1': loss.dice_coef_1,
                                       'dice_coef_2': loss.dice_coef_2,
                                       'dice_coef_3': loss.dice_coef_3,
                                       'dice_coef_4': loss.dice_coef_4,
                                       'average_dice_coef': loss.average_dice_coef})



    print('-' * 30)
    print('Evaluating Multiview model  ...')
    print('-' * 30)


    y_predict = model.predict(data, batch_size=batch_size, verbose=0)

    # Reorganize prediction data
    y_predict = np.argmax(y_predict, axis=-1)
    y_predict = y_predict.reshape(data.shape[1], data.shape[2], data.shape[3])
    y_predict = np.asarray(y_predict, dtype=np.int16)
    print(y_predict.shape)


    return y_predict

def test_model(params,data):
    """Segmentation network for each view (frontal,axial and sagittal)
    Args:
        params: train parameters of the network
        data: ndarray (int or float) containing the fat image

    Returns:
        out :ndarray, prediction array of 5 classes for each view
    """
    # ============  Path Configuration ==================================

    model_name = params['modelParams']['ModelName']
    #model_path = params['modelParams']['SavePath'] + model_name + '/'
    model_path = params['modelParams']['SavePath']

    # ============  Model Configuration ==================================
    n_ch = params['modelParams']['nChannels']
    nb_classes = params['modelParams']['nClasses']
    batch_size = params['modelParams']['BatchSize']
    MedBalFactor = params['modelParams']['MedFrequency']
    loss_type = params['modelParams']['Loss_Function']
    sigma = params['modelParams']['GradientSigma']
    plane = params['modelParams']['Plane']
    print('-' * 30)
    print(plane)
    print('Testing %s'%model_name)
    print('model path')
    print(model_path + '/' + model_name + '_best_weights.h5')

    model = load_model(model_path + '/' + model_name + '_best_weights.h5',
                       custom_objects={'logistic_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'weighted_logistic_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'weighted_gradient_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'mixed_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'dice_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'dice_coef': loss.dice_coef,
                                       'dice_coef_0': loss.dice_coef_0,
                                       'dice_coef_1': loss.dice_coef_1,
                                       'dice_coef_2': loss.dice_coef_2,
                                       'dice_coef_3': loss.dice_coef_3,
                                       'dice_coef_4': loss.dice_coef_4,
                                       'average_dice_coef': loss.average_dice_coef})




    X_test=np.copy(data)

    print('input size')
    print(X_test.shape)
    X_test,idx_low,idx_high = change_data_plane(X_test, plane=params['modelParams']['Plane'],return_index=True)

    X_test=X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], n_ch))

    # ============  Evaluating ==================================
    print('-' * 30)
    print('Evaluating %s...'%plane)
    print('-' * 30)
    y_predict = model.predict(X_test, batch_size=batch_size, verbose=0)
    print('Change Plane to %s'%plane)
    y_predict = change_data_plane(y_predict, plane=params['modelParams']['Plane'])
    y_predict=y_predict[idx_low:idx_high, :, :, :]
    print(y_predict.shape)


    return y_predict


def test_localization_model(params,data):
    """Segmentation network for localizing the region of intertest (frontal,axial and sagittal)
    Args:
        params: train parameters of the network
        data: ndarray (int or float) containing the fat image

    Returns:
        out : slices boundaries of the ROI
    """
    # ============  Path Configuration ==================================

    model_name = params['modelParams']['ModelName']
    #model_path = params['modelParams']['SavePath'] + model_name + '/'
    model_path = params['modelParams']['SavePath']

    # ============  Model Configuration ==================================
    n_ch = params['modelParams']['nChannels']
    nb_classes = params['modelParams']['nClasses']
    batch_size = params['modelParams']['BatchSize']
    MedBalFactor = params['modelParams']['MedFrequency']
    loss_type = params['modelParams']['Loss_Function']
    sigma = params['modelParams']['GradientSigma']
    plane = params['modelParams']['Plane']
    print('-' * 30)
    print(plane)
    print('Testing %s'%model_name)
    print('model path')
    print(model_path + '/' + model_name + '_best_weights.h5')

    model = load_model(model_path + '/' + model_name + '_best_weights.h5',
                       custom_objects={'logistic_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'weighted_logistic_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'weighted_gradient_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'mixed_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'dice_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'dice_coef': loss.dice_coef,
                                       'dice_coef_0': loss.dice_coef_0,
                                       'dice_coef_1': loss.dice_coef_1,
                                       'dice_coef_2': loss.dice_coef_2,
                                       'dice_coef_3': loss.dice_coef_3,
                                       'dice_coef_4': loss.dice_coef_4,
                                       'average_dice_coef': loss.average_dice_coef})

    X_test = np.copy(data)
    X_test=swap_axes(X_test,plane)
    print('input size')
    print(X_test.shape)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], n_ch))

    # ============  Evaluating ==================================
    print('-' * 30)
    print('Evaluating %s...'%plane)
    print('-' * 30)
    y_predict = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_predict = np.argmax(y_predict, axis=-1)
    y_predict=swap_axes(y_predict,plane)
    print('Change Plane to %s'%plane)
    print(y_predict.shape)
    high_idx,low_idx=find_unique_index_slice(y_predict)


    return high_idx,low_idx


def find_unique_index_slice(data):
    new_data=np.copy(data)
    aux_index=np.where(new_data==2)
    high_aux_index=np.max(aux_index[0])
    low_aux_index = np.min(aux_index[0])

    new_data[data==0]=2
    index=[]
    for z in range(data.shape[0]):
        slice_values=np.unique(new_data[z,:,:])
        if len(slice_values)==1:
            index.append(z)

    higher_index=(np.max(index)+ high_aux_index) // 2
    lower_index=(np.min(index) + low_aux_index) // 2

    return higher_index,lower_index


def run_adipose_localization(data):
    
    planes=['frontal','sagital']
    high_idx=0
    low_idx=0
    for plane in planes:
        plane_model = os.path.join(localizationModels, 'Loc_CDFNet_Baseline_'+str(plane))
        params_path = os.path.join(plane_model, 'train_parameters.npy')
        params = np.load(params_path).item()
        params['modelParams']['SavePath'] = plane_model
        tmp_high_idx,tmp_low_idx=test_localization_model(params,data)
        high_idx += tmp_high_idx
        low_idx +=tmp_low_idx

    high_idx=int(high_idx // 2)
    low_idx=int(low_idx // 2)
    return high_idx,low_idx




def run_adipose_segmentation(data):
    # ============  Load Params ==================================
    # Multiviewmodel
    multiview_path = os.path.join(multiviewModel, 'Baseline_Mixed_Multi_Plane')
    multiview_params = np.load(os.path.join(multiview_path,'train_parameters.npy')).item()
    multiview_params['modelParams']['SavePath']= multiview_path
    nbclasses = multiview_params['modelParams']['nClasses']
    # uni axial Model Path
    base_line_dir_axial  = os.path.join(singleViewModels,'CDFNet_Baseline_axial')
    base_line_dir_frontal= os.path.join(singleViewModels,'CDFNet_Baseline_frontal')
    base_line_dir_sagital= os.path.join(singleViewModels,'CDFNet_Baseline_sagital')
    
    base_line_dirs=[]
    base_line_dirs.append(base_line_dir_axial)
    base_line_dirs.append(base_line_dir_frontal)
    base_line_dirs.append(base_line_dir_sagital)

    test_data = np.zeros((1, multiview_params['modelParams']['PatchSize'][0],
                          multiview_params['modelParams']['PatchSize'][1],
                          multiview_params['modelParams']['PatchSize'][2], len(base_line_dirs) * nbclasses))
    i = 0
    for plane_model in base_line_dirs:
        print(plane_model)
        params_path = os.path.join(plane_model , 'train_parameters.npy')
        params = np.load(params_path).item()
        params['modelParams']['SavePath'] = plane_model
        test_data[0, 0:data.shape[0], :, :, i * nbclasses:(i + 1) * nbclasses] = test_model(params, data)
        i += 1
    final_img = test_multiplane(multiview_params, test_data)

    return final_img


def extreme_AAT_increase_flag(predict_array,threshold=0.3):

    extreme_increase_flag = False
    for slice in range(1,(predict_array.shape[0]-1)) :
        previous_sat =np.sum(predict_array[slice-1,:,:] == 1)
        previous_vat =np.sum(predict_array[slice-1,:,:] == 2)
        current_sat=np.sum(predict_array[slice,:,:] == 1)
        current_vat =np.sum(predict_array[slice,:,:] == 2)
        following_sat=np.sum(predict_array[slice+1,:,:] == 1)
        following_vat=np.sum(predict_array[slice+1,:,:] == 2)

        sat_threshold=current_sat*threshold
        vat_threshold = current_vat * threshold

        if np.abs(current_sat-previous_sat) > sat_threshold or np.abs(current_sat-following_sat) > sat_threshold:
            extreme_increase_flag= 'SAT increase over the threshold'
        elif np.abs(current_vat-previous_vat) > vat_threshold or np.abs(current_vat-following_vat) > vat_threshold:
            extreme_increase_flag = 'VAT increase over the threshold'

    return  extreme_increase_flag



def perimeter_calculation(label_mask):

    perimeter_val=[]

    for slice in range(label_mask.shape[0]):
        perimeter_val.append(perimeter(label_mask[slice,:,:]))

    average_perimeter = np.sum(perimeter_val)/label_mask.shape[0]

    return  average_perimeter

def calculate_areas(final_img, img_spacing,columns):

    if len(final_img.shape) == 2:
        final_img =np.reshape(final_img,(1,final_img.shape[0],final_img.shape[1]))


    pixel_area = (img_spacing[0] * img_spacing[1]) * 0.01

    statiscs_matrix = np.zeros(( 1, columns),dtype=float)

    abdominal_region_mask= np.zeros(final_img.shape,dtype=bool)
    abdominal_region_mask[final_img >= 1] = True

    #  Metric Measurements
    statiscs_matrix[0,0]= final_img.shape[0] * img_spacing[2] * 0.1 # Height ROI
    statiscs_matrix[0,1]= (np.sum(abdominal_region_mask) * pixel_area) / final_img.shape[0] #Average_Area
    statiscs_matrix[0, 2] = perimeter_calculation(abdominal_region_mask) * img_spacing[0] * 0.1  # Average_perimeter

    return  statiscs_matrix.round(decimals=4)

def calculate_volumes(final_img, water_array, fat_array, img_spacing,columns,weighted=True):

    if len(final_img.shape) == 2:
        final_img =np.reshape(final_img,(1,final_img.shape[0],final_img.shape[1]))
        water_array = np.reshape(water_array, (1, water_array.shape[0], water_array.shape[1]))
        fat_array = np.reshape(water_array, (1, fat_array.shape[0], fat_array.shape[1]))

    voxel_volume = (img_spacing[0] * img_spacing[1] * img_spacing[2]) * 0.001



    abdominal_region_mask= np.zeros(final_img.shape,dtype=bool)
    abdominal_region_mask[final_img >= 1] = True

    vat_mask = np.zeros(final_img.shape, dtype=bool)
    vat_mask[final_img == 2] = True

    sat_mask = np.zeros(final_img.shape, dtype=bool)
    sat_mask[final_img == 1] = True

    combine_array = water_array + fat_array
    fat_fraction_array = np.clip(fat_array,0.00001,None) / np.clip(combine_array,0.00001,None)

    if weighted:
        vat_fraction = np.sum(fat_fraction_array[vat_mask])
        sat_fraction = np.sum(fat_fraction_array[sat_mask])
        abdominal_region_fraction= np.sum (fat_fraction_array[abdominal_region_mask])
    else:
        sat_fraction = np.sum(sat_mask)
        vat_fraction = np.sum(vat_mask)
        abdominal_region_fraction= np.sum (abdominal_region_mask)

    #print('the vat fraction values are %d, the sat fraction values are %d' % (vat_fraction, sat_fraction))

    statiscs_matrix = np.zeros(( 1, columns),dtype=float)

    #  Metric Measurements
    statiscs_matrix[0,0] = abdominal_region_fraction * voxel_volume # Volume of Abdominal Region

    # Pixel not Weighted
    statiscs_matrix[0, 1] = sat_fraction * voxel_volume  # VOL_SAT
    statiscs_matrix[0, 2] = vat_fraction * voxel_volume  # VOL_VAT
    statiscs_matrix[0, 3] = statiscs_matrix[0, 1] + statiscs_matrix[0, 2]  # VOL_AAT

    statiscs_matrix[0, 4] = statiscs_matrix[0,2] / statiscs_matrix[0, 1]  # VAT/SAT
    statiscs_matrix[0, 5] = statiscs_matrix[0, 2] / statiscs_matrix[0, 3]  # VAT/AAT
    statiscs_matrix[0, 6] = statiscs_matrix[0, 1] / statiscs_matrix[0, 3]  # SAT/AAT

    #statiscs_matrix[0,17]=extreme_AAT_increase_flag(final_img,threshold=increase_thr)

    return statiscs_matrix.round(decimals=4)

def calculate_statistics_v2(final_img, water_array, fat_array, low_idx, high_idx, columns,base_variables_len,img_spacing,increase_thr=0.15,comparments=4):



    statiscs_matrix = np.zeros((1, len(columns)),dtype=object)
    size_base=base_variables_len['Area']+base_variables_len['Volume']+base_variables_len['W_Volume']


    print('Whole Body')

    final_area=base_variables_len['Area']
    statiscs_matrix[0, 0:final_area]=calculate_areas(final_img,img_spacing,base_variables_len['Area'])
    final_volume=final_area +base_variables_len['Volume']
    statiscs_matrix[0,final_area:final_volume]=calculate_volumes(final_img,water_array,fat_array,
                                                                              img_spacing, base_variables_len['Volume'], weighted=False)
    final_volume2= final_volume +base_variables_len['Volume']
    statiscs_matrix[0,final_volume:final_volume2] = calculate_volumes(final_img,water_array,fat_array,img_spacing,
                                                                                                      base_variables_len['Volume'],
                                                                                                      weighted=True)

    if comparments !=0:

        interval = (high_idx - low_idx)
        interval_step = np.around((interval / comparments), decimals=2)
        interval_steps = np.arange(0, interval, interval_step).round(decimals=2)
        if not interval_steps[-1] == interval:
            interval_steps = np.append(interval_steps, interval)

        slice=0

        for i in np.arange(0,comparments):
            lower_limit=np.ceil(interval_steps[i])
            higher_limit=np.floor(interval_steps[i+1])
            complete_slices=np.arange(lower_limit,higher_limit)
            #print (complete_slices)
            #Calculate Complete Slices
            if complete_slices.size != 0 :
                min_slice=int(np.min(complete_slices))
                max_slice=int(np.max(complete_slices))+1


                area_initial_len= size_base * (i+1)
                area_final_len =size_base * (i+1)+base_variables_len['Area']

                #TO-DO check that is empty
                if statiscs_matrix[0, area_initial_len + 1] != 0:

                    statiscs_matrix[0, area_initial_len:area_final_len] = statiscs_matrix[0, area_initial_len:area_final_len] + calculate_areas(final_img[min_slice:max_slice, :, :],img_spacing,base_variables_len['Area'])
                    statiscs_matrix[0,area_initial_len+1] = statiscs_matrix[0,area_initial_len+1] / 2
                    statiscs_matrix[0, area_initial_len + 2] = statiscs_matrix[0, area_initial_len + 2] / 2

                else:
                    statiscs_matrix[0, area_initial_len:area_final_len] = statiscs_matrix[0,area_initial_len:area_final_len] + calculate_areas(final_img[min_slice:max_slice, :, :], img_spacing, base_variables_len['Area'])

                vol_final_len= area_final_len + base_variables_len['Volume']

                statiscs_matrix[0, area_final_len:vol_final_len] =  statiscs_matrix[0, area_final_len:vol_final_len] + calculate_volumes(final_img[min_slice:max_slice, :, :], water_array[min_slice:max_slice, :, :],
                                                                                                                                         fat_array[min_slice:max_slice, :, :],
                                                                                                                                         img_spacing, base_variables_len['Volume'],weighted=False)

                vol2_final_len=vol_final_len +base_variables_len['W_Volume']

                statiscs_matrix[0, vol_final_len:vol2_final_len] = statiscs_matrix[0,vol_final_len:vol2_final_len] + calculate_volumes(final_img[min_slice:max_slice, :, :], water_array[min_slice:max_slice, :, :],
                                                                                                                                       fat_array[min_slice:max_slice, :, :],
                                                                                                                                       img_spacing, base_variables_len['Volume'], weighted=True)
            residual=np.around(interval_steps[i+1]-int(interval_steps[i+1]),decimals=2)
            if residual !=0 :
                slice=int(np.floor(interval_steps[i+1]))

                area_stats= calculate_areas(final_img[slice, :, :], img_spacing, base_variables_len['Area'])
                volume_stats= calculate_volumes(final_img[slice, :, :], water_array[slice, :, :],fat_array[slice, :, :],img_spacing, base_variables_len['Volume'], weighted=False)
                weighted_volume_stats = calculate_volumes(final_img[slice, :, :], water_array[slice, :, :],fat_array[slice, :, :], img_spacing, base_variables_len['Volume'],weighted=True)

                area_initial_len = size_base * (i + 1)
                area_final_len = size_base * (i + 1) + base_variables_len['Area']
                areas_residual=np.ones(base_variables_len['Area'])
                areas_residual[0]=residual

                if statiscs_matrix[0,area_initial_len+1] != 0:

                    statiscs_matrix[0, area_initial_len:area_final_len] = statiscs_matrix[0,area_initial_len:area_final_len] + areas_residual * area_stats

                    statiscs_matrix[0,area_initial_len+1] = statiscs_matrix[0,area_initial_len+1] / 2
                    statiscs_matrix[0, area_initial_len + 2] = statiscs_matrix[0, area_initial_len + 2] / 2

                else:
                    statiscs_matrix[0, area_initial_len:area_final_len] = statiscs_matrix[0,area_initial_len:area_final_len] + areas_residual * area_stats


                vol_final_len = area_final_len + base_variables_len['Volume']

                statiscs_matrix[0, area_final_len:vol_final_len] = statiscs_matrix[0,area_final_len:vol_final_len] + residual * volume_stats

                vol2_final_len = vol_final_len + base_variables_len['W_Volume']

                statiscs_matrix[0, vol_final_len:vol2_final_len] = statiscs_matrix[0,vol_final_len:vol2_final_len] + residual * weighted_volume_stats



                residual_next_compartment=np.around(np.ceil(interval_steps[i+1])-interval_steps[i+1],decimals=2)


                area_initial_len = size_base * (i + 2)
                area_final_len = size_base * (i + 2) + base_variables_len['Area']
                areas_residual=np.ones(base_variables_len['Area'])
                areas_residual[0]=residual_next_compartment

                if statiscs_matrix[0, area_initial_len + 1] != 0:
                    statiscs_matrix[0, area_initial_len:area_final_len] = statiscs_matrix[0,area_initial_len:area_final_len] + areas_residual * area_stats

                    statiscs_matrix[0,area_initial_len+1] = statiscs_matrix[0,area_initial_len+1] / 2
                    statiscs_matrix[0, area_initial_len + 2] = statiscs_matrix[0, area_initial_len + 2] / 2

                else :
                    statiscs_matrix[0, area_initial_len:area_final_len] = statiscs_matrix[0,area_initial_len:area_final_len] + areas_residual * area_stats

                vol_final_len = area_final_len + base_variables_len['Volume']

                statiscs_matrix[0, area_final_len:vol_final_len] = statiscs_matrix[0,area_final_len:vol_final_len] + residual_next_compartment * volume_stats

                vol2_final_len = vol_final_len + base_variables_len['W_Volume']

                statiscs_matrix[0, vol_final_len:vol2_final_len] = statiscs_matrix[0,vol_final_len:vol2_final_len] + residual_next_compartment * weighted_volume_stats

    return statiscs_matrix


def stats_variable_initialization(nb_comparments):

    # initialize Stats Variables
    variable_columns = []

    volume_variable_columns = ['VOL_cm3', 'SAT_VOL_cm3', 'VAT_VOL_cm3', 'AAT_VOL_cm3',
                               'VAT_VOL_TO_SAT_VOL', 'VAT_VOL_TO_AAT_VOL', 'SAT_VOL_TO_AAT_VOL']

    w_volume_variable_columns= ['W_VOL_cm3','WSAT_VOL_cm3', 'WVAT_VOL_cm3',
                               'WAAT_VOL_cm3', 'WVAT_VOL_TO_WSAT_VOL', 'WVAT_VOL_TO_WAAT_VOL', 'WSAT_VOL_TO_WAAT_VOL']

    area_variable_columns = ['HEIGHT_cm', 'AVG_AREA_cm2', 'AVG_PERIMETER_cm']

    base_variable_len={}
    base_variable_len['Area']=len(area_variable_columns)
    base_variable_len['Volume']=len(volume_variable_columns)
    base_variable_len['W_Volume']=len(w_volume_variable_columns)

    roi_areas = ['wb']
    if nb_comparments != 0:
        # From Feet to Head
        for i in range(int(nb_comparments), 0, -1):
            roi_areas.append('Q' + str(i))

    for roi in roi_areas:
        for area_id in area_variable_columns:
            variable_columns.append(roi + '_' + area_id)
        for vol_id in volume_variable_columns:
            variable_columns.append(roi + '_' + vol_id)
        for w_vol_id in w_volume_variable_columns:
            variable_columns.append(roi + '_' + w_vol_id)



    variable_columns.insert(0, 'imageid')
    variable_columns.insert(1, '#_Slices')
    variable_columns.insert(2,'FLAGS')

    return variable_columns,base_variable_len


def check_image_contrast(water_array,fat_array):

    slice = fat_array.shape[0] // 2

    water_slice=water_array[slice,20:-20,20:-20]
    fat_slice=fat_array[slice,20:-20,20:-20]

    intensity_max=np.max([np.max(water_slice),np.max(fat_slice)])

    water_slice=water_slice/intensity_max
    fat_slice=fat_slice/intensity_max

    new_fat=np.zeros((fat_slice.shape[0],fat_slice.shape[1]))

    new_fat[fat_slice >= (0.10 * np.max(fat_slice))] = 2
    new_fat[fat_slice >= (0.30*np.max(fat_slice))] = 1

    border_idx=np.where(new_fat == 2)

    point_index=np.arange(0,len(border_idx[0]),10)
    point_y=border_idx[0][point_index]
    point_x=border_idx[1][point_index]

    fat_count=0
    no_fat_count=0
    for j in range(len(point_x)):

        value = fat_slice[point_y[j],point_x[j]] -water_slice[point_y[j],point_x[j]]

        if value < 0 :
            fat_count += 1
        else:
            no_fat_count += 1

    if no_fat_count > fat_count or ((no_fat_count/fat_count) > 0.75):
        FLAG='Check image contrast'
    else:
        FLAG = False

    return FLAG

def check_flags(predicted_array,water_array,fat_array,ratio_vat_sat,threshold=0.30,sat_to_vat_threshold=2.0):
    FLAG = check_image_contrast(water_array,fat_array)

    if FLAG == False:

        FLAG=extreme_AAT_increase_flag(predicted_array,threshold=threshold)

        if ratio_vat_sat > sat_to_vat_threshold:
            FLAG = 'High VAT to SAT ratio'

    return FLAG


def adipose_segmentation(fat_file, wat_file, control_images=False):
    
    import os
    from adipose_pipeline.configoptions import imgSize,run_localization,compartments,increase_threshold,sat_to_vat_threshold
    from keras import backend as K
    from adipose_pipeline.utilities import own_itk as oitk
    import numpy as np
    import pandas as pd
    from adipose_pipeline.utilities.visualization_misc import multiview_plotting
    from adipose_pipeline.adipose_pipeline import (run_adipose_localization,
                                                   run_adipose_segmentation,
                                                   calculate_statistics_v2,
                                                   clean_segmentations,
                                                   stats_variable_initialization,
                                                   check_flags)

    
    
    output_stats = 'AAT_variables_summary.json'
    output_pred = 'AAT_pred.nii.gz'
    qc_images=[]

    variable_columns, base_variable_len = stats_variable_initialization(compartments)
    ratio_position = variable_columns.index('wb_VAT_VOL_TO_SAT_VOL')

    pixel_matrix = np.zeros((1,len(variable_columns)),dtype=object)
    row_px = 0


    if not os.path.isdir(os.path.join(os.getcwd(), 'control_images')):
        os.mkdir(os.path.join(os.getcwd(), 'control_images'))

    if fat_file:
        proto_img = oitk.get_itk_image(fat_file)
        fat_array = oitk.get_itk_array(fat_file)

        if fat_array.shape[0] == imgSize[0] and fat_array.shape[1] == imgSize[1]  and fat_array.shape[2] == imgSize[2] :
            img_spacing = proto_img.GetSpacing()

            water_array=oitk.get_itk_array(wat_file)

            # Select Location
            if run_localization:
                high_idx,low_idx=run_adipose_localization(fat_array)
                K.clear_session()
            else:
                high_idx=fat_array.shape[0]
                low_idx= 0

            print('the index values are %d, %d' % (low_idx, high_idx))

            # Image Segmentation
            final_img=run_adipose_segmentation(fat_array)
            K.clear_session()

            final_img[0:low_idx,:,:]=0
            final_img[high_idx:,:,:]=0

            final_img[low_idx:high_idx, :, :] = clean_segmentations(final_img[low_idx:high_idx, :, :])

            pixel_matrix[row_px:row_px + 1, 0] = 'subjuuid'
            pixel_matrix[row_px:row_px+1,3:] = calculate_statistics_v2(final_img[low_idx:high_idx, :, :],water_array[low_idx:high_idx, :, :],fat_array[low_idx:high_idx, :, :],
                                                                     low_idx,high_idx,variable_columns[3:],
                                                                       base_variable_len,
                                                                       img_spacing,
                                                                       increase_threshold,
                                                                       compartments)

            pixel_matrix[row_px:row_px + 1, 1] = pixel_matrix[row_px:row_px + 1, 3] / (img_spacing[2] * 0.1)

            pixel_matrix[row_px:row_px + 1, 2] = check_flags(final_img[low_idx:high_idx, :, :],water_array=water_array,fat_array=fat_array,
                                                             ratio_vat_sat=pixel_matrix[row_px, ratio_position],
                                                             threshold=increase_threshold,sat_to_vat_threshold=sat_to_vat_threshold)

            df = pd.DataFrame(pixel_matrix[row_px:row_px + 1, :], columns=variable_columns)

            df.to_csv('AAT_variables_summary.csv', sep='\t', index=False)
            
            df.to_json(output_stats, orient='records')

            row_px += 1
            # Save prediction

            predict_image = oitk.make_itk_image(final_img, proto_image=proto_img)
            oitk.write_itk_image(predict_image, output_pred)

            # Modified images for display
            disp_fat = np.flipud(fat_array[:])
            disp_fat = np.fliplr(disp_fat[:])
            disp_pre = np.flipud(final_img[:])
            disp_pre = np.fliplr(disp_pre)
            idx = (np.where(disp_pre > 0))
            low_idx = np.min(idx[0])
            high_idx = np.max(idx[0])

            interval = (high_idx - low_idx) // 4

            # Control images of the segmentation
            if control_images:
                for i in range(4):
                    control_point = [0, int(np.ceil(disp_fat.shape[1] / 2)), int(np.ceil(disp_fat.shape[2] / 2))]
                    control_point[0] = int(np.ceil(np.random.uniform(high_idx - interval * i, high_idx - interval * ((i + 1)))))
                    qc_image = os.path.join(os.getcwd(), 'control_images/QC_%s.png'%i)
                    multiview_plotting(disp_fat, disp_pre, control_point, qc_image, classes=5, alpha=0.5, nbviews=3)
                    qc_images.append(os.path.abspath(qc_image))

            print('-' * 30)
        else:
            print('Subject has different dimension than 72,224,256')
            print('-' * 30)

    else:
        print('Subject doesnt have a Fat Image')
        print('-' * 30)


    print('-' * 30)#K.clear_session()

    print('Finish Segmentation.')

    print('-' * 30)
        

    return os.path.abspath(output_stats), os.path.abspath(output_pred), qc_images

    
    
def adipose_pipeline(scans_dir, work_dir, outputdir, subject_ids, 
                     num_threads, device, cts=False,  name='wmhs_preproc'):
    
    adiposewf = pe.Workflow(name=name)
    adiposewf.base_dir = work_dir

    inputnode = pe.Node(interface=IdentityInterface(fields=['subject_ids', 'outputdir']), name='inputnode')
    inputnode.iterables = [('subject_ids', subject_ids)]
    inputnode.inputs.subject_ids = subject_ids
    inputnode.inputs.outputdir = outputdir

    #template for input files
    templates = {"FATF": "{subject_id}/*FatImaging_F.nii.gz",
                 "FATW": "{subject_id}/*FatImaging_W.nii.gz"
                 }
                 
    fileselector = pe.Node(SelectFiles(templates), name='fileselect')
    fileselector.inputs.base_directory = scans_dir


    #%% step-1 segment images
    
    adipose_seg = pe.Node(interface=util.Function(input_names=['fat_file','wat_file','control_images'],
                                                  output_names=['output_stats','output_pred','qc_images'],
                                                  function=adipose_segmentation),name='adipose_seg')
    adipose_seg.n_procs = num_threads
    adipose_seg.inputs.control_images=control_images


    #%% collect outputs
    datasinkout = pe.Node(interface=DataSink(), name='datasinkout')
    datasinkout.inputs.parameterization=False

    # %% workflow connections
    
    #step 1
    adiposewf.connect(inputnode        , 'subject_ids',      fileselector,'subject_id')
    adiposewf.connect(fileselector     , 'FATF',             adipose_seg, 'fat_file')
    adiposewf.connect(fileselector     , 'FATW',             adipose_seg, 'wat_file')
    
               
    # outputs
    adiposewf.connect(inputnode        , 'subject_ids',     datasinkout, 'container')
    adiposewf.connect(inputnode        , 'outputdir',       datasinkout, 'base_directory')


    adiposewf.connect(adipose_seg      , 'qc_images',     datasinkout,'QC.@qc_images')
    adiposewf.connect(adipose_seg      , 'output_pred',   datasinkout,'Segmentations.@output_pred')
    adiposewf.connect(adipose_seg      , 'output_stats',  datasinkout,'Segmentations.@output_stats')

    
    return adiposewf
    

