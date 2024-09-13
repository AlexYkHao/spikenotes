import os
import sys
import glob
import pickle
from functools import partial
from multiprocessing.pool import Pool

import numpy as np
from skimage import io
from skimage.filters import threshold_otsu
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
#from scipy.ndimage import gaussian_filter

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs # nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


class MiniDetector(object):
    def __init__(self, image_path, save_dir) -> None:
        self.img_path = image_path
        self.save_dir = save_dir
        self.binned_img = None
        self.mask = None
        self.soma_location = None
        self.save_base_name = os.path.basename(self.img_path).split('.')[0]
        self.filtered_traces = None
        self.time_vector = None 
        self.mini_info_list = []

    def bin_image(self, bin_size=8):
        # first check if the binned image already exists in npy file
        npy_file = os.path.join(os.path.dirname(self.img_path), self.save_base_name+'_binned_img.npy')
        if os.path.exists(npy_file):
            self.binned_img = np.load(npy_file)
        else:
            self.create_binned_img(bin_size)

    def create_binned_img(self, bin_size=8):
        img = io.imread(self.img_path).astype('int16')
        # permute image in (x,y,t) order
        img = np.transpose(img, (2,1,0))

        binned_img = np.zeros((img.shape[0]//bin_size, img.shape[1]//bin_size, img.shape[2]))
        for i in range(binned_img.shape[0]):
            for j in range(binned_img.shape[1]):
                binned_img[i,j] = np.mean(img[i*bin_size:(i+1)*bin_size, j*bin_size:(j+1)*bin_size], axis=(0,1))
        del img
        self.binned_img = binned_img
        # save the binned image as a npy file
        npy_file = os.path.join(os.path.dirname(self.img_path), self.save_base_name+'_binned_img.npy')
        np.save(npy_file, binned_img)

    def draw_mask(self):
        mean_img = self.binned_img.mean(2)
        thresh1 = threshold_otsu(mean_img)  # otsu method minimizes intra-class variance
        neuron_mask_1 = mean_img < thresh1  # the iteration here is assuming there are 3 classes, soma, dendrite, background
        img_2 = mean_img[neuron_mask_1]
        thresh = threshold_otsu(img_2)
        neuron_mask = mean_img > thresh
        self.mask = neuron_mask

    def determine_soma_location(self):
        # determine the soma location by finding the center of the neuron mask determined by otsu method
        mean_img = self.binned_img.mean(2)
        thresh = threshold_otsu(mean_img) 
        neuron_mask = mean_img > thresh  # this determines the soma mask
        x = np.where(neuron_mask)[0]
        y = np.where(neuron_mask)[1]
        x_center = np.mean(x)
        y_center = np.mean(y)
        self.soma_location = (x_center, y_center)

    def mask_sanity_check(self):
        mean_img = self.binned_img.mean(2)
        mask = self.mask
        # show both mean_img and mask in two plots and save as one image file
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(mean_img, cmap='gray')
        axs[1].imshow(mask, cmap='gray')
        fig.savefig(os.path.join(self.save_dir, self.save_base_name+'_mask.png'))

    def pre_processing(self, fs=200, cutoff=2, order=5):
        self.bin_image()
        self.draw_mask()
        self.mask_sanity_check()
        assert self.mask.sum() > 0, 'mask is empty, please check the mask'
        neuron_traces = self.binned_img[self.mask]
        neuron_traces = neuron_traces[:, int(fs):] # remove the first 1s of data
        # take the 10th percentile of the mean binned_img as the baseline
        background = np.percentile(self.binned_img.mean(2), 10)
        neuron_traces = neuron_traces - background

        neuron_traces_filt  = butter_lowpass_filter(neuron_traces, cutoff, fs, order)
        dff = (neuron_traces - neuron_traces_filt)/neuron_traces_filt
        self.time_vector = np.arange(neuron_traces.shape[1])/fs
        self.filtered_traces = dff
                

    def find_mini_events(self, threshold=3.5, distance=10):
        """
        threshold: the number of sigmas of the peak above the baseline
        """
        dff = self.filtered_traces
        dff_std = dff.std(1)
        mini_info_list = []
        for trace_id in range(dff.shape[0]):
            peaks, _ = find_peaks(-dff[trace_id], height=dff_std[trace_id]*threshold, distance=distance)
            # convert trace_id to x,y coordinates
            x = np.where(self.mask)[0][trace_id]
            y = np.where(self.mask)[1][trace_id]
            if len(peaks) > 0:
                mini_info_list.append({'id':trace_id, 'x':x, 'y':y, 'peaks':peaks, 'dff':dff[trace_id]})
        self.mini_info_list = mini_info_list

    def save_data(self):
        # save the mini_info_list as a pickle file
        with open(os.path.join(self.save_dir, self.save_base_name+'_mini_info_list.pkl'), 'wb') as f:
            pickle.dump(self.mini_info_list, f)
            pickle.dump(self.binned_img, f)
            pickle.dump(self.mask, f)


def process_image(img_path, save_dir):
    print('Processing {}'.format(img_path))
    mini_detector = MiniDetector(img_path, save_dir)
    mini_detector.pre_processing(fs=200, cutoff=0.5, order=5)
    mini_detector.find_mini_events(threshold=3.5)
    mini_detector.save_data()


# if __name__ == '__main__':
#     # take arguments from command line
#     img_dir = sys.argv[1]
#     #img_dir = '/Volumes/CLab/dendritic_scaling/test/'
#     print(img_dir)
#     img_paths = glob.glob(os.path.join(img_dir, '*.tif'))
#     save_dir = os.path.join(img_dir, 'results')
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     process_image2 = partial(process_image, save_dir=save_dir)
#     p = Pool(processes=4)
#     p.map(process_image2, img_paths)
    
if __name__ == '__main__':
    # image_dir_list = [
    # '/Volumes/CLab/dendritic_scaling/20231215_B33_neurons_DIV24_dendritic_scaling_2/B2-470/B33-DIV24-B2-470-TIFF/TTX',
    # '/Volumes/CLab/dendritic_scaling/20231215_B33_neurons_DIV24_dendritic_scaling_2/B3-470/B33-DIV24-B3-470-TIFF/TTX',
    # '/Volumes/CLab/dendritic_scaling/20231215_B33_neurons_DIV24_dendritic_scaling_2/C2-482/B33-DIV24-C2-482-TIFF/TTX',
    # '/Volumes/CLab/dendritic_scaling/20231215_B33_neurons_DIV24_dendritic_scaling_2/C3-482/B33-DIV24-C3-482-TIFF/TTX',
    # '/Volumes/CLab/dendritic_scaling/20231212_B33-neurons-DIV21_dendritic-scaling/Well2/B33-DIV21-corti-glassbottom-well2-TIFF/TTX',
    # '/Volumes/CLab/dendritic_scaling/20231212_B33-neurons-DIV21_dendritic-scaling/Well3/B33-DIV21-corti-glassbottom-well3-TIFF/TTX',
    # '/Volumes/CLab/dendritic_scaling/20231212_B33-neurons-DIV21_dendritic-scaling/Well4/B33-DIV21-corti-glassbottom-well4-TIFF/TTX']
    image_dir_list = [
    '/Volumes/MyPassport/dendritic_scaling/20240209/C1_wellB2/TTX_PTX/',
    '/Volumes/MyPassport/dendritic_scaling/20240209/C1_wellC2/TTX_PTX/',
    '/Volumes/MyPassport/dendritic_scaling/20240209/C1_wellC3/TTX_PTX/',
    '/Volumes/MyPassport/dendritic_scaling/20240210/plateC2_WellB2/TTX_PTX/',
    '/Volumes/MyPassport/dendritic_scaling/20240210/plateC2_WellB3/TTX_PTX/',
    '/Volumes/MyPassport/dendritic_scaling/20240210/plateC2_WellC2/TTX_PTX/',
    '/Volumes/MyPassport/dendritic_scaling/20240210/plateC2_WellC3/TTX_PTX/',
    '/Volumes/MyPassport/dendritic_scaling/20240210/plateC2_WellB2/W-view-images/TTX_PTX/']
    # image_dir_list = [
    # '/Volumes/MyPassport/dendritic_scaling/20240215/plateC3_WellB2/TTX_PTX/',
    # '/Volumes/MyPassport/dendritic_scaling/20240215/plateC3_WellB3/TTX_PTX/',
    # '/Volumes/MyPassport/dendritic_scaling/20240215/plateC3_WellB4/TTX_PTX/',
    # '/Volumes/MyPassport/dendritic_scaling/20240215/plateC3_WellC2/TTX_PTX/',
    # '/Volumes/MyPassport/dendritic_scaling/20240215/plateC3_WellC3/TTX_PTX/']
    # image_dir_list = [
    # '/Volumes/MyPassport/dendritic_scaling/20240216/plateC2_WellB2/',
    # '/Volumes/MyPassport/dendritic_scaling/20240216/plateC2_WellB3/',
    # '/Volumes/MyPassport/dendritic_scaling/20240216/plateC2_WellC2/',
    # '/Volumes/MyPassport/dendritic_scaling/20240216/plateC2_WellC3/']
    
    # image_dir_list = ['/Users/ykhao/Downloads/mini_analysis/figure5A_neuron_example_nonTTX/']

    for img_dir in image_dir_list:
        print(img_dir)
        img_paths = glob.glob(os.path.join(img_dir, '*.tif'))
        save_dir = os.path.join(img_dir, 'results0p5Hz')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        process_image2 = partial(process_image, save_dir=save_dir)
        p = Pool(processes=4)
        p.map(process_image2, img_paths)