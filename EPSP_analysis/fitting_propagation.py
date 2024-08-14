# A pyqt-based GUI for selecting images
# The GUI first read a list of pickle files
# Each pickle file contains a list of dictionaries
# The GUI generates plots based on the dictionaries, and allow the user to select the plots
# Once the plot is selected, it is saved to a list
# After selecting all the plots, the user can click a button, all the dictionaries whose plot gets saved to the list will be preserved
# That's all for now. I will add more features later.

import sys
import os
import glob
import pickle
from functools import partial

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


def mini_prop_func(x, L, a, x0):
    return a * np.exp((x-x0)/L)


# class for the main window
class Event_selector_GUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Event Selector')
        self.setGeometry(100, 100, 1500, 600)
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        
        # create a label for displaying the image
        self.QC_fig, self.QC_axes = plt.subplots(1, 4)
        # change the fontsize of the axes
        plt.rcParams.update({'font.size': 5})
        # add padding to subplots and preserve titles
        plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
        self.QC_canvas = FigureCanvas(self.QC_fig)
        self.layout.addWidget(self.QC_canvas)

        
        # create a button for selecting the image
        self.select_button = QtWidgets.QPushButton('Select Image Folder')
        self.select_button.clicked.connect(self.select_image_folder)
        self.layout.addWidget(self.select_button)
        
        # create a button for saving the image
        self.save_button = QtWidgets.QPushButton('Save Image')
        self.save_button.clicked.connect(self.save_image)
        self.layout.addWidget(self.save_button)
        
        # create a button for discarding the image
        self.discard_button = QtWidgets.QPushButton('Discard Image')
        self.discard_button.clicked.connect(self.discard_image)
        self.layout.addWidget(self.discard_button)

        # create a button for going to the next image
        self.next_button = QtWidgets.QPushButton('Next Image')
        self.next_button.clicked.connect(self.next_image)
        self.layout.addWidget(self.next_button)

        # create a button for going back
        self.back_button = QtWidgets.QPushButton('Back')
        self.back_button.clicked.connect(self.previous_image)
        self.layout.addWidget(self.back_button)
        
        # create a button for discarding the selected images
        self.discard_selected_button = QtWidgets.QPushButton('Discard Selected Images')
        self.discard_selected_button.clicked.connect(self.discard_selected_images)
        self.layout.addWidget(self.discard_selected_button)
        
        # create a button for quitting the program
        self.quit_button = QtWidgets.QPushButton('Quit')
        self.quit_button.clicked.connect(self.quit_program)
        self.layout.addWidget(self.quit_button)

        self.reset_pkl_vars()
        
    def reset_pkl_vars(self):
        # create a list to store the image paths
        self.pkl_paths = []
        self.propogation_events = []
        
        # create a list to store the image indices
        self.image_indices = []

        # create a list of booleans to store the choice of the images
        self.discard_or_not = []
        
        # create a list to store the image indices
        self.current_image_index = 0
        
    def select_image_folder(self):
        '''select a folder and get all the png images in the folder'''
        self.reset_pkl_vars()
        self.image_folder = QFileDialog.getExistingDirectory()
        self.pkl_paths = glob.glob(os.path.join(self.image_folder, '*.pkl'))
        for pkl in self.pkl_paths:
            with open(pkl, 'rb') as f:
                propogation_events_loaded = pickle.load(f)
                self.propogation_events += propogation_events_loaded
        self.image_indices = list(range(len(self.propogation_events)))
        self.discard_or_not = [False]*len(self.propogation_events)
        self.image_amount = len(self.propogation_events)
        self.current_image_index = 0
        if len(self.pkl_paths) == 0:
            QMessageBox.warning(self, 'Warning', 'No images found in the folder')
            return
        self.display_image()

    def display_image(self):
        # display the image
        pe = self.propogation_events[self.current_image_index]
        for ax in self.QC_axes:
            ax.clear()
        self.QC_axes[0].axis('off')
        self.QC_axes[0].imshow(pe['neuron_img'], cmap='gray')
        self.QC_axes[0].scatter(pe['soma_mask'][0], pe['soma_mask'][1], s=1, c='b')
        (x, y) = pe['all_pixels_of_propagation']
        (xp, yp) = pe['initiation_site']
        DOI_vertices = pe['dendritic_path']
        distance_to_soma = pe['initiation_distance_to_soma'] 
        self.QC_axes[0].scatter(x, y, s=2, c='g')
        self.QC_axes[0].scatter(xp, yp, s=3, c='r')
        self.QC_axes[0].plot(DOI_vertices[:, 0], DOI_vertices[:, 1], c='b', linewidth=1)
        self.QC_axes[0].set_title('DOI distance to soma: '+str(np.round(distance_to_soma)))
        self.QC_axes[1].plot(pe['initiation_waveform'], linewidth=1)
        self.QC_axes[1].set_title('peak amplitude: '+str(np.round(pe['initiation_peak_amplitude'], 2)))


        self.QC_axes[2].plot(pe['soma_waveform'], linewidth=1)
        # print the correlation coefficient to 4 decimal places
        self.QC_axes[2].set_title('soma-dendrite correlation')

        distances = pe['distances']
        peak_amplitude = pe['peak_amplitudes']
        initiation_distance = pe['initiation_distance_to_soma']
        # keep the points that are closer to the soma than the initiation distance
        peak_amplitude = peak_amplitude[distances <= initiation_distance]
        distances = distances[distances <= initiation_distance]
        # add the initial point
        distances = np.insert(distances, 0, 0)
        peak_amplitude = np.insert(peak_amplitude, 0, pe['soma_peak_amplitude'])
        # fit the curve
        mini_prop_func_partial = partial(mini_prop_func, x0=initiation_distance)
        try:
            popt, pcov = curve_fit(mini_prop_func_partial, distances, peak_amplitude, p0=[10, 0.2], bounds=(0, [100, 1]))
            r = np.linspace(0, distances.max(), 100)
            famp = mini_prop_func_partial(r, *popt)
            self.QC_axes[3].plot(r, famp, linewidth=1, c='orange')
            self.QC_axes[3].set_title('propagation_length: '+str(np.round(popt[0], 2)))
        except:
            self.QC_axes[3].set_title('fitting failed')
        self.QC_axes[3].scatter(distances, peak_amplitude, s=2)
        self.QC_axes[3].scatter(0.2, pe['soma_peak_amplitude'], s=2, c='r')
        self.QC_axes[3].scatter(pe['initiation_distance_to_soma'], pe['initiation_peak_amplitude'], s=10, facecolors='none', edgecolors='r', linewidth=0.5)
        self.QC_axes[3].set_xlim([0, np.max(distances)+5])
        self.QC_axes[3].set_ylim([0, np.max(peak_amplitude)+0.05])
        #self.QC_axes[3].set_xlabel('distance to soma')
        #self.QC_axes[3].set_ylabel('peak amplitude')
        plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
        self.QC_canvas.draw()

    def next_image(self):
        # go to the next image
        if self.current_image_index < self.image_amount - 1:
            self.current_image_index += 1
            self.display_image()
        else:
            QMessageBox.warning(self, 'Warning', 'No more images!')
            return
    
    def previous_image(self):
        # go back to the previous image
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_image()
        else:
            QMessageBox.warning(self, 'Warning', 'No more images!')
            return

    def save_image(self):
        # save the image to the selected images list
        self.discard_or_not[self.current_image_index] = False
        if self.current_image_index < self.image_amount - 1:
            self.current_image_index += 1
            self.display_image()
        else:
            QMessageBox.warning(self, 'Warning', 'Selection finished!')
            return
    
    def discard_image(self):
        # discard the image to the discarded images list
        self.discard_or_not[self.current_image_index] = True
        if self.current_image_index < self.image_amount - 1:
            self.current_image_index += 1
            self.display_image()
        else:
            QMessageBox.warning(self, 'Warning', 'Selection finished!')
            return
    
    def discard_selected_images(self):
        # For all the chosen plot, save the list of the corresponding dictionaries to a new pickle file
        saved_events = []
        archived_events = []
        for i in self.image_indices:
            propogation_event = self.propogation_events[i]
            if not self.discard_or_not[i]:
                saved_events.append(propogation_event)
            else:
                archived_events.append(propogation_event)

        save_dir = os.path.join(self.image_folder, 'summary')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'QC_passed.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(saved_events, f)
        # save the discarded events
        save_path = os.path.join(save_dir, 'QC_failed.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(archived_events, f)


    def quit_program(self):
        # quit the program
        sys.exit()


# main function
def main():
    app = QtWidgets.QApplication(sys.argv)
    event_selector = Event_selector_GUI()
    event_selector.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()