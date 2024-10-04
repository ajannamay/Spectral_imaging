import javabridge
import bioformats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import simps
from PIL import Image
import matplotlib.patches as patches
from skimage import io, transform

javabridge.start_vm(class_path=bioformats.JARS)


class Spectra2:
    """ 
    For analyzing spectrum of image data
    """
    def __init__(self, raw_data_filename, unmixed_filename, bin_size, Num_pixels, Num_CH, zstack_start, zstack_end, organelle1, organelle2, start_nm):
        self.raw_data_filename = raw_data_filename
        self.unmixed_filename = unmixed_filename
        self.bin_size = bin_size
        self.Num_pixels = Num_pixels
        self.Num_CH = Num_CH
        self.zstack_start = zstack_start
        self.zstack_end = zstack_end
        self.organelle1 = organelle1  # 1st CH in unmixed image
        self.organelle2 = organelle2  # 2nd CH in unmixed image
        self.start_nm = start_nm

    # ------------------------------------------------------------------------
    def load_zslice(self, zstack_ind):
        """
        Load the zstack slice from the raw nd2 file
        """
        # Load raw data
        Raw_data = bioformats.load_image(
            path=f'{self.raw_data_filename}.nd2',
            c=None,
            z=zstack_ind-1,
            rescale=False)
        
        return Raw_data
        
    # ------------------------------------------------------------------------
    def pix_aligned_cell_list_seg_mask(self,cell_list_remove,mask_filename):
        """ 
        Obtain pix from mask created in ImageJ
        """
        # Define the similarity transformation matrix
        matrix_similarity = np.array(
            [[ 0.90625,  0.0390625,  16.0],
            [-0.0390625,  0.90625,  84.0],
            [0.000000,  0.000000,  1.000000]])
        similarity_transform = transform.SimilarityTransform(matrix=matrix_similarity)
        mask = io.imread(f'{mask_filename}.tif')
        mask[np.isin(mask, cell_list_remove)] = 0 # remove cells in cell_list_remove
        mask = transform.rotate(mask, 90)
        mask = transform.warp(mask, similarity_transform)
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        mask_indx = np.where(mask > 0) 
        return mask_indx  
    
    # ------------------------------------------------------------------------
    def calc_peaks_mean(self,zslice,min1,max1,min2,max2,withMask,maskpix):
        """ 
        Calculate mean intensity of two regions of determined peaks of the two spectra
        """
        if withMask == True:
            y_coord = maskpix[0]
            x_coord = maskpix[1]

            Raw_data = self.load_zslice(zslice)

            # Create a 2D array with same shape as the raw data
            peak1 = np.zeros(Raw_data[:,:,0].shape)
            peak2 = np.zeros(Raw_data[:,:,0].shape)

            for x, y in zip(x_coord, y_coord):
                # Get the values of the two peaks
                values = Raw_data[y, x, :]
                # Calculate the mean intensity of the two peaks
                peak1[y, x] = np.mean(values[min1:max1])  # ER peak
                peak2[y, x] = np.mean(values[min2:max2])  # M peak

            # Flatten the arrays
            peak2 = peak2.flatten()
            peak1 = peak1.flatten()
 
        else:
            Raw_data = self.load_zslice(zslice)
            # Create a 2D array with same shape as the raw data
            peak1 = np.zeros(Raw_data[:,:,0].shape)
            peak2 = np.zeros(Raw_data[:,:,0].shape)
            # Iterate over the rows of the array
            for i in range(Raw_data.shape[0]):
                # Iterate over the columns of the array
                for j in range(Raw_data.shape[1]):
                    # Get the values of the two peaks
                    values = Raw_data[i, j, :]
                    # Calculate the mean intensity of the two peaks
                    peak1[i, j] = np.mean(values[min1:max1])  # ER peak
                    peak2[i, j] = np.mean(values[min2:max2])  # M peak

            # Flatten the arrays
            peak2 = peak2.flatten()
            peak1 = peak1.flatten()

        return peak1, peak2
    
    # ------------------------------------------------------------------------
    def generate_peaks_hist2D_plot(self,peak1,peak2,min1,max1,min2,max2):
        """ 
        Generate scatter plot of mean intensity of two regions of determined peaks of the two spectra
        """
        # Flatten the arrays
        peak2 = peak2.flatten()
        peak1 = peak1.flatten()
        # Find indices of points that are non-zero
        indices = np.where(np.logical_and(peak1 > 0, peak2 > 0))
        print(len(indices[0]))
        # Create a contour plot to map the density of the points
        # plt.hexbin(peak2[indices], peak1[indices], gridsize=100, bins='log', cmap='inferno') # bins='log',
        # fig = plt.figure(figsize=(5,5))
        # plt.scatter(peak2, peak1, alpha=0.3, marker='.')
        # sns.kdeplot(x = peak2[indices], y = peak1[indices], color='black',alpha=0.5, levels=7, linewidths=1.5)
        sns.jointplot(x = peak2[indices], y = peak1[indices], kind="hex", gridsize=100, bins='log', cmap='inferno')
        # plt.colorbar()
        plt.ylim(bottom=0, top=1500)
        plt.xlim(left=0, right=1500)
        plt.ylabel(f'Mean intensity of ER peak (CHs {min1+1}-{max1+1})')
        plt.xlabel(f'Mean intensity of M peak (CHs {min2+1}-{max2+1})')

        return plt
    
    # ------------------------------------------------------------------------
    def generate_peaks_joint_plot(self,peak1,peak2,min1,max1,min2,max2):
        """ 
        Generate scatter plot of mean intensity of two regions of determined peaks of the two spectra
        """
        # Find indices of points that are non-zero
        indices = np.where(np.logical_and(peak1 > 0, peak2 > 0))

        # Generate the joint plot
        joint_plot = sns.jointplot(x = peak2[indices], y = peak1[indices], kind="scatter", color='black', alpha=0.1)
        joint_plot.ax_joint.set_xlim(0,1500)
        joint_plot.ax_joint.set_ylim(0,1500)
        joint_plot.set_axis_labels(f'Mean intensity of ER peak (CHs {min1+1}-{max1+1})',
                                   f'Mean intensity of M peak (CHs {min2+1}-{max2+1})')
        return joint_plot

    # ----------------------------------------------------------------------
    def generate_scatter_cell_list(self,cell_list,mask_filename,min1,max1,min2,max2):
        """ 
       
        """
        ERpeak_mix_zslice = []
        Mpeak_mix_zslice = []
        cellspix = self.pix_aligned_cell_list_seg_mask(cell_list,mask_filename)
        for zslice in range(self.zstack_start, self.zstack_end+1):
            ERpeak_mix, Mpeak_mix = self.calc_peaks_mean(zslice,min1,max1,min2,max2,True,cellspix)
            ERpeak_mix_zslice.append(ERpeak_mix)
            Mpeak_mix_zslice.append(Mpeak_mix)

        ERpeak = np.concatenate(ERpeak_mix_zslice)
        Mpeak = np.concatenate(Mpeak_mix_zslice)
 
        self.generate_peaks_joint_plot(ERpeak, Mpeak, min1, max1, min2, max2)

        self.generate_peaks_hist2D_plot(ERpeak, Mpeak, min1, max1, min2, max2)
        plt.show()

