import javabridge
import bioformats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.integrate import simps
from PIL import Image
import matplotlib.patches as patches
from skimage import io, transform

javabridge.start_vm(class_path=bioformats.JARS)


class CompareDist:
    """ 
    For analyzing spectrum of image data
    """
    def __init__(self, raw_data_filename_list, remove_cell_list, mask_filename_list, max_cell_num, bin_size, Num_pixels, Num_CH, zstack_start_list, zstack_end_list, organelle1, organelle2, start_nm):
        self.raw_data_filename_list = raw_data_filename_list
        self.remove_cell_list = remove_cell_list
        self.mask_filename_list = mask_filename_list
        self.max_cell_num = max_cell_num
        self.bin_size = bin_size
        self.Num_pixels = Num_pixels
        self.Num_CH = Num_CH
        self.zstack_start_list = zstack_start_list
        self.zstack_end_list = zstack_end_list
        self.organelle1 = organelle1  # 1st CH in unmixed image
        self.organelle2 = organelle2  # 2nd CH in unmixed image
        self.start_nm = start_nm

        self.dict = {}

        i = 0
        for raw_data_filename in self.raw_data_filename_list:
            self.dict[f'raw_data_{i}'] = []
            for zslice in range(self.zstack_start_list[i], self.zstack_end_list[i]+1):
                # Load the zstack slice from the raw nd2 file
                Raw_data = bioformats.load_image(
                    path=f'{raw_data_filename}.nd2',
                    c=None,
                    z=zslice-1,
                    rescale=False)
                self.dict[f'raw_data_{i}'].append(Raw_data)

            self.dict[f'raw_data_{i}'] = np.asarray(self.dict[f'raw_data_{i}'])
            i += 1   

        print(self.dict['raw_data_0'].shape)             

        del i
    
    # ------------------------------------------------------------------------
    def pix_aligned_percell_seg_mask(self,cell_num,mask_filename):
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
        mask[mask != cell_num] = 0
        mask = transform.rotate(mask, 90)
        mask = transform.warp(mask, similarity_transform)
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        mask_indx = np.where(mask > 0) 
        return mask_indx
    
    # ------------------------------------------------------------------------
    def rand_FOV_cell(self):
        """
        Randomly select a cell number on a randomly selected FOV
        """
        FOV_num = random.randint(0, len(self.raw_data_filename_list) - 1)
        
        # Keep selecting a cell number until it's not in the remove_cell_list
        while True:
            cell_num = random.randint(1, self.max_cell_num[FOV_num])
            if cell_num not in self.remove_cell_list[FOV_num]:
                break
        
        return FOV_num, cell_num
    
    # ------------------------------------------------------------------------
    def rand_FOV_cell_list(self, Num_cellsamples):
        """
        Create list with non-repeating, randomly selected cell numbers on randomly selected FOVs
        """
        FOV_num_list = np.zeros(Num_cellsamples)
        cell_num_list = np.zeros(Num_cellsamples)

        selected_pairs = set()

        for i in range(Num_cellsamples):
            while True:
                FOV_num, cell_num = self.rand_FOV_cell()
                if (FOV_num, cell_num) not in selected_pairs:
                    selected_pairs.add((FOV_num, cell_num))
                    FOV_num_list[i] = FOV_num
                    cell_num_list[i] = cell_num
                    break
        # for i in range(Num_cellsamples):
        #     FOV_num, cell_num = self.rand_FOV_cell()
        #     FOV_num_list[i] = FOV_num
        #     cell_num_list[i] = cell_num

        # Convert to array
        FOV_num_list = FOV_num_list.astype(int)
        cell_num_list = cell_num_list.astype(int)
        
        return FOV_num_list, cell_num_list

    # ------------------------------------------------------------------------
    def calc_peaks_mean_percell(self, FOV_num, cell_num, min1, max1, min2, max2):
        """
        Calculate mean intensity of two regions of determined peaks of the two spectra per cell
        """
        maskpix = self.pix_aligned_percell_seg_mask(cell_num, self.mask_filename_list[FOV_num])
        
        y_coord = maskpix[0]
        x_coord = maskpix[1]


        self.dict[f'raw_data_{FOV_num}']  

        # Create list to store the mean intensity of the two peaks
        peak1 = []
        peak2 = []

        # Iterate over the zstack slices
        for zslice in range(self.zstack_end_list[FOV_num]+1 - self.zstack_start_list[FOV_num]):
            for x, y in zip(x_coord, y_coord):
                # Get the values of the two peaks
                values = self.dict[f'raw_data_{FOV_num}'][zslice, y, x, :]
                # Calculate the mean intensity of the two peaks
                peak1.append(np.mean(values[min1:max1]))  # ER peak
                peak2.append(np.mean(values[min2:max2]))  # M peak

        peak1 = np.asarray(peak1)
        peak2 = np.asarray(peak2)  
        
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
    
    # ------------------------------------------------------------------------
    def generate_scatter_randcells(self,min1,max1,min2,max2,Num_cellsamples):
        """ 
       
        """
        FOV_list, cell_list = self.rand_FOV_cell_list(Num_cellsamples)
        print(FOV_list)
        print(cell_list)

        ERpeak_cell_list = []
        Mpeak_cell_list = []

        for FOV_num, cell_num in zip(FOV_list, cell_list):
            ERpeak_mix, Mpeak_mix = self.calc_peaks_mean_percell(FOV_num,cell_num,min1,max1,min2,max2)
            ERpeak_cell_list.append(ERpeak_mix)
            Mpeak_cell_list.append(Mpeak_mix)

        ERpeak = np.concatenate(ERpeak_cell_list)
        Mpeak = np.concatenate(Mpeak_cell_list)
 
        self.generate_peaks_joint_plot(ERpeak, Mpeak, min1, max1, min2, max2)

        self.generate_peaks_hist2D_plot(ERpeak, Mpeak, min1, max1, min2, max2)
        plt.show()


