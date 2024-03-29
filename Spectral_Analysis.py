
import javabridge
import bioformats
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from PIL import Image
import matplotlib.patches as patches

javabridge.start_vm(class_path=bioformats.JARS)

class Spectra:
    """ 
    For analyzing spectrum of image data
    """
    def __init__(self, raw_data_filename, unmixed_filename, bin_size, Num_pixels, Num_CH, zstack_ind, organelle1, organelle2, start_nm):
        self.raw_data_filename = raw_data_filename
        self.unmixed_filename = unmixed_filename
        self.bin_size = bin_size
        self.Num_pixels = Num_pixels
        self.Num_CH = Num_CH
        self.zstack_ind = zstack_ind
        self.organelle1 = organelle1  # 1st CH in unmixed image
        self.organelle2 = organelle2  # 2nd CH in unmixed image
        self.start_nm = start_nm

        if self.zstack_ind == 0:
            # Load raw data
            self.Raw_data = bioformats.load_image(
                path=f'{self.raw_data_filename}.nd2',
                c=None,
                rescale=False)
            # Load unmixed data (if needed)
            if self.unmixed_filename == False:
                pass
            else:
                self.Unmixed_data = bioformats.load_image(
                    path=f'{self.unmixed_filename}.nd2',
                    c=None,
                    rescale=False)
                # Specify array of organelle
                self.arr_organelle1 = self.Unmixed_data[ :, :, 0]
                self.arr_organelle2 = self.Unmixed_data[ :, :, 1]
        else:
            # Load raw data
            self.Raw_data = bioformats.load_image(
                path=f'{self.raw_data_filename}.nd2',
                c=None,
                z=zstack_ind-1,
                rescale=False)
            # Load unmixed data (if needed)
            if self.unmixed_filename == False:
                pass
            else:
                self.Unmixed_data = bioformats.load_image(
                    path=f'{self.unmixed_filename}.nd2',
                    c=None,
                    z=zstack_ind-1,
                    rescale=False)
                # Specify array of organelle
                self.arr_organelle1 = self.Unmixed_data[ :, :, 0]
                self.arr_organelle2 = self.Unmixed_data[ :, :, 1]
                
        # x-axis for the spectrum later on
        self.xaxis = [f'{a+1} ({self.start_nm + self.bin_size * a})' for a in range(self.Num_CH)]
        self.xvalues = np.array([a+1 for a in range(self.Num_CH)])
        self.xvalues_nm = np.array([self.start_nm + self.bin_size * a for a in range(self.Num_CH)])
    
    # ------------------------------------------------------------------------
    def pix_mask_ImageJ(self,mask_filename):
        """ 
        Obtain pix from mask created in ImageJ
        """
        mask = Image.open(f'{mask_filename}.tif')
        mask = np.asarray(mask)
        mask_indx = np.where(mask > 0)
        return mask_indx
        
    # ------------------------------------------------------------------------
    def generate_scatter_plot(self, saturated_pix, Num_toppixels):
        """ 
        Generate scatter plot from unmixed data
        """
        def select_pix(array1, array2):
            # Create a boolean mask where the condition is true for elements in array2 equal to 0
            mask = (array2 == 0)
            # Use the boolean mask to index array1, extracting values where the corresponding element in array2 is 0
            result = array1[mask]
            # Sort the result array in increasing order
            sorted_result = np.sort(result)
            # Ignore saturated pixels
            filtered = sorted_result[np.logical_and(100 <= sorted_result, sorted_result <= 4050)]
            # Get the top pixels from the sorted result w/o saturated pixels
            top_numbers = filtered[-Num_toppixels:]
            return top_numbers
        
        def indx_select_pix(array1, array2, select_pix_array1):
            condition = np.logical_and(np.logical_and(array2 == 0, array1 >= np.min(select_pix_array1)), array1 <= np.max(select_pix_array1))
            indices = np.where(condition)
            return indices
        
        if saturated_pix == False:
            # Location of saturated pixels
            ind_saturated = np.where(self.Raw_data >= 4050)
            # Mask for the saturated pixels
            mask_saturated =  np.ones(self.arr_organelle1.shape)
            # Set saturated to zero across all channels
            mask_saturated[ind_saturated[0],ind_saturated[1]] = 0
            
            # Apply to unmixed data
            arr_organelle1_nosat = self.arr_organelle1*mask_saturated
            arr_organelle2_nosat = self.arr_organelle2*mask_saturated
            
            # Pixels that we are sure of to be in a certain organelle (based from unmixed data)
            organelle1_pix = select_pix(arr_organelle1_nosat, arr_organelle2_nosat)
            organelle2_pix = select_pix(arr_organelle2_nosat, arr_organelle1_nosat)

            # Find indices of points that satisfy the condition
            indx_pixorg1 = indx_select_pix(arr_organelle1_nosat, arr_organelle2_nosat, organelle1_pix)
            indx_pixorg2 = indx_select_pix(arr_organelle2_nosat, arr_organelle1_nosat, organelle2_pix)

            # Plot the 2 arrays in a scatter plot
            plt.scatter(arr_organelle2_nosat, arr_organelle1_nosat, alpha=0.3)

            # Highlight the selected point
            plt.scatter(arr_organelle2_nosat[indx_pixorg1], arr_organelle1_nosat[indx_pixorg1], color='red',
                        label=f'${self.organelle2} = 0, {self.organelle1} \geq {np.min(organelle1_pix)}$')
            plt.scatter(arr_organelle2_nosat[indx_pixorg2], arr_organelle1_nosat[indx_pixorg2], color='green',
                        label=f'${self.organelle2} \geq {np.min(organelle2_pix)}, {self.organelle1} = 0$')
        else:
            # Pixels that we are sure of to be in a certain organelle (based from unmixed data)
            organelle1_pix = select_pix(self.arr_organelle1, self.arr_organelle2)
            organelle2_pix = select_pix(self.arr_organelle2, self.arr_organelle1)

            # Find indices of points that satisfy the condition
            indx_pixorg1 = indx_select_pix(self.arr_organelle1, self.arr_organelle2, organelle1_pix)
            indx_pixorg2 = indx_select_pix(self.arr_organelle2, self.arr_organelle1, organelle2_pix)

            # Plot the 2 arrays in a scatter plot
            plt.scatter(self.arr_organelle2, self.arr_organelle1, alpha=0.3)

            # Highlight the selected point
            plt.scatter(self.arr_organelle2[indx_pixorg1], self.arr_organelle1[indx_pixorg1], color='red',
                        label=f'${self.organelle2} = 0, {self.organelle1} \geq {np.min(organelle1_pix)}$')
            plt.scatter(self.arr_organelle2[indx_pixorg2], self.arr_organelle1[indx_pixorg2], color='green',
                        label=f'${self.organelle2} \geq {np.min(organelle2_pix)}, {self.organelle1} = 0$')

        # Add labels and title
        plt.ylabel(f'Pixel intensity ({self.organelle1})')
        plt.xlabel(f'Pixel intensity ({self.organelle2})')

        return plt, indx_pixorg1, indx_pixorg2
    
    # ------------------------------------------------------------------------
    def remove_saturated_pix(self):
        """ 
        Create mask to set intensity values of saturated pixels to zero
        """
        # Location of saturated pixels
        ind_saturated = np.where(self.Raw_data >= 4050)
        # Mask for the saturated pixels
        mask_saturated =  np.ones(self.Raw_data[:,:,0].shape)
        # Set saturated to zero across all channels
        mask_saturated[ind_saturated[0],ind_saturated[1]] = 0
        processed_matrix = self.Raw_data*mask_saturated[:,:,np.newaxis]
        return processed_matrix

    # ------------------------------------------------------------------------
    def generate_indx_select_pix(self,Num_toppixels):
        """ 
        Pick specific number of brightest pixels in single organelle image
        """
        # Ignore saturated pixels
        filtered_data = self.remove_saturated_pix()
        # Calc. ave intensity across CHs
        mean_Raw_data = np.round(np.mean(filtered_data, axis=-1), decimals=2)
        # Sort the result array in increasing order
        sorted_result = np.sort(mean_Raw_data.flatten())
        # Get the top pixels from the sorted result w/o saturated pixels
        top_numbers = sorted_result[-Num_toppixels:]
        indices = np.where(mean_Raw_data >= np.min(top_numbers))
        return indices


    # ------------------------------------------------------------------------
    def generate_normSpectrum(self, indx_pixorg, Num_CHtocombi, withIinfo, Label):
        """ 
        Generate spectrum plot of chosen pixels in one plot
        """
        if Num_CHtocombi == 0: # if you don't combine CHs
            values = []

            if withIinfo == True:
                area_under_curve_list = []
                values_norm_Iinfo_list = []
                for ind in range(len(indx_pixorg[0])):
                    values_along_CH = self.Raw_data[indx_pixorg[0][ind], indx_pixorg[1][ind], :]
                    # Calculate the definite integral of the curve
                    area_under_curve = simps(values_along_CH, self.xvalues_nm)
                    # Normalize the curve by dividing each data point by the area under the curve
                    norm_values_along_CH = values_along_CH / area_under_curve
                    values.append(norm_values_along_CH)
                    area_under_curve_list.append(area_under_curve)

                # Calculating mean area under the curve to recover I info
                meanI = np.mean(area_under_curve_list, axis=0)

                # Recover I info with respect to the mean
                for ind in range(len(values)):
                    values_norm_Iinfo = values[ind]*meanI
                    values_norm_Iinfo_list.append(values_norm_Iinfo)

                    plt.plot(self.xaxis, values_norm_Iinfo, color='gray', alpha=0.07, marker='o')
                    plt.xticks(self.xaxis, rotation=90)
                    plt.xlabel(r'CH # ($\lambda$ in nm)')

                # Calculating mean and std
                mean = np.mean(values_norm_Iinfo_list, axis=0)
                std = np.std(values_norm_Iinfo_list, axis=0, ddof=1)
                sem = std/len(values)
            else: 
                for ind in range(len(indx_pixorg[0])):
                    values_along_CH = self.Raw_data[indx_pixorg[0][ind], indx_pixorg[1][ind], :]
                    area_under_curve = simps(values_along_CH, self.xvalues_nm)
                    norm_values_along_CH = values_along_CH / area_under_curve
                    values.append(norm_values_along_CH)

                    plt.plot(self.xaxis, norm_values_along_CH, color='gray', alpha=0.07, marker='o')
                    plt.xticks(self.xaxis, rotation=90)
                    plt.xlabel(r'CH # ($\lambda$ in nm)')

                # Calculating mean and std
                mean = np.mean(values, axis=0)
                std = np.std(values, axis=0, ddof=1)
                sem = std/len(values)

            # Plotting mean and std
            plt.plot(self.xaxis, mean, marker='o',label=f'{Label} (Mean)')
            plt.fill_between(self.xaxis, mean-std, mean+std,
                alpha=0.7,label='Std')
            # plt.fill_between(self.xaxis, mean-sem, mean+sem,
            #     alpha=0.7,label='SEM')
            
        else:
            values = []
            for ind in range(len(indx_pixorg[0])):
                values_along_CH = self.Raw_data[indx_pixorg[0][ind], indx_pixorg[1][ind], :]
                consecutive_means = []
                combi_xvalues_nm = []
                combi_CH = []
                for i in range(0, len(values_along_CH) - 1, Num_CHtocombi):
                    mean_value = np.mean(values_along_CH[i:i+Num_CHtocombi])
                    consecutive_means.append(mean_value)
                    combi_xvalues_nm.append(np.mean(self.xvalues_nm[i:i+Num_CHtocombi]))
                    combi_CH.append(np.array(range(self.Num_CH))[i:i+Num_CHtocombi] + 1)

                consecutive_means = np.asarray(consecutive_means)
                combi_xvalues_nm = np.asarray(combi_xvalues_nm)
                # Calculate the definite integral of the curve
                area_under_curve = simps(consecutive_means, combi_xvalues_nm)
                # Normalize the curve by dividing each data point by the area under the curve
                norm_consecutive_means = consecutive_means / area_under_curve
                values.append(norm_consecutive_means)
                # xvalues of combined CHs
                combi_xvalues = [f'{combi_CH[a][0]}-{combi_CH[a][-1]} ({combi_xvalues_nm[a]})' for a in range(len(combi_xvalues_nm))]

                plt.plot(combi_xvalues, norm_consecutive_means, color='gray', alpha=0.07, marker='o')
                plt.xticks(rotation=90)
                plt.xlabel(r'CH # ($\lambda$ in nm)')

            # Plotting mean and std
            mean = np.mean(values, axis=0)
            std = np.std(values, axis=0, ddof=1)

            plt.plot(combi_xvalues, mean, marker='o',label=f'{Label} (Mean)')
            plt.fill_between(combi_xvalues, mean-std, mean+std,
                alpha=0.7, label='Std')
        
        # Other plot features
        plt.ylim(bottom=0)
        # Adjust layout to prevent overlapping
        plt.tight_layout()
        
        return plt, mean
    
    # ------------------------------------------------------------------------
    def spectrum_indiv_pixel_sep(self, indx_top10org):
        """ 
        Plot raw spectra of pixels separately
        """
        # Assuming indices1 is a tuple of arrays
        num_plots = len(indx_top10org[0])

        # Create subplots
        fig, axs = plt.subplots(num_plots, 1, figsize=(8, 1.5 * num_plots), sharex=True, sharey=True)

        for ind, ax in enumerate(axs):
            values_along_CH = self.Raw_data[indx_top10org[0][ind], indx_top10org[1][ind], :]

            # Plot values along CH
            ax.plot(self.xaxis, values_along_CH, marker='o')
            ax.set_xticks(self.xaxis)
            ax.set_xticklabels(self.xaxis, rotation=90)

        fig.supylabel('Pixel intensity')
        fig.supxlabel(r'CH # ($\lambda$ in nm)')
        # Adjust layout to prevent overlapping
        plt.tight_layout()

        return fig
    
    # ------------------------------------------------------------------------
    def calc_ratio_peaks(self,values,min1,max1,min2,max2):
        """ 
        Calculate the ratio of specified wavelength range
        """
        peak1 = np.mean(values[min1:max1])
        peak2 = np.mean(values[min2:max2])
        return peak1/peak2
    
    # ------------------------------------------------------------------------
    def det_contact(self,values,min1,max1,min2,max2):
        """ 
        Calc of ratios to det contact
        """
        if np.max(values) >= 350: # threshold for bkg 
            test_org1 = self.calc_ratio_peaks(values,min1,max1,min2,max2)
            test_org2 = self.calc_ratio_peaks(values,min2,max2,min1,max1)
            # if  0.95 <= test_org1  and  test_org1 <= 1.05 and 0.95 <= test_org2  and test_org2 <= 1.05:
            if test_org1 <= 1.05 and test_org2 <= 1.05:
                return True
            else:
                return False
        else:
            return False

    # ------------------------------------------------------------------------ 
    def pix_contact_ROI(self,pix_ROI,min1,max1,min2,max2):
        """ 
        Obtain pix determined as contact
        """
        x_list = []
        y_list = []
        for ind in range(len(pix_ROI[0])):
            values_along_CH = self.Raw_data[pix_ROI[0][ind], pix_ROI[1][ind], :]    
            test_for_contact = self.det_contact(values_along_CH,min1,max1,min2,max2)
            if test_for_contact == True:
                x_list.append(pix_ROI[1][ind])
                y_list.append(pix_ROI[0][ind])

        x_list = np.asarray(x_list)
        y_list = np.asarray(y_list)

        return np.stack((y_list,x_list))
    
    # ------------------------------------------------------------------------
    def pix_in_image(self,indx_pixorg,composite_filename,color,vmax_value):
        """ 
        Encircle selected pix in image
        """
        image = Image.open(f'{composite_filename}.png')
        # Specify the coordinates around which you want to draw circles
        y_coord = indx_pixorg[0]
        x_coord = indx_pixorg[1]
        
        # Create the heatmap
        plt.imshow(image, cmap='gray', vmax=vmax_value)
        # plt.colorbar()
        
        # Add circles around the specified coordinates
        circle_radius = 0.7  # You can adjust the radius as needed
        
        for x, y in zip(x_coord, y_coord):
            circle = patches.Circle((x, y), circle_radius, edgecolor= color, facecolor='none', linewidth=0.5)
            plt.gca().add_patch(circle)