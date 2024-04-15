
import javabridge
import bioformats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    def obtain_unmixed(self):
        """ 
        Obtain unmixed data
        """
        return self.arr_organelle1, self.arr_organelle2
    
    # ------------------------------------------------------------------------
    def remove_dead_cells(self,mask_filename):
        """ 
        Use ROI corres. to dead cells to remove them from the image
        """
        mask = Image.open(f'{mask_filename}.tif')
        mask = np.asarray(mask)
        mask[mask == 0] = 10
        mask[mask == 255] = 0
        mask[mask == 10] = 1
        self.Raw_data = self.Raw_data*mask[:,:,np.newaxis]
        return self.Raw_data
    
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

                # Recalculating mean area under the curve
                meanI = simps(mean, self.xvalues_nm)
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

                # Calculating mean area under the curve
                meanI = simps(mean, self.xvalues_nm)

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
        
        return plt, mean, meanI
    
    # ------------------------------------------------------------------------
    def spectrum_indiv_pixel_sep(self, indx_pix):
        """ 
        Plot raw spectra of pixels separately
        """
        # Assuming indx_pix is a tuple of arrays
        num_plots = len(indx_pix[0])
        plots_per_column = 10
        num_columns = (num_plots + plots_per_column - 1) // plots_per_column  # Ceiling division to get number of columns

        # Create subplots
        fig, axs = plt.subplots(plots_per_column, num_columns, figsize=(5*num_columns, 1.5 * plots_per_column), sharex=True, sharey=True)

        for col in range(num_columns):
            for row in range(plots_per_column):
                plot_index = col * plots_per_column + row
                if plot_index >= num_plots:
                    break

                values_along_CH = self.Raw_data[indx_pix[0][plot_index], indx_pix[1][plot_index], :]
                axs[row, col].plot(self.xaxis, values_along_CH, marker='o')
                if row == plots_per_column - 1:
                    axs[row, col].set_xticks(self.xaxis)
                    axs[row, col].set_xticklabels(self.xaxis, rotation=90)
                else:
                    axs[row, col].set_xticks(self.xaxis)
                    axs[row, col].set_xticklabels([])  # Remove tick labels for non-last rows

        # fig.supylabel('Pixel intensity')
        # fig.supxlabel(r'CH # ($\lambda$ in nm)')
        # Adjust layout to prevent overlapping
        plt.tight_layout()

        return plt
    
    # ------------------------------------------------------------------------
    def spectrum_indiv_pixel_sep_hor(self, indx_pix):
        """ 
        Plot raw spectra of pixels separately
        """
        # Assuming indx_pix is a tuple of arrays
        num_plots = len(indx_pix[0])
        plots_per_column = 10
        num_columns = (num_plots + plots_per_column - 1) // plots_per_column  # Ceiling division to get number of columns

        # Create subplots
        fig, axs = plt.subplots(plots_per_column, num_columns, figsize=(5*num_columns, 1.5 * plots_per_column), sharex=True, sharey=True)

        for col in range(num_columns):
            for row in range(plots_per_column):
                plot_index = col * plots_per_column + row
                if plot_index >= num_plots:
                    break

                values_along_CH = self.Raw_data[indx_pix[0][plot_index], indx_pix[1][plot_index], :]
                axs[row, col].plot(self.xaxis, values_along_CH, marker='o')
                if row == plots_per_column - 1:
                    axs[row, col].set_xticks(self.xaxis)
                    axs[row, col].set_xticklabels(self.xaxis, rotation=90)
                else:
                    axs[row, col].set_xticks(self.xaxis)
                    axs[row, col].set_xticklabels([])  # Remove tick labels for non-last rows

        # fig.supylabel('Pixel intensity')
        # fig.supxlabel(r'CH # ($\lambda$ in nm)')
        # Adjust layout to prevent overlapping
        plt.tight_layout()

        return plt
    
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
            if test_org1 <= 1.15 and test_org2 <= 1.15:
                return True
            else:
                return False
        else:
            return False
        
    # ------------------------------------------------------------------------
    def det_contact_byintgrl(self,values,bkg_mean_area_under_curve,min1,max1,min2,max2):
        """ 
        Calc of ratios to det contact
        """
        area_under_curve = simps(values, self.xvalues_nm)
        if area_under_curve >= bkg_mean_area_under_curve: # threshold for bkg 
            test_org1 = self.calc_ratio_peaks(values,min1,max1,min2,max2)
            test_org2 = self.calc_ratio_peaks(values,min2,max2,min1,max1)
            # if  0.95 <= test_org1  and  test_org1 <= 1.05 and 0.95 <= test_org2  and test_org2 <= 1.05:
            if test_org1 <= 1.35 and test_org2 <= 1.35:
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
    def pix_contact_ROI_byintgrl(self,pix_ROI,bkg_mean_area_under_curve,min1,max1,min2,max2):
        """ 
        Obtain pix determined as contact
        """
        x_list = []
        y_list = []
        for ind in range(len(pix_ROI[0])):
            values_along_CH = self.Raw_data[pix_ROI[0][ind], pix_ROI[1][ind], :]    
            test_for_contact = self.det_contact_byintgrl(values_along_CH,bkg_mean_area_under_curve,min1,max1,min2,max2)
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
    
    # ------------------------------------------------------------------------
    def calc_peaks_mean(self,min1,max1,min2,max2,withMask,maskpix):
        """ 
        Generate scatter plot of mean intensity of two regions of determined peaks of the two spectra
        """
        if withMask == True:
            y_coord = maskpix[0]
            x_coord = maskpix[1]
            # Create array with same len as num of pix
            peak1 = np.zeros(len(y_coord))
            peak2 = np.zeros(len(y_coord))
            for x, y in zip(x_coord, y_coord):
                values = self.Raw_data[y, x, :]
                peak1[y] = np.mean(values[min1:max1])  # ER peak
                peak2[y] = np.mean(values[min2:max2])  # M peak
            # print(values)
        else:
            # Create a 2D array with same shape as the raw data
            peak1 = np.zeros(self.Raw_data[:,:,0].shape)
            peak2 = np.zeros(self.Raw_data[:,:,0].shape)
            # Iterate over the rows of the array
            for i in range(self.Raw_data.shape[0]):
                # Iterate over the columns of the array
                for j in range(self.Raw_data.shape[1]):
                    # Get the values of the two peaks
                    values = self.Raw_data[i, j, :]
                    # Calculate the mean intensity of the two peaks
                    peak1[i, j] = np.mean(values[min1:max1])  # ER peak
                    peak2[i, j] = np.mean(values[min2:max2])  # M peak
        return peak1, peak2

    # ------------------------------------------------------------------------
    def generate_peaks_scatter_plot(self,peak1,peak2,min1,max1,min2,max2):
        """ 
        Generate scatter plot of mean intensity of two regions of determined peaks of the two spectra
        """
        # Define a function that selects pixels based on the mean intensity of the two peaks
        slope_diff = 0.5
        bkg_yint = 400
        bkg_xint = 200
        def indx_cntct_pix(array1, array2):
            ratio1 = array1 / array2
            ratio2 = array2 / array1
            # condition = np.logical_and(np.logical_and(ratio1 <= 1+slope_diff, ratio2 <= 1+slope_diff ), array1 > bkg_yint - ((bkg_yint/bkg_xint)*array2) )
            condition = np.logical_and(np.logical_and(ratio1 < 1+slope_diff, ratio2 < 1+slope_diff), array1 > bkg_yint - ((bkg_yint/bkg_xint)*array2) )
            indices = np.where(condition)
            return indices
        def indx_orgnanelle_pix(array1, array2):
            ratio1 = array1 / array2
            ratio2 = array2 / array1
            condition1 = np.logical_and(ratio1 > 1+slope_diff, array1 > bkg_yint - ((bkg_yint/bkg_xint)*array2) )
            condition2 = np.logical_and(ratio2 > 1+slope_diff, array1 > bkg_yint - ((bkg_yint/bkg_xint)*array2) )
            indices1 = np.where(condition1)
            indices2 = np.where(condition2)
            return indices1, indices2
        def indx_inspct_pix(array1, array2):
            ratio1 = array1 / array2
            # ratio2 = array2 / array1
            # condition = np.logical_and(np.logical_and(ratio1 <= 1+slope_diff, ratio2 <= 1+slope_diff ), array1 > bkg_yint - ((bkg_yint/bkg_xint)*array2) )
            condition = np.logical_and(ratio1 < 2/5, array2 > 300)
            indices = np.where(condition)
            return indices
        # Find indices of points that satisfy the condition
        # contact_pix = indx_cntct_pix(peak1, peak2)
        # organelle1_pix, organelle2_pix = indx_orgnanelle_pix(peak1, peak2)
        inspct_pix = indx_inspct_pix(peak1, peak2)
        
        # Create a scatter plot of the mean intensity of the two peaks
        plt.scatter(peak2, peak1, alpha=0.07, marker='.')
        # Highlight the selected points
        # plt.scatter(peak2[contact_pix], peak1[contact_pix], color='orange', alpha=0.05, marker='.', label='Contact points')
        # plt.scatter(peak2[organelle1_pix], peak1[organelle1_pix], color='red', marker='.', label=f'{self.organelle1} points')
        # plt.scatter(peak2[organelle2_pix], peak1[organelle2_pix], color='green', marker='.', label=f'{self.organelle2} points')
        plt.scatter(peak2[inspct_pix], peak1[inspct_pix], color='green', marker='.', label=f'{self.organelle2} points')
        # Create fitted lines
        # x = np.linspace(0, 4050, 1000)
        # plt.plot(x,x, color='grey', linestyle='--', label=f'ratio1,2 = 1')
        # plt.plot(x,bkg_yint - ((bkg_yint/bkg_xint)*x), color='black', linestyle='--', label='Bkg threshold')  
        # plt.plot(x,(1/(1+slope_diff))*x, color='grey')
        # plt.plot(x,(1+slope_diff)*x, color='grey')
        # Add labels and title
        plt.ylabel(f'Mean intensity of ER peak (CHs {min1+1}-{max1+1})')
        plt.xlabel(f'Mean intensity of M peak (CHs {min2+1}-{max2+1})')

        return plt, inspct_pix  # organelle1_pix, organelle2_pix, contact_pix 
    
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

        # Create a contour plot to map the density of the points
        plt.hexbin(peak2[indices], peak1[indices], gridsize=100, bins='log', cmap='inferno') # bins='log',
        # fig = plt.figure(figsize=(5,5))
        # plt.scatter(peak2, peak1, alpha=0.3, marker='.')
        # sns.kdeplot(x = peak2[indices], y = peak1[indices], color='black',alpha=0.5, levels=5, linewidths=1.5)
        plt.colorbar()
        plt.ylabel(f'Mean intensity of ER peak (CHs {min1+1}-{max1+1})')
        plt.xlabel(f'Mean intensity of M peak (CHs {min2+1}-{max2+1})')

        return plt
    
    # ------------------------------------------------------------------------
    def generate_peaktounmixed_scatter_plot(self, indx_pixorg1, indx_pixorg2, indx_pixcontact):
        """ 
        Generate scatter plot from unmixed data
        """
        # Plot the 2 arrays in a scatter plot
        plt.scatter(self.arr_organelle2, self.arr_organelle1, marker='.', alpha=0.1)

        # Highlight the selected point
        plt.scatter(self.arr_organelle2[indx_pixorg1], self.arr_organelle1[indx_pixorg1], marker='.',color='red',
                    label=f'${self.organelle1}$')
        plt.scatter(self.arr_organelle2[indx_pixorg2], self.arr_organelle1[indx_pixorg2], color='green', marker='.',
                    label=f'${self.organelle2}$')
        plt.scatter(self.arr_organelle2[indx_pixcontact], self.arr_organelle1[indx_pixcontact], marker='.', color='orange',
                    label=f'$contact$')

        # Add labels and title
        plt.ylabel(f'Pixel intensity ({self.organelle1})')
        plt.xlabel(f'Pixel intensity ({self.organelle2})')

        return plt
    
    # ------------------------------------------------------------------------
    def generate_peak1org_scatter_plot(self,min1,max1,min2,max2,indx_pixorg1,indx_pixorg2,color1,color2):
        """ 
        Generate scatter plot of mean intensity of two regions of determined peaks of the two spectra
        """
        # Create a 2D array with same shape as the raw data
        peak1 = np.zeros(self.Raw_data[:,:,0].shape)
        peak2 = np.zeros(self.Raw_data[:,:,0].shape)
        # Iterate over the rows of the array
        for i in range(self.Raw_data.shape[0]):
            # Iterate over the columns of the array
            for j in range(self.Raw_data.shape[1]):
                # Get the values of the two peaks
                values = self.Raw_data[i, j, :]
                # Calculate the mean intensity of the two peaks
                peak1[i, j] = np.mean(values[min1:max1])  # ER peak
                peak2[i, j] = np.mean(values[min2:max2])  # M peak

        # Define a function that selects pixels based on the mean intensity of the two peaks
        slope_diff = 0.5
        bkg_yint = 400
        bkg_xint = 200
        def indx_cntct_pix(array1, array2):
            ratio1 = array1 / array2
            ratio2 = array2 / array1
            # condition = np.logical_and(np.logical_and(ratio1 <= 1+slope_diff, ratio2 <= 1+slope_diff ), array1 > bkg_yint - ((bkg_yint/bkg_xint)*array2) )
            condition = np.logical_and(np.logical_and(ratio1 < 1+slope_diff, ratio2 < 1+slope_diff), array1 > bkg_yint - ((bkg_yint/bkg_xint)*array2) )
            indices = np.where(condition)
            return indices
        def indx_orgnanelle_pix(array1, array2):
            ratio1 = array1 / array2
            ratio2 = array2 / array1
            condition1 = np.logical_and(ratio1 > 1+slope_diff, array1 > bkg_yint - ((bkg_yint/bkg_xint)*array2) )
            condition2 = np.logical_and(ratio2 > 1+slope_diff, array1 > bkg_yint - ((bkg_yint/bkg_xint)*array2) )
            indices1 = np.where(condition1)
            indices2 = np.where(condition2)
            return indices1, indices2
        # Find indices of points that satisfy the condition
        # contact_pix = indx_cntct_pix(peak1, peak2)
        # organelle1_pix, organelle2_pix = indx_orgnanelle_pix(peak1, peak2)
        
        # Create a scatter plot of the mean intensity of the two peaks
        plt.scatter(peak2, peak1, alpha=0.01, marker='.')
        # Highlight the selected points
        # plt.scatter(peak2[contact_pix], peak1[contact_pix], color='orange', alpha=0.1, marker='.', label='Contact points')
        # plt.scatter(peak2[organelle1_pix], peak1[organelle1_pix], color='red', marker='.', label=f'{self.organelle1} points')
        # plt.scatter(peak2[organelle2_pix], peak1[organelle2_pix], color='green', marker='.', label=f'{self.organelle2} points')
        plt.scatter(peak2[indx_pixorg1], peak1[indx_pixorg1], color=f'{color1}', alpha=0.03, marker='.')
        plt.scatter(peak2[indx_pixorg2], peak1[indx_pixorg2], color=f'{color2}', alpha=0.03, marker='.')
        # Create fitted lines
        x = np.linspace(0, 4050, 1000)
        plt.plot(x,x, color='grey', linestyle='--', label=f'ratio1,2 = 1')
        plt.plot(x,bkg_yint - ((bkg_yint/bkg_xint)*x), color='black', linestyle='--', label='Bkg threshold')  
        plt.plot(x,(1/(1+slope_diff))*x, color='grey')
        plt.plot(x,(1+slope_diff)*x, color='grey')
        # Add labels and title
        plt.ylabel(f'Mean intensity of ER peak (CHs {min1+1}-{max1+1})')
        plt.xlabel(f'Mean intensity of M peak (CHs {min2+1}-{max2+1})')
        
        return plt
    
    