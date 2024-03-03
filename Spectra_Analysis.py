
import javabridge
import bioformats
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

javabridge.start_vm(class_path=bioformats.JARS)

class Spectra:
    """ 
    For analyzing spectrum of image data with zstacks 
    """
    def __init__(self, raw_data_filename, unmixed_filename, bin_size, Num_pixels, Num_CH, Num_zstack, organelle1, organelle2, start_nm):
        self.raw_data_filename = raw_data_filename
        self.unmixed_filename = unmixed_filename
        self.bin_size = bin_size
        self.Num_pixels = Num_pixels
        self.Num_CH = Num_CH
        self.Num_zstack = Num_zstack
        self.organelle1 = organelle1
        self.organelle2 = organelle2
        self.start_nm = start_nm

        # Load raw data
        self.Raw_data = np.empty([self.Num_zstack, self.Num_pixels, self.Num_pixels, self.Num_CH])
        image_temp = bioformats.load_image(
            path=f'C:\\Users\\admin\\OneDrive - Washington University in St. Louis\\PhD\\Research\\Biophys\\image_data\\{self.raw_data_filename}.nd2',
            c=None,
            rescale=False
        )
        self.Raw_data[: ,:, :, :] = image_temp
        
        if self.unmixed_filename == False:
            pass
        else:
            # Load unmixed data
            self.Unmixed_data = np.empty([self.Num_zstack, self.Num_pixels, self.Num_pixels, 2])
            image_temp = bioformats.load_image(
                path=f'C:\\Users\\admin\\OneDrive - Washington University in St. Louis\\PhD\\Research\\Biophys\\image_data\\{self.unmixed_filename}.nd2',
                c=None,
                rescale=False
            )
            self.Unmixed_data[:, :, :, :] = image_temp
            
        # x-axis for the spectrum later on
        self.xaxis = [f'{a+1} ({self.start_nm + self.bin_size * a})' for a in range(self.Num_CH)]
        self.xvalues = np.array([a+1 for a in range(self.Num_CH)])
        self.xvalues_nm = np.array([self.start_nm + self.bin_size * a for a in range(self.Num_CH)])
    
    # Generate scatter plot
    def generate_scatter_plot(self,saturated_pix, Num_toppixels, zstack_ind):
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
        
        # Specify array of organelle
        arr_organelle1 = self.Unmixed_data[zstack_ind, :, :, 0]
        arr_organelle2 = self.Unmixed_data[zstack_ind, :, :, 1]
        
        if saturated_pix == False:
            # Location of saturated pixels
            ind_saturated = np.where(self.Raw_data[zstack_ind,:,:,:] >= 4050)
            # Mask for the saturated pixels
            mask_saturated =  np.ones(arr_organelle1.shape)
            # Set saturated to zero across all channels
            mask_saturated[ind_saturated[0],ind_saturated[1]] = 0
            
            # Apply to unmixed data
            arr_organelle1_nosat = arr_organelle1*mask_saturated
            arr_organelle2_nosat = arr_organelle2*mask_saturated
            
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
            organelle1_pix = select_pix(arr_organelle1, arr_organelle2)
            organelle2_pix = select_pix(arr_organelle2, arr_organelle1)

            # Find indices of points that satisfy the condition
            indx_pixorg1 = indx_select_pix(arr_organelle1, arr_organelle2, organelle1_pix)
            indx_pixorg2 = indx_select_pix(arr_organelle2, arr_organelle1, organelle2_pix)

            # Plot the 2 arrays in a scatter plot
            plt.scatter(arr_organelle2, arr_organelle1, alpha=0.3)

            # Highlight the selected point
            plt.scatter(arr_organelle2[indx_pixorg1], arr_organelle1[indx_pixorg1], color='red',
                        label=f'${self.organelle2} = 0, {self.organelle1} \geq {np.min(indx_pixorg1)}$')
            plt.scatter(arr_organelle2[indx_pixorg2], arr_organelle1[indx_pixorg2], color='green',
                        label=f'${self.organelle2} \geq {np.min(indx_pixorg2)}, {self.organelle1} = 0$')
            

        # Add labels and title
        plt.ylabel(f'Pixel intensity ({self.organelle1})')
        plt.xlabel(f'Pixel intensity ({self.organelle2})')


        return plt, indx_pixorg1, indx_pixorg2

    # Generate spectrum plot of chosen pixels
    def generate_normSpectrum(self, indx_pixorg, Num_CHtocombi, zstack_ind):
        if Num_CHtocombi == 0: # if you don't combine CHs
            values = []
            area_under_curve_list = []
            values_norm_Iinfo_list = []
            for ind in range(len(indx_pixorg[0])):
                values_along_CH = self.Raw_data[zstack_ind, indx_pixorg[0][ind], indx_pixorg[1][ind], :]
                # Calculate the definite integral of the curve
                area_under_curve = simps(values_along_CH, self.xvalues_nm)
                norm_values_along_CH = values_along_CH / area_under_curve
                values.append(norm_values_along_CH)
                area_under_curve_list.append(area_under_curve)

            # Calculating mean and std
            meanI = np.mean(area_under_curve_list, axis=0)

            # Normalize the curve by dividing each data point by the area under the curve
            for ind in range(len(values)):
                values_norm_Iinfo = values[ind]*meanI
                values_norm_Iinfo_list.append(values_norm_Iinfo)

                plt.plot(self.xaxis, values_norm_Iinfo, color='gray', marker='o')
                plt.xticks(self.xaxis, rotation=90)
                plt.xlabel(r'CH # ($\lambda$ in nm)')

            # Plotting mean and std
            mean = np.mean(values_norm_Iinfo_list, axis=0)
            std = np.std(values_norm_Iinfo_list, axis=0, ddof=1)

            plt.plot(self.xaxis, mean, color='#CC4F1B', marker='o',label='Mean')
            plt.fill_between(self.xaxis, mean-std, mean+std,
                alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848',label='Std')
        else:
            values = []
            for ind in range(len(indx_pixorg[0])):
                values_along_CH = self.Raw_data[zstack_ind, indx_pixorg[0][ind], indx_pixorg[1][ind], :]
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

                plt.plot(combi_xvalues, norm_consecutive_means, color='gray', marker='o')
                plt.xticks(rotation=90)
                plt.xlabel(r'CH # ($\lambda$ in nm)')

            # Plotting mean and std
            mean = np.mean(values, axis=0)
            std = np.std(values, axis=0, ddof=1)

            plt.plot(combi_xvalues, mean, color='#CC4F1B', marker='o',label='Mean')
            plt.fill_between(combi_xvalues, mean-std, mean+std,
                alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848',label='Std')
        
        # Other plot features
        plt.ylim(bottom=0)
        # Adjust layout to prevent overlapping
        plt.tight_layout()
        
        return plt

class Spectra_no_z:
    """ 
    For analyzing spectrum of image data w/o zstacks
    """
    def __init__(self, raw_data_filename, unmixed_filename, bin_size, Num_pixels, Num_CH, organelle1, organelle2, start_nm):
        self.raw_data_filename = raw_data_filename
        self.unmixed_filename = unmixed_filename
        self.bin_size = bin_size
        self.Num_pixels = Num_pixels
        self.Num_CH = Num_CH
        self.organelle1 = organelle1
        self.organelle2 = organelle2
        self.start_nm = start_nm

        # Load raw data
        self.Raw_data = np.empty([self.Num_pixels, self.Num_pixels, self.Num_CH])
        image_temp = bioformats.load_image(
            path=f'C:\\Users\\admin\\OneDrive - Washington University in St. Louis\\PhD\\Research\\Biophys\\image_data\\{self.raw_data_filename}.nd2',
            c=None,
            rescale=False
        )
        self.Raw_data[:, :, :] = image_temp
        
        if self.unmixed_filename == False:
            pass
        else:
            # Load unmixed data
            self.Unmixed_data = np.empty([self.Num_pixels, self.Num_pixels, 2])
            image_temp = bioformats.load_image(
                path=f'C:\\Users\\admin\\OneDrive - Washington University in St. Louis\\PhD\\Research\\Biophys\\image_data\\{self.unmixed_filename}.nd2',
                c=None,
                rescale=False
            )
            self.Unmixed_data[:, :, :] = image_temp

            # Specify array of organelle
            self.arr_organelle1 = self.Unmixed_data[:, :, 0]
            self.arr_organelle2 = self.Unmixed_data[:, :, 1]

        # x-axis for the spectrum later on
        self.xaxis = [f'{a+1} ({self.start_nm + self.bin_size * a})' for a in range(self.Num_CH)]
        self.xvalues = np.array([a+1 for a in range(self.Num_CH)])
        self.xvalues_nm = np.array([self.start_nm + self.bin_size * a for a in range(self.Num_CH)])

    def generate_scatter_plot_varNumpixels(self,saturated_pix, Num_toppixels):
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
                        label=f'${self.organelle2} = 0, {self.organelle1} \geq {np.min(indx_pixorg1)}$')
            plt.scatter(self.arr_organelle2[indx_pixorg2], self.arr_organelle1[indx_pixorg2], color='green',
                        label=f'${self.organelle2} \geq {np.min(indx_pixorg2)}, {self.organelle1} = 0$')

        # Add labels
        plt.ylabel(f'Pixel intensity ({self.organelle1})')
        plt.xlabel(f'Pixel intensity ({self.organelle2})')

        return plt, indx_pixorg1, indx_pixorg2