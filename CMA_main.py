#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectrum analysis and optimization using classes.

Created on Mon Aug 26 07:57:23 2024
@author: fg
"""

import numpy as np
import pandas as pd
from PIL import Image
import glob
import re
import matplotlib.pyplot as plt
import cma
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm
class Config:
    """Configuration parameters for the analysis."""
    N_guess = 10  # Number of energy guesses for the optimization. Be carefull highly non-linear ! Do not go above 200 points on local machine !
    
    E_guess_range = (np.log10(5e-2), np.log10(100))  # Energy range for guesses (log scale)
    smooth_factor = 1.3e-5 * np.mean(E_guess_range)  # Smoothing factor for the optimization objective
    lower_bound = -10  # Lower bound for optimization variables
    upper_bound = 10  # Upper bound for optimization variables
    
    folder_path = ""  # Path to response matrix files
    image_path = ""  # Path to image file for analysis

    # Region of Interest (ROI) for analyzing the image data
    ROI = np.array([[524, 649, 933, 1121], [709, 828, 933, 1104], [866, 901, 926, 1114], [940, 964, 915, 1118], 
                    [1006, 1034, 915, 1121], [1080, 1111, 912, 1111], [1157, 1185, 915, 1125], [1227, 1255, 908, 1125], 
                    [1283, 1321, 905, 1121], [1356, 1387, 915, 1118], [1422, 1454, 915, 1118], [1489, 1524, 919, 1118], 
                    [1562, 1594, 908, 1111], [1632, 1664, 915, 1104], [1702, 1730, 919, 1107], [1772, 1800, 915, 1118], 
                    [1828, 1860, 919, 1111]])  # ROI coordinates for scintillators

    N_sim = 2000  # Number of simulation runs
    N_FLUKA = 17  # Number of FLUKA simulations
    N_data = 449 - 35 - 15 - 26  # Number of data points after filtering

    # Calibration factors for the different scintillator regions
    factor = np.ones((N_FLUKA,))
    # Initialize ErrorAnalysis with error data and parameters
    error_files = ['mean1_VAC.txt', 'mean2_VAC.txt', 'mean3_VAC.txt', 'mean4_VAC.txt']
    E_error = np.array([3.31926620e-02, 3.99210913e-02, 4.80134292e-02, 5.64290847e-02,
                        6.63198120e-02, 7.97633907e-02, 9.37440879e-02, 1.12746796e-01,
                        1.35601511e-01, 1.59369353e-01, 1.91674849e-01, 2.25271064e-01,
                        2.70935387e-01, 3.18424160e-01, 3.82971391e-01, 4.50097513e-01,
                        5.41336030e-01, 6.51069356e-01, 7.65186915e-01, 8.99306672e-01,
                        1.10684849e+00, 1.30085370e+00, 1.56454759e+00, 1.88169442e+00,
                        2.21151240e+00, 2.59913992e+00, 3.12600724e+00, 3.75967497e+00,
                        4.41865996e+00, 5.31435916e+00, 6.24584471e+00, 7.51192949e+00,
                        8.82859884e+00, 1.06182294e+01, 1.27706330e+01, 1.50090328e+01,
                        1.80514888e+01, 2.12155017e+01, 2.55160607e+01, 2.99884423e+01,
                        3.60673495e+01, 4.33785019e+01, 5.09817606e+01, 6.13161884e+01,
                        7.20635132e+01, 8.66713880e+01, 1.04240401e+02, 1.22511358e+02,
                        1.43984795e+02, 1.73171713e+02, 2.08275063e+02, 2.44780916e+02,
                        2.87685409e+02, 3.54077391e+02, 4.16139055e+02, 5.00493910e+02,
                        6.01948197e+02, 6.91320378e+02, 8.31456781e+02, 1.00000000e+03])
    ddv = np.logspace(-10, 0, 1000)  # Array representing variation in parameter (e.g., detector distance)

class DataProcessor:
    """Handles data processing and image reading."""

    @staticmethod
    def initialize_array(N_data, N_sim):
        """Initialize arrays for spectrum and FLUKA data."""
        Spectrum_tot = np.zeros((N_data, N_sim))  # Spectrum data array
        FLUKA_tot = np.zeros((N_data, Config.N_FLUKA))  # FLUKA data array
        FLUKA_fact = np.zeros((N_data, 1))  # FLUKA normalization factors
        return Spectrum_tot, FLUKA_tot, FLUKA_fact

    @staticmethod
    def find_nearest(array, values):
        """Find nearest indices in array for given values."""
        used_indices = set()  # To keep track of used indices and avoid duplicates
        indices = []

        for val in values:
            index = np.argmin(np.abs(val - array))  # Find index of nearest value
            while index in used_indices:  # If index is already used, find the next available one
                index += 1
                if index >= len(array):
                    index = 0
                    print('Wrong end of E')  # Error message if no suitable index is found
                    return None
            if val > np.amax(array) or val < np.amin(array):  # Check if value is out of bounds
                print('E bin not in RM')  # Error message if value is out of range
                raise ValueError
            used_indices.add(index)
            indices.append(index)

        return np.array(indices)

    @staticmethod
    def read_image(norm=True):
        """Read and process image data."""
        im = Image.open(Config.image_path)  # Open the image
        im = np.array(im)  # Convert image to numpy array
        
        exp_spectrum_image = []
        # Process each ROI to extract the scintillator values
        for y_min, y_max, x_min, x_max in zip(Config.ROI[:, 0], Config.ROI[:, 1], Config.ROI[:, 2], Config.ROI[:, 3]):
            ROI_image = im[x_min:x_max, y_min:y_max]  # Extract the ROI from the image
            value_scint = np.sum(ROI_image)  # Sum the pixel values in the ROI
            
            if norm:
                value_scint = value_scint / ROI_image.size  # Normalize by the size of the ROI
            
            exp_spectrum_image.append(value_scint)
        exp_spectrum_image = np.array(exp_spectrum_image)
        return exp_spectrum_image

    @staticmethod
    def sort_numerically(filename):
        """Extract numerical value from filename for sorting."""
        num = re.findall(r"\d+\.\d+|\d+", filename)  # Extract numbers from filename
        return float(num[0]) if num else 0  # Return first number found as float, or 0 if none

    @classmethod
    def import_RM(cls):
        """Import Response Matrix data."""
        Spectrum_tot, FLUKA_tot, FLUKA_fact = cls.initialize_array(Config.N_data, Config.N_sim)  # Initialize arrays
        name_list = [0] * Config.N_data  # Initialize name list
        sorted_filenames_gauss = sorted(glob.glob(Config.folder_path), key=cls.sort_numerically)  # Sort files by numerical value
        start_name = 100 - 12 - 2  # Starting index for filename reduction

        for compteur, file in enumerate(sorted_filenames_gauss):
            file_red = file[start_name:]  # Reduce filename for easier handling
            if file_red.find("Spectrum") < 0:
                continue  # Skip if 'Spectrum' not found in reduced filename

            ind = file[start_name:].find(".") - file_red.find(".")  # Calculate index for file naming
            name = file_red[:file[start_name+4:].find(".")-5]  # Extract name from filename
            
            try:
                Energy = pd.read_csv(file[:ind+start_name]+name+"_Energy.txt", header=None)  # Read energy data
            except:
                print('File name error ', file, compteur)  # Error message if file reading fails
                continue
            
            FLUKA = pd.read_csv(file[:ind+start_name]+name+"_FLUKA.txt", header=None)  # Read FLUKA data
            FLUKA = np.array(FLUKA.values[:, 0])  # Convert FLUKA data to numpy array
            FLUKA *= Config.factor  # Apply calibration factors to FLUKA data
            FLUKA_fact_sum = np.amax(FLUKA)  # Find max value for normalization
            if FLUKA_fact_sum != 0:
                FLUKA /= np.amax(FLUKA)  # Normalize FLUKA data
            else:
                print('Empty file ', file, compteur)  # Error message for empty file
                continue
            FLUKA_tot[compteur, :] = FLUKA  # Store FLUKA data
            FLUKA_fact[compteur] = FLUKA_fact_sum  # Store normalization factor
            name_list[compteur] = float(name)  # Store name for sorting

        args = np.argsort(name_list)  # Sort indices by name
        E = np.array(name_list, dtype=float)[args]  # Sorted energy values
        Spectrum_tot = Spectrum_tot[args, :]  # Sort Spectrum array
        FLUKA_tot = FLUKA_tot[args, :]  # Sort FLUKA array
        FLUKA_fact = FLUKA_fact[args, :]  # Sort FLUKA factors
        return Spectrum_tot, FLUKA_tot, FLUKA_fact, E

class Optimizer:
    """Handles the CMA-ES optimization process."""

    def __init__(self, E, E_guess, FLUKA_tot, FLUKA_fact, Exp_FLUKA, smooth_factor):
        self.E = E  # Actual energy values
        self.E_guess = E_guess  # Guessed energy values
        self.FLUKA_tot = FLUKA_tot  # FLUKA data array
        self.FLUKA_fact = FLUKA_fact  # FLUKA normalization factors
        self.Exp_FLUKA = Exp_FLUKA  # Experimental FLUKA data
        self.smooth_factor = smooth_factor  # Smoothing factor for optimization
        self.nearest_indices = DataProcessor.find_nearest(self.E, self.E_guess)  # Indices of nearest E_guess in E

    def mmin(self, Simulated):
        """Objective function for optimization."""
        # Calculate the simulated FLUKA data
        Simulated_FLUKA = np.sum(10**Simulated.reshape(Config.N_guess, 1) * self.FLUKA_tot[self.nearest_indices, :] * 
                                 self.FLUKA_fact[self.nearest_indices, :], axis=0)
        
        # Calculate the error term (difference between experimental and simulated data)
        error_term = np.sum((self.Exp_FLUKA - Simulated_FLUKA) ** 2 / self.Exp_FLUKA**2)
        
        # Calculate the smoothness term (to enforce smooth solution)
        smoothness_term = np.sum(np.diff(Simulated) ** 2 / np.diff(np.log10(self.E_guess))**2)
        
        # Return the combined objective value
        return error_term + self.smooth_factor * smoothness_term

    def calc_FLUKA(self, Simulated):
        """Calculate FLUKA simulation based on optimized parameters."""
        # Sum over the FLUKA data to generate the simulated spectrum
        Simulated_FLUKA = np.sum(Simulated.reshape(Config.N_guess, 1) * self.FLUKA_tot[self.nearest_indices, :] * 
                                 self.FLUKA_fact[self.nearest_indices, :], axis=0)
        return Simulated_FLUKA

    def run_CMA(self):
        """Run CMA-ES optimization."""
        # Define CMA-ES options
        options = {
            'bounds': [Config.lower_bound, Config.upper_bound],  # Parameter bounds
            'maxiter': 100000,  # Maximum number of iterations
            'seed': 123456,  # Random seed for reproducibility
            'verb_disp': 1000,  # Display verbosity
            'tolx': 1e-6  # Tolerance for stopping criteria
        }

        initial_mean = np.random.uniform(Config.lower_bound, Config.upper_bound, Config.N_guess)  # Initial guess
        sigma = 10  # Initial step size

        es = cma.CMAEvolutionStrategy(initial_mean, sigma, options)  # Initialize CMA-ES
        while not es.stop():  # Run optimization until stopping criteria is met
            solutions = es.ask()  # Ask for new candidate solutions
            es.tell(solutions, [self.mmin(sol) for sol in solutions])  # Evaluate solutions and update strategy
            es.logger.add()  # Log the progress
            es.disp()  # Display the progress
        optimized_params = es.result.xbest  # Best found solution
        sim = 10**optimized_params  # Convert log-space parameters to linear scale
        return sim

class Plotter:
    """Handles plotting of results."""

    @staticmethod
    def plot_results(Exp_FLUKA, FLUKA_sim):
        """Plot experimental and simulated FLUKA results."""
        plt.figure(figsize=(10, 6))
        plt.plot(Exp_FLUKA, label='Exp')  # Plot experimental FLUKA data
        plt.plot(FLUKA_sim / np.amax(FLUKA_sim), label='Sim')  # Plot simulated FLUKA data
        plt.xlabel('Bin')
        plt.ylabel('Normalized Intensity')
        plt.title('Experimental vs Simulated FLUKA')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_spectrum(E_guess, sim):
        """Plot unfolded energy spectrum."""
        plt.figure(figsize=(10, 6))
        plt.plot(E_guess, sim)  # Plot the energy spectrum
        plt.xlabel('Energy (MeV)')
        plt.ylabel(r'dN/dE$_{log10}$')
        plt.title('Unfolded spectrum')
        plt.xscale('log')  # Logarithmic scale for energy axis
        plt.yscale('log')  # Logarithmic scale for intensity axis
        plt.legend()
        plt.grid(True)
        plt.show()
    
    


class ErrorAnalysis:
    """Handles error analysis, including smoothing and interpolation."""

    def __init__(self, E_error, error_files, ddv, window_size=200):
        """
        Initialize the ErrorAnalysis object.

        Parameters:
        - E_error (np.array): Array of energy error values.
        - error_files (list): List of file paths for error data.
        - ddv (np.array): Array representing variation in some parameter (e.g., number of photons (normalized ot the max)).
        - window_size (int): The size of the moving window for smoothing (default is 200).
        """
        self.E_error = E_error
        self.ddv = ddv
        self.error_matrix = np.zeros((len(ddv), len(E_error)))  # Initialize error matrix
        self.window_size = window_size

        # Load error data from provided files
        self.error1 = pd.read_csv(error_files[0], header=None).values[:, 0]
        self.error2 = pd.read_csv(error_files[1], header=None).values[:, 0]
        self.error3 = pd.read_csv(error_files[2], header=None).values[:, 0]
        self.error4 = pd.read_csv(error_files[3], header=None).values[:, 0]

        # Create the error matrix based on ddv and energy error levels
        self.create_error_matrix()

        # Smooth the error matrix
        self.error_matrix = self.smooth_2d_array(self.error_matrix, self.window_size)

    def create_error_matrix(self):
        """Fill the error matrix based on the error levels and ddv values."""
        # Fill the matrix using different error levels depending on the ddv value
        for i, xi in enumerate(self.E_error):
            for j, dv in enumerate(self.ddv):
                if dv > 0.1:
                    self.error_matrix[j, i] = self.error1[i]
                elif dv > 0.01:
                    self.error_matrix[j, i] = self.error2[i]
                elif dv > 0.001:
                    self.error_matrix[j, i] = self.error3[i]
                else:
                    self.error_matrix[j, i] = self.error4[i]

    @staticmethod
    def smooth_2d_array(data, window_size, axis=0):
        """
        Smooth a 2D array along a specified axis using a moving average.

        Parameters:
        - data (np.array): The input 2D array.
        - window_size (int): The size of the moving window (number of elements to average).
        - axis (int): The axis along which to smooth the data (0 for rows, 1 for columns).

        Returns:
        - np.array: The smoothed 2D array.
        """
        # Construct the moving average filter
        filter = np.ones(window_size) / window_size

        # Apply the convolution along the specified axis
        return np.apply_along_axis(lambda m: np.convolve(m, filter, mode='same'), axis=axis, arr=data)

    def get_errors(self, x_vals, data_vals):
        """
        Interpolate errors from the matrix based on given data and x values.

        Parameters:
        - x_vals (np.array): The x values for interpolation (e.g., energy values).
        - data_vals (np.array): The corresponding data values (e.g., signal values).

        Returns:
        - np.array: The interpolated errors.
        """
        # Ensure x_vals and data_vals are numpy arrays for vectorized operations
        x_vals = np.array(x_vals).flatten()
        data_vals = np.array(data_vals).flatten()
        data_vals_max = np.amax(data_vals)
        data_vals /= data_vals_max
        # Initialize an array for results
        errors = np.zeros_like(x_vals)
        
        # Create an interpolator function for the error matrix
        interp_func = RegularGridInterpolator((self.ddv, self.E_error), self.error_matrix, bounds_error=True, fill_value=True)
        
        # Iterate over each pair of (data_val, x_val) and interpolate to find errors
        for i in range(len(x_vals)):
            try:
                errors[i] = interp_func(np.array([[data_vals[i], x_vals[i]]]))
            except:
                
                errors[i] = np.nan  # Set error to NaN if interpolation fails
        
        return errors
    
    def plot_error_results(self, E_guess, signal, y_errors, std):
        """
        Plot the error results including the unfolded spectrum and theoretical lines.

        Parameters:
        - E_guess (np.array): Array of guessed energy values.
        - signal (np.array): Signal data to plot.
        - y_errors (np.array): Interpolated errors for the y-axis.
        - std (float): Standard deviation for error bars.
        """
        # Plot settings and initialization
        plt.rcParams.update({'font.size': 24})
        plt.figure(figsize=(10, 8))

        
        
        # Calculate logarithmic differences for x-axis scaling
        log_diff = np.diff(np.log10(E_guess))
        min_log_diff = np.array([np.amin(log_diff)])
        diff_log10 = np.concatenate((min_log_diff, log_diff))
        signal /= diff_log10
        
        # Calculate relative errors for error bars with std the error over the stack unfolding data and y_errors the error on the numerical unfolding method
        relative_error = y_errors + std
        
        y_err_lower = signal - signal / (1 + relative_error) #Error for the errorbar plot
        y_err_upper = signal * relative_error #Error for the errorbar plot
        y_err_lower3 = signal / (1 + 3*relative_error) # Error for the gray area plot 
        
        
        # Plot the unfolded spectrum with error bars
        plt.errorbar(E_guess, signal, yerr=[y_err_lower, y_err_upper], color='k', ecolor='b', capsize=5, label='Unfolded')
        
        # Fill between error bounds
        plt.fill_between(E_guess, 3 * y_err_upper + signal, y_err_lower3, color='gray', alpha=0.2, label=r'3 $\sigma$ error')

        # Configure plot axes and labels
        plt.ylabel(r'$dN/dE_{log10}$')
        plt.xlabel('E (MeV)')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend(loc='upper left', fontsize=19)
        plt.show()

    
    

def main():
    # Initialize DataProcessor to handle data reading and processing
    data_processor = DataProcessor()
    '''User Input required'''
    '''
    Provide here the experimental data you have. Either image or custom function for passive stack
    # Read experimental data from the image
    Exp_FLUKA = data_processor.read_image()
    Exp_max = np.amax(Exp_FLUKA)  # Find maximum value in experimental data for normalization
    Exp_FLUKA = Exp_FLUKA.reshape(1, -1)  # Reshape experimental data array
    Exp_FLUKA /= Exp_max  # Normalize experimental data
    '''
    # Import response matrix from files
    '''User Input required'''
    '''
    Please provide the following variables 
    
    - Spectrum_tot : Spectrum of used for the RM calculation, can be used to be multiplied to spectrum not implemented yet. Shape is ((N_data,N_spectrum))
    - FLUKA_tot : Response Matrix calculated from the Monte-Carlo code and calculate energy depostion for mono-energetic photons (or quasi-mono-energetic). Must be normalized to the max along N_FLUKA axis, ie each deposited energy pattern is normalized to the max. Shape is ((N_data,N_FLUKA))
    - FLUKA_fact : Normalizing factor of FLUKA_tot. Can be calculated by np.amax(FLUKA_tot,axis = 1).
    
    
    Spectrum_tot, FLUKA_tot, FLUKA_fact, E = data_processor.import_RM()
    '''

    # Generate energy guess range for the optimization process
    E_guess = np.logspace(*Config.E_guess_range, Config.N_guess)

    # Initialize Optimizer with necessary data for CMA-ES optimization
    optimizer = Optimizer(E, E_guess, FLUKA_tot, FLUKA_fact, Exp_FLUKA, Config.smooth_factor)

    # Run the optimization to find the best-fit parameters
    sim = optimizer.run_CMA()

    # Calculate the simulated FLUKA spectrum based on optimization results
    FLUKA_sim = optimizer.calc_FLUKA(sim / np.sum(sim))

    # Plot experimental and simulated results
    Plotter.plot_results(Exp_FLUKA.T, FLUKA_sim)
    Plotter.plot_spectrum(E_guess, sim)

    # Calculate and print statistical information about the fit
    M = np.mean(Exp_FLUKA * Exp_max / FLUKA_sim)
    S = np.std(Exp_FLUKA * Exp_max / FLUKA_sim)
    rel_error = M/S
    print(f'N: {M} +- {S} ')

    # Normalize the signal for error analysis
    signal = sim / np.amax(sim) * M
    
    # Instantiate the ErrorAnalysis class
    error_analysis = ErrorAnalysis(Config.E_error, Config.error_files, Config.ddv)
    
    
    '''User Input required'''
    '''
    Run the Calc_errors.py to determine the error in the unfolding of your RM you can put 1 for test
    
    # Calculate errors using the interpolator
    y_errors = error_analysis.get_errors(E_guess.reshape(-1, 1), signal)
    y_errors = 1
    '''
    # Plot the error analysis results
    error_analysis.plot_error_results(E_guess, signal, y_errors, rel_error)

# Run the main function when the script is executed
if __name__ == "__main__":
    main()

