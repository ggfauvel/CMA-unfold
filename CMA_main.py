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
import os
import re
import matplotlib.pyplot as plt
import cma
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm
class Config:
    """Configuration parameters for the analysis."""
    N_guess = 25  # Number of energy guesses for the optimization. Be carefull highly non-linear ! Do not go above 200 points on local machine !
    
    E_guess_range = (np.log10(5e-2), np.log10(100))  # Energy range for guesses (log scale)
    E_guess = np.logspace(*E_guess_range, N_guess)
    smooth_factor = 1.3e-5 * np.mean(np.diff(np.log10(E_guess)))/100  # Smoothing factor for the optimization objective
    
    lower_bound = -10  # Lower bound for optimization variables
    upper_bound = 10  # Upper bound for optimization variables
    
    folder_path = "./RM/Response_matrix/"  # Path to response matrix files
    image_path = "./images/raw_data.tiff"  # Path to image file for analysis

    # Region of Interest (ROI) for analyzing the image data
    ROI = np.array([[524+60, 649, 933, 1121], [709+30, 828, 933, 1104], [866, 901, 926, 1114], [940, 964, 915, 1118], 
                    [1006, 1034, 915, 1121], [1080, 1111, 912, 1111], [1157, 1185, 915, 1125], [1227, 1255, 908, 1125], 
                    [1283, 1321, 905, 1121], [1356, 1387, 915, 1118], [1422, 1454, 915, 1118], [1489, 1524, 919, 1118], 
                    [1562+5, 1594, 908, 1111], [1632, 1664, 915, 1104], [1702, 1730, 919, 1107], [1772, 1800, 915, 1118], 
                    [1828, 1860, 919, 1111]])  # ROI coordinates for scintillators

    N_sim = 2000  # Number of bin inside the spectrum used for the FLUKA simulation
    N_FLUKA = 17  # Number of slices/detectors inside the stack
    N_data = 2000  # Number of data points inside the response matrix, ie. total number of FLUKA simulations performed

    # Calibration factors for the different scintillator regions
    '''USER INPUT NEEDED'''
    ''' Please provide the calibration factor calculated for your detector. These factors will be multiplied to the FLUKA output. If the RM is already good put 1.
    '''
    factor = np.array([ 107.01256591,  114.32747207,  250.14450456,  318.13876634,
            297.9812333 ,  269.50250641,  244.36715103,  235.70913968,
            243.62062148,  265.55290113,  224.2890357 ,  269.87905736,
           1826.3594274 , 1736.53814584, 2016.53732247, 1443.53111315,
           1892.13289656])
    
    # Initialize ErrorAnalysis with error data and parameters. 
    error_files = ['./Error/mean1_VAC.txt', './Error/mean2_VAC.txt', './Error/mean3_VAC.txt', './Error/mean4_VAC.txt']
    E_error = np.array([5.00000000e-02, 6.41142715e-02, 8.34786374e-02, 1.07662418e-01,
           1.38766479e-01, 1.78856616e-01, 2.25271064e-01, 2.97129582e-01,
           3.82971391e-01, 4.82354943e-01, 6.36219849e-01, 8.20026062e-01,
           1.05693455e+00, 1.36228676e+00, 1.75585633e+00, 2.26312957e+00,
           2.85042632e+00, 3.75967497e+00, 4.84585864e+00, 6.24584471e+00,
           8.05029181e+00, 1.03760502e+01, 1.33737286e+01, 1.72374471e+01,
           2.22174078e+01, 2.86360972e+01, 3.60673495e+01, 4.75723614e+01,
           6.13161884e+01, 7.90306568e+01, 1.01862899e+02, 1.31291459e+02,
           1.69222035e+02, 2.18110892e+02, 2.81123917e+02, 3.62341632e+02,
           4.56371628e+02, 6.01948197e+02, 7.75853206e+02, 9.77192128e+02])
    ddv = np.logspace(-10,0,1000)  # Array representing variation in parameter (e.g., detector distance)

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
                print(array)    
                print('E bin not in RM')  # Error message if value is out of range
                raise ValueError
            used_indices.add(index)
            indices.append(index)

        return np.array(indices)

    @staticmethod
    def read_image(norm=True,plot = False):
        """Read and process image data."""
        im = Image.open(Config.image_path)  # Open the image
        im = np.array(im)  # Convert image to numpy array
        
        exp_spectrum_image = []
        # Process each ROI to extract the scintillator values
        for y_min, y_max, x_min, x_max in zip(Config.ROI[:, 0], Config.ROI[:, 1], Config.ROI[:, 2], Config.ROI[:, 3]):
            ROI_image = im[x_min:x_max, y_min:y_max]  # Extract the ROI from the image
            value_scint = np.sum(ROI_image)  # Sum the pixel values in the ROI
            if plot == True:
                plt.imshow(ROI_image)
                plt.show()
            if norm == True:
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
        # Initialize arrays
        Spectrum_tot, FLUKA_tot, FLUKA_fact = cls.initialize_array(Config.N_data, Config.N_sim)
        name_list = np.zeros(Config.N_data, dtype=float)
    
        # Grab all files in the folder
        all_files = glob.glob(os.path.join(Config.folder_path, '*'))
        # Keep only the Spectrum files and sort them
        spectrum_files = sorted(
            [f for f in all_files if os.path.basename(f).endswith('_Spectrum.txt')],
            key=cls.sort_numerically
        )
    
        for idx, spec_path in enumerate(spectrum_files):
            base = os.path.basename(spec_path)
            # Extract the 'name' as everything before "_Spectrum"
            m = re.match(r'(.+?)_Spectrum', base)
            if not m:
                print(f"Couldn't parse name from '{base}' — skipping.")
                continue
            name = m.group(1)
    
            # Build the corresponding filenames
            energy_path = os.path.join(os.path.dirname(spec_path), f"{name}_Energy.txt")
            fluka_path  = os.path.join(os.path.dirname(spec_path), f"{name}_FLUKA.txt")
    
            # Read energy
            try:
                Energy = pd.read_csv(energy_path, header=None)
            except FileNotFoundError:
                print(f"Energy file missing for '{name}' — skipping.")
                continue
    
            # Read and normalize FLUKA
            FLUKA = pd.read_csv(fluka_path, header=None).iloc[:, 0].to_numpy()
            FLUKA *= Config.factor
            peak = FLUKA.max()
            if peak == 0:
                print(f"Empty FLUKA data for '{name}' — skipping.")
                continue
            FLUKA /= peak
    
            # Store into arrays
            
            FLUKA_tot[idx, :]    = FLUKA
            FLUKA_fact[idx]      = peak
            name_list[idx]       = float(name)
    
        # Sort everything by ascending 'name'
        order = np.argsort(name_list)
        E            = name_list[order]
        Spectrum_tot = Spectrum_tot[order, :]
        FLUKA_tot    = FLUKA_tot[order, :]
        FLUKA_fact   = FLUKA_fact[order]
    
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
            'verb_disp': 1000,  # Display every X iterations
            'tolx': 1e-6  # Tolerance for stopping criteria
        }

        initial_mean = np.random.uniform(Config.lower_bound, Config.upper_bound, Config.N_guess)  # Initial guess, random
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
        # Example function to fill the error matrix
        # Create a 2D matrix of error values
        error_matrix = np.zeros((len(self.ddv), len(self.E_error)))
        #print(error_matrix.shape)
        for i, xi in enumerate(self.E_error):
            for j, dv in enumerate(self.ddv):
                if dv > 0.1:
                    error_matrix[j, i] = self.error1[i] if not np.isnan(self.error1[i]) else 1
                elif dv > 0.01:
                    error_matrix[j, i] = self.error2[i] if not np.isnan(self.error2[i]) else 1
                elif dv>0.001:
                    error_matrix[j, i] = self.error3[i] if not np.isnan(self.error3[i]) else 1
                else :
                    error_matrix[j, i] = self.error4[i] if not np.isnan(self.error4[i]) else 1
        error_matrix = self.smooth_2d_array(error_matrix,200)
        return error_matrix

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
        
        
        
        deltaE = np.diff(E_guess)   
        
        # total counts over all (used) bins
        N_tot_exp = np.sum(signal)
        scale_exp = N_tot_exp / np.sum(signal[:-1] * deltaE)
        
        # true dN/dE in particles/MeV
        signal =signal[:-1] * scale_exp
        dN = signal*deltaE
        N_error = np.sqrt(dN)/dN
        # Calculate relative errors for error bars with std the error over the stack unfolding data and y_errors the error on the numerical unfolding method
        relative_error = y_errors[:-1] + std  + N_error
        
        y_err_lower = signal - signal / (1 + relative_error) #Error for the errorbar plot
        y_err_upper = signal * relative_error #Error for the errorbar plot
        y_err_lower3 = signal / (1 + 3*relative_error) # Error for the gray area plot 
        
        
        # Plot the unfolded spectrum with error bars
        plt.errorbar(E_guess[:-1], signal, yerr=[y_err_lower, y_err_upper], color='k', ecolor='b', capsize=5, label='Unfolded')
        
        # Fill between error bounds
        plt.fill_between(E_guess[:-1], 3 * y_err_upper + signal, y_err_lower3, color='gray', alpha=0.2, label=r'3 $\sigma$ error')

        # Configure plot axes and labels
        plt.ylabel(r'$dN/dE (MeV^{-1})$')
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
    '''
    Exp_FLUKA = data_processor.read_image()
    Exp_max = np.amax(Exp_FLUKA)  # Find maximum value in experimental data for normalization
    Exp_FLUKA = Exp_FLUKA.reshape(1, -1)  # Reshape experimental data array
    Exp_FLUKA /= Exp_max  # Normalize experimental data
    
    # Import response matrix from files
    '''User Input required'''
    '''
    Please provide the following variables 
    
    - Spectrum_tot : Spectrum of used for the RM calculation, can be used to be multiplied to spectrum not implemented yet. Shape is ((N_data,N_spectrum))
    - FLUKA_tot : Response Matrix calculated from the Monte-Carlo code and calculate energy depostion for mono-energetic photons (or quasi-mono-energetic). Must be normalized to the max along N_FLUKA axis, ie each deposited energy pattern is normalized to the max. Shape is ((N_data,N_FLUKA))
    - FLUKA_fact : Normalizing factor of FLUKA_tot. Can be calculated by np.amax(FLUKA_tot,axis = 1).
    '''
    
    Spectrum_tot, FLUKA_tot, FLUKA_fact, E = data_processor.import_RM()
    

    
    

    # Initialize Optimizer with necessary data for CMA-ES optimization
    optimizer = Optimizer(E, Config.E_guess, FLUKA_tot, FLUKA_fact, Exp_FLUKA, Config.smooth_factor)

    # Run the optimization to find the best-fit parameters
    sim = optimizer.run_CMA()

    # Calculate the simulated FLUKA spectrum based on optimization results
    FLUKA_sim = optimizer.calc_FLUKA(sim / np.sum(sim))

    # Plot experimental and simulated results
    Plotter.plot_results(Exp_FLUKA.T, FLUKA_sim)
    Plotter.plot_spectrum(Config.E_guess, sim)

    # Calculate and print statistical information about the fit
    M = np.mean(Exp_FLUKA * Exp_max / FLUKA_sim)
    S = np.std(Exp_FLUKA * Exp_max / FLUKA_sim)
    rel_error = S/M
    print(f'N: {M} +- {S} ')

    # Normalize the signal for error analysis
    signal = sim / np.amax(sim) * M
    
    # Instantiate the ErrorAnalysis class
    error_analysis = ErrorAnalysis(Config.E_error, Config.error_files, Config.ddv)
    
    
    '''User Input required'''
    '''
    Run the Calc_errors.py to determine the error in the unfolding of your RM you can put 1 for test
    '''
    # Calculate errors using the interpolator
    y_errors = error_analysis.get_errors(Config.E_guess.reshape(-1, 1), signal)
    #y_errors = 1
    
    
   
    # Plot the error analysis results
    error_analysis.plot_error_results(Config.E_guess, signal, y_errors, rel_error)

# Run the main function when the script is executed
if __name__ == "__main__":
    main()

