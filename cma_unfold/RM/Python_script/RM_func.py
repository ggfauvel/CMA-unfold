import numpy as np
import os
import scipy.stats as stats
import pandas as pd
import RM_variables as variables

class Least_squared:
    @staticmethod
    def boucle():
        """Parse data from FLUKA output and save into a CSV file.
        No inputs or outputs handled directly; uses global paths and configurations.
        """
        # Parse data using another class method and directly save to file
        FLUKA_spectrum_data = FLUKA_spectrum.data_parse(variables.folder_path_exp)
        # Write parsed data to a CSV without headers or index
        pd.DataFrame(FLUKA_spectrum_data).to_csv("FLUKA_boucle.txt", header=None, index=None)
        return

class Sim_spectrum:
    # Generate energy values using logarithmic or linear spacing based on a log flag
    E = np.logspace(np.log10(variables.Sim_spectrum_start), np.log10(variables.Sim_spectrum_end), variables.N_sim_spectrum)
    if not variables.log:
        Ec_liste = np.linspace(variables.Ec_fit_min, variables.Ec_fit_max, variables.max_iterations)
    else:
        Ec_liste = np.logspace(np.log10(variables.Ec_fit_min), np.log10(variables.Ec_fit_max), variables.max_iterations)

    @staticmethod
    def Gaussian(E, Ec, sigma_percent):
        """Calculate Gaussian distribution for energy calibration.
        
        Args:
        - E (float): Incident energy
        - Ec (float): Central energy around which to center the Gaussian
        - sigma_percent (float): Width of the Gaussian as a percentage of Ec
        
        Returns:
        - y (ndarray): Normalized Gaussian distribution values
        """
        # Convert energy to a higher scale (from MeV to keV) for precision
        E *= 1e6 * variables.q
        # Calculate the standard deviation based on the percentage
        sigma = Ec * sigma_percent * variables.q * 1E6
        # Adjust center energy similarly
        Ec *= variables.q * 1e6
        # Calculate the Gaussian probability density function
        y = stats.norm.pdf(E, Ec, sigma)
        # Normalize the distribution so the sum is 1
        return y / np.sum(y)

    @staticmethod
    def save_spectrum(Values):
        """Save the simulated spectrum to CSV files.
        
        Args:
        - Values (ndarray): Calculated spectrum values to save
        """
        # Dictionary to hold the energy values (converted to keV) and the spectrum
        dictr_b = {
            "x": Sim_spectrum.E / 1e3,
            "y": Values
        }
        # Save the spectrum values with no headers and indexed by column names
        pd.DataFrame(dictr_b).to_csv(variables.folder_path_exp + "/Spectrum.txt", header=None, columns=["y"], index=False)
        # Save a second version with tabs, including both energy and spectrum values
        pd.DataFrame(dictr_b).to_csv(variables.folder_path_exp + "/Spectrum_b.txt", header=None, columns=["x", "y"], index=False, sep="\t")

class FLUKA_spectrum:
    @staticmethod
    def read_files(file_path, Nz, plot=False, integrate=False):
        """Read raw FLUKA simulation output and process into structured data.
        
        Args:
        - file_path (str): Path to the FLUKA output file
        - Nz (int): Number of depth slices in the simulation
        
        Returns:
        - x (list): Structured data array from the raw output
        """
        # Open and read the entire file
        with open(file_path, 'r') as file:
            content = file.readlines()
        data_rows = []
        start_processing = False
        # Parse lines after the marker indicating the start of relevant data
        for line in content:
            if "accurate deposition" in line:
                start_processing = True
                continue
            if start_processing and line.strip():
                # Convert space-separated values to float and store
                row = list(map(float, line.strip().split()))
                if row:
                    data_rows.append(row)
        # Reshape the raw flat list into a structured 3D array
        matrix = np.array(data_rows).reshape((Nz, 100, 100))
        return matrix.tolist()

    @staticmethod
    def data_parse(folder_path, plot=False):
        """Aggregate and process multiple FLUKA simulations for statistical analysis.
        
        Args:
        - folder_path (str): Directory containing the simulation outputs
        - plot (bool): Flag to enable plotting (functionality not implemented here)
        
        Returns:
        - Spectrum_FLUKA (ndarray): Averaged spectrum over multiple simulations
        """
        # Initialize an array to hold aggregated spectral data
        Spectrum_FLUKA = np.zeros((variables.N_detect,))
        # Process each simulation output file
        for Cycle_number in range(1, variables.Number_cycle_FLUKA + 1):
            file_path = os.path.join(folder_path, variables.name_INP + "00" + str(Cycle_number) + "_fort") + ".21"
            if os.path.isfile(file_path):
                # Read the structured data from the file
                Spectrum_tot = FLUKA_spectrum.read_files(file_path, variables.N_z)
                # Linear spacing for z-axis
                z_linspace = np.linspace(variables.z_min, variables.z_max, variables.N_z)
                # Determine z-interval indices for integration
                z_intervals_idx = [[int(np.argmin(np.abs(z_linspace - z_start))), int(np.argmin(np.abs(z_linspace - z_end)))] for z_start, z_end in variables.z_intervals_cm]
                # Average over x, y dimensions and selected z-intervals
                sum_over_xy = np.mean(np.mean(Spectrum_tot, axis=1), axis=1)
                average_over_z_intervals = []
                for z_start_idx, z_end_idx in z_intervals_idx:
                    average_over_z_intervals.append(np.mean(sum_over_xy[z_start_idx:z_end_idx + 1]))
                # Aggregate the results from this file
                Spectrum_FLUKA += np.array(average_over_z_intervals)
        # Normalize by the number of cycles to get the average spectrum
        Spectrum_FLUKA /= variables.Number_cycle_FLUKA
        return Spectrum_FLUKA
