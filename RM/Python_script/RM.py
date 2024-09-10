# -*- coding: utf-8 -*-
"""
Script for simulating and processing spectral data.
Created on Fri Apr 28 10:56:58 2023
@author: gaetan.fauvel
"""

import numpy as np
import os
import subprocess
import pandas as pd
import glob
from RM_func import Sim_spectrum, Least_squared
import variables

print('Start script', flush=True)

# Cleanup existing output files from previous runs
for file_delete in glob.glob(variables.folder_path_exp + '/*' + 'fort*'):
    os.remove(file_delete)  # Remove temporary FLUKA output files

# Remove old control files indicating the progress or completion of the process
if os.path.exists(variables.folder_path_exp + '/Ec.txt'):
    os.remove(variables.folder_path_exp + '/Ec.txt')

if os.path.exists(variables.folder_path_exp + "/Finished.txt"):
    os.remove(variables.folder_path_exp + "/Finished.txt")

# Main simulation loop starting from i_start to max_iterations, step by i_ind
for iteration in range(variables.i_start, variables.max_iterations, variables.i_ind):
    print(Sim_spectrum.Ec_liste[iteration], flush=True)

    # Skip the loop if control file 'Finished.txt' exists
    if variables.i_start != 0 and iteration < variables.i_start:
        continue

    if os.path.exists(variables.folder_path_exp + "/Finished.txt"):
        os.remove(variables.folder_path_exp + "/Finished.txt")
        break

    print('Iteration number:', iteration + 1, flush=True)
    
    # Calculate the simulated spectrum for the current energy center
    Simulated_spectrum = Sim_spectrum.Gaussian(Sim_spectrum.E, Sim_spectrum.Ec_liste[iteration], variables.sigma_percent)

    # If the spectrum is mostly zeros, skip to the next iteration
    if np.count_nonzero(Simulated_spectrum == 0) > variables.N_sim_spectrum - 2:
        continue

    # Save the spectrum and its corresponding energy and center energy values
    Sim_spectrum.save_spectrum(Simulated_spectrum)
    pd.DataFrame(Sim_spectrum.E).to_csv(variables.folder_path_exp + '/Sim_spectrum_energy.txt', header=None, index=None)
    pd.DataFrame([Sim_spectrum.Ec_liste[iteration]]).to_csv(variables.folder_path_exp + '/Ec.txt', header=None, index=None)
    pd.DataFrame(Simulated_spectrum).to_csv(variables.folder_path_exp + '/Sim_spectrum.txt', header=None, index=None)

    # Execute the FLUKA simulation command
    subprocess.run(variables.cmd_FLUKA, shell=True)
    
    # Process and analyze the simulation results
    Least_squared.boucle()
    
    # Load results from the simulation processing
    FLUKA_spectrum_data = pd.read_csv("FLUKA_boucle.txt", header=None).values
    Spectrum_theory = np.array(pd.read_csv(variables.folder_path_exp + '/Sim_spectrum.txt', header=None))
    E_liste_data = Sim_spectrum.E
    Ec = pd.read_csv(variables.folder_path_exp + "/Ec.txt", header=None)[0][0]

    # Save the processed data for this iteration into the response folder
    pd.DataFrame(Spectrum_theory).to_csv(variables.folder_path_response + "/" + str(Ec) + "_Spectrum.txt", header=None, index=None)
    pd.DataFrame(FLUKA_spectrum_data).to_csv(variables.folder_path_response + "/" + str(Ec) + "_FLUKA.txt", header=None, index=None)
    pd.DataFrame(E_liste_data).to_csv(variables.folder_path_response + "/" + str(Ec) + "_Energy.txt", header=None, index=None)
