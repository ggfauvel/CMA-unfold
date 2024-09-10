''' Script variables '''

#Energy spectrum for the FLUKA input spectrum as FLUKA do not take mono-energetic spectrum, we create a very narrow gaussian shape
N_sim_spectrum = 2000# NUmber of points for the simulated spectrum
Sim_spectrum_start = 0.005#Energy start for the simulated spectrum in MeV
Sim_spectrum_end = 2000# Energy stop for the simulated spectrum in MeV


# Mono-energetic bins for the RM
Ec_fit_max =1000#Maximum energy to look for in MeV
Ec_fit_min   = 0.01#Minimum energy to look for in MeV
log = True # If the loop for the RM is in logspace 
max_iterations = 500#Maximum number of iterations for the RM

'''FLUKA variables'''
z_intervals_cm = [[0.2, 2.2], [2.4, 4.4]] + [[4.6 + i * 0.7, 5.1 + i * 0.7] for i in range(15)] #Position of the detectors inside your simulation along the z axis
N_z = 800 #Number of points inside the USRBIN along the Z axis
z_min = 0 # Starting of the USRBIN in the Z axis
z_max = 20 # Stop of the USRBIN in the Z axis
N_detect = 17 # Number of detectors inside your FLUKA simulation
Number_cycle_FLUKA = 8# Number of cycle FLUKA for each iteration. Minimum recommended 3. Must be integer inferior strict to 10
name_INP = 'Test'# name of the input FLUKA

'''Path variables'''
folder_path_exp = '/HOME/FLUKA/CMA-unfold' #Main folder to work with, must be current working directory
folder_path_response = folder_path_exp + "/Response_matrix" #Folder where the Response Matrix (RM) will be saved
folder_PYTHON_script = folder_path_exp+'/Python_script'#Folder where are located the python script
exe_path = '$FLUPRO/Synchrotron'
cmd_FLUKA = "$FLUPRO/bin/rfluka -M"+ str(Number_cycle_FLUKA)+" -e  exe_path "+folder_path_exp + '/'+name_INP+'.inp'#Command given to terminal for FLUKA launch


''' Cluster variables'''
# Used for cluster. The index will vary by step of 10 during the loop over the energies if i_ind = 10. 
#This way you can start the script over 10 nodes, still it must be launched independently.
i_ind = 1 
i_start = 0# Start index, same used for cluster. Each node should start at a different start.

'''Do not change anything after this'''
q = 1.609e-19
sigma_percent = 0.005