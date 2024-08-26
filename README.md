# CMA-ES Spectrum Unfolding

**Welcome to the CMA-ES Spectrum Unfolding project!** This project provides a robust framework for unfolding particle spectra using the Covariance Matrix Adaptation Evolution Strategy (CMA-ES). By leveraging data from Monte-Carlo simulations and experimental measurements, this script can effectively reconstruct the underlying particle energy distribution.

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation and Usage](#installation-and-usage)
4. [Expert Usage](#expert-usage)
5. [Contact Information](#contact-information)
6. [License](#license)

## Overview

This project utilizes CMA-ES to unfold a particle spectrum from an array of deposited energy values obtained through Monte-Carlo simulations. The method is tailored for scenarios where experimental data is captured using instruments like scintillators, providing a way to reconstruct the original energy distribution of particles.

## Requirements

To run this project, ensure you have the following Python packages installed:
```bash
pip3 install numpy pandas cma glob re matplotlib.pyplot scipy.interpolate scipy.stats
```


These libraries provide essential functionalities for numerical computations, data handling, optimization, and visualization.

## Installation and Usage

Follow these steps to set up and use the project:

1. **Clone or Download the Repository:**

   Clone the repository using Git:
```bash
git clone https://github.com/ggfauvel/CMA-unfold.git
```

Alternatively, you can download the repository as a ZIP file and extract it.

2. **Prepare Your Data:**

- **Response Matrix:** Provide a response matrix calculated from a Monte-Carlo code. This matrix should be shaped as `((N_bin, N_dep))`, where:
  - `N_bin` is the number of different mono-energetic particle energies used in the simulation.
  - `N_dep` is the number of experimental data points (e.g., number of imaging plates or scintillators).

- **Experimental Data:** Supply the experimental data named `Exp_FLUKA` in the shape `((N_dep,))`, representing the deposited energy data points.

3. **Run the Script:**

Execute the main script to perform the spectrum unfolding. The script will process the data, perform optimization using CMA-ES, and visualize the results. Key variable to observe is `sim`, which represents the unfolded spectrum.
```bash
python spectrum_analysis.py
```

This will generate plots comparing experimental and simulated data and display the unfolded spectrum with error bounds.

## Expert Usage

For more advanced users, you can calculate the errors associated with the unfolding method using the provided script:

- **Error Calculation Script:** Use `Calc_errors.py` to evaluate the accuracy and reliability of the unfolding results. This script analyzes the uncertainty in the unfolded spectrum, providing a detailed error profile.

## Contact Information

For further information, questions, or collaboration, please contact:

**Fauvel Gaëtan**  
Email: [gaetan.fauvel@eli-beams.eu](mailto:gaetan.fauvel@eli-beams.eu)

---

Thank you for using the CMA-ES Spectrum Unfolding project! We hope it serves your research and analytical needs effectively.


7. **License**: This script uses the CMA algortihm from nikohansen available at https://github.com/CMA-ES/pycma under the licensing
The BSD 3-Clause License
Copyright (c) 2014 Inria
Author: Nikolaus Hansen, 2008-
Author: Petr Baudis, 2014
Author: Youhei Akimoto, 2016-

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright and
   authors notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   and authors notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided with
   the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors nor the authors names may be used to endorse or promote
   products derived from this software without specific prior written
   permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

This `README.md` file is designed to be comprehensive, giving users a clear understanding of what the project does, how to set it up, and how to use it. Adjust sections as needed to fit the specifics of your project and data.
