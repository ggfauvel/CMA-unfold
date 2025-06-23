# CMA-ES Spectrum Unfolding

**Welcome to the CMA-ES Spectrum Unfolding project!** This project provides a robust framework for unfolding particle spectra using the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) created by Hansen et al. By leveraging data from Monte-Carlo simulations and experimental measurements, this script can effectively reconstruct the underlying particle energy distribution.

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation and Usage](#installation-and-usage)
4. [Expert Usage](#expert-usage)
5. [Results of Calibrated Spectrometer](#results-of-calibrated-spectrometer)
6. [How to Cite](#how-to-cite)
7. [Contact Information](#contact-information)
8. [Acknowledgments](#acknowledgments)
9. [License](#license)

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
- **Response Matrix (RM) calculation:** This script uses FLUKA. You have to fill the RM_variables.py script. It uses the Test.inp. You must launch the script while being in the folder of the .inp, here for example 
```bash
cd RM
```
```bash
python3 Python_script/RM.py
```
Using flair, you must compile a custom executable using the source_final.f in the RM folder.

- **Error Calculation Script:** Use `Calc_errors.py` to evaluate the accuracy and reliability of the unfolding results. This script analyzes the uncertainty in the unfolded spectrum, providing a detailed error profile.
## Results of Calibrated Spectrometer

This section presents the calibration results of a stacking scintillator calorimeter using a Co-60 radioactive source. Calibration with Co-60, a commonly used radioactive source, provides two distinct gamma-ray peaks at energies of 1.17 MeV and 1.33 MeV. These calibration points are critical for accurately interpreting the energy response of the spectrometer.

### Calibration Lines

The calibration of the spectrometer is visualized by plotting the detected signal against the known gamma-ray energies from the Co-60 source:

- **Co-60 Energy Lines**: Two prominent lines at 1.17 MeV and 1.33 MeV are used to calibrate the spectrometer's energy scale. These lines are indicative of the spectrometer's ability to resolve distinct energy peaks accurately.

Below are some images showing the calibration process and the results:

1. **Raw Data Visualization**: A plot showing the raw output from the scintillator array when exposed to the Co-60 source.

<img src="images/raw_data.tiff" alt="Raw Data Visualization" width="300"/>

2. **Calibrated Spectrum**: A comparison of the theoretical data and the unfolding.

<img src="images/spectrum.png" alt="Calibrated Spectrum" width="500"/>

3. **Calibrated Spectrum**: More precise spectrum

It is possible to achieve a better precision on 'mono-energetic' spectrum but cannot be extrapolated to continuous distribution. It is up to the user to define the needs of the detector.

<img src="images/Precise_Spectro_A.png" alt="Calibrated Spectrum" width="300"/>

These images provide a clear view of the spectrometer's calibration, demonstrating its capability to accurately detect and resolve gamma-ray peaks from radioactive sources like Co-60.


### Tips to make the dream work
1. The more detectors you have, the more precise you need to be with your .inp input as a small deviation from reality accumulates quickly. EVERY element close to the detector must be included.
2. When using the algorithm for peaks solution, you can use a smoothing factor close to zero it finds peaks very accurately but then struggles for continuous distributions.
3. The experimental setup must be included inside the FLUKA simulation if no/low shielding used or a long detector is used without pinhole.


## How to Cite

If you use this code in your research or publication, please cite it as follows:

G. Fauvel, K. Tangtartharakul, A. Arefiev, J. De Chant, S. Hakimi, O. Klimo, M. Manuel, A. McIlvenny, K. Nakamura, L. Obst-Huebl, P. Rubovic, S. Weber, F. P. Condamine; Compact in-vacuum gamma-ray spectrometer for high-repetition rate PW-class laser–matter interaction. Rev. Sci. Instrum. 1 February 2025; 96 (2): 023102. https://doi.org/10.1063/5.0206348

Alternatively, you can use the following BibTeX entry for LaTeX users:

```bibtex
@article{10.1063/5.0206348,
    author = {Fauvel, G. and Tangtartharakul, K. and Arefiev, A. and De Chant, J. and Hakimi, S. and Klimo, O. and Manuel, M. and McIlvenny, A. and Nakamura, K. and Obst-Huebl, L. and Rubovic, P. and Weber, S. and Condamine, F. P.},
    title = {Compact in-vacuum gamma-ray spectrometer for high-repetition rate PW-class laser–matter interaction},
    journal = {Review of Scientific Instruments},
    volume = {96},
    number = {2},
    pages = {023102},
    year = {2025},
    month = {02},
    issn = {0034-6748},
    doi = {10.1063/5.0206348},
    url = {https://doi.org/10.1063/5.0206348},
    eprint = {https://pubs.aip.org/aip/rsi/article-pdf/doi/10.1063/5.0206348/20383725/023102\_1\_5.0206348.pdf},
}
```

## Contact Information

For further information, questions, or collaboration, please contact:

**Fauvel Gaëtan**  
Email: [fauvel.gaetan@outlook.com](mailto:fauvel.gaetan@outlook.com)

## Acknowledgments

We wish to acknowledge the support of the National Sci-
ence Foundation (NSF Grant No. PHY-2206777) and the Czech Science Foundation
(GA ČR) for funding on project number No. 22-42890L in the frame of the National Science Foundation–Czech Science Foundation partnership.

## Third-Party Licenses

This project uses the `py-cma` library, which is licensed under the BSD 3-Clause License.  

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

---

Thank you for using the CMA-ES Spectrum Unfolding project! We hope it serves your research and analytical needs.
