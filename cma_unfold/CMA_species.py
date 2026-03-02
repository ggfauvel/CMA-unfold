# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 09:47:04 2026

@author: ggfauvel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# CMA-based Optimizer for Custom ML Training
# Multi-species spectral unfolding from scintillator stack detector data
# using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
#
# The code:
#   1. Imports a response matrix (RM) produced by FLUKA Monte Carlo simulations.
#   2. Reads experimental scintillator signals from a TIFF image.
#   3. Optimises a piecewise log-energy spectrum (1–3 particle species)
#      by minimising a Huber-robust fidelity term, optionally regularised
#      with a smoothness penalty and/or per-detector calibration factors.
#   4. Provides error analysis via interpolation on a pre-computed error matrix.
#
# Author: G. Fauvel | © 2025
# License: PolyForm Noncommercial 1.0.0
# Signature: 34c6bd9b (commit hash seed)
# ----------------------------------------------------------------------------

import numpy as np
import pandas as pd
from PIL import Image
import glob
import os
import re
import matplotlib.pyplot as plt
import cma
from scipy.interpolate import RegularGridInterpolator
from dataclasses import dataclass, field
from typing import List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Global configuration
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    """
    Central configuration for the analysis pipeline.

    All user-tunable parameters (file paths, energy grids, ROI coordinates,
    calibration factors, error file paths) live here.  Modify this class
    to adapt the code to a new experimental setup.
    """

    # ── Primary species (photons) energy grid ────────────────────────────────
    N_guess = 25                                        # Number of spectral bins (CMA variables per species)
    E_guess_range = (np.log10(5e-2), np.log10(100))     # log10(E/MeV) bounds
    E_guess = np.logspace(*E_guess_range, N_guess)      # Bin centroids in MeV

    # Smoothness weight, scaled by mean log-energy spacing
    smooth_factor = 1.3e-5 * np.mean(np.diff(np.log10(E_guess))) * 100

    n_species = 2           # 1 = photon only, 2 = photon + electron
    lower_bound = -10       # CMA lower bound for log10(dN/dE) variables
    upper_bound = 10        # CMA upper bound

    # ── File paths ───────────────────────────────────────────────────────────
    folder_path = "./RM/Response_matrix_double_population/Response_matrix_p/"
    image_path = "./images/raw_data.tiff"

    # ── Region of Interest (ROI) for each scintillator in the stack ──────────
    # Each row: [y_min, y_max, x_min, x_max] in pixel coordinates.
    ROI = np.array([
        [524+60, 649, 933, 1121], [709+30, 828, 933, 1104],
        [866, 901, 926, 1114],    [940, 964, 915, 1118],
        [1006, 1034, 915, 1121],  [1080, 1111, 912, 1111],
        [1157, 1185, 915, 1125],  [1227, 1255, 908, 1125],
        [1283, 1321, 905, 1121],  [1356, 1387, 915, 1118],
        [1422, 1454, 915, 1118],  [1489, 1524, 919, 1118],
        [1562+5, 1594, 908, 1111],[1632, 1664, 915, 1104],
        [1702, 1730, 919, 1107],  [1772, 1800, 915, 1118],
        [1828, 1860, 919, 1111],
    ])

    # ── FLUKA simulation dimensions ──────────────────────────────────────────
    N_sim = 2000      # Number of energy bins per FLUKA simulation
    N_FLUKA = 17      # Number of scintillator slices (detectors) in the stack
    N_data = 2000     # Total number of mono-energetic FLUKA runs (RM rows)

    # ── Per-detector calibration factors ─────────────────────────────────────
    # USER INPUT: multiply FLUKA output by these factors.
    # Set all to 1.0 if the RM is already properly calibrated.
    factor = np.array([
        107.01256591,  114.32747207,  250.14450456,  318.13876634,
        297.9812333,   269.50250641,  244.36715103,  235.70913968,
        243.62062148,  265.55290113,  224.2890357,   269.87905736,
        1826.3594274,  1736.53814584, 2016.53732247, 1443.53111315,
        1892.13289656,
    ])

    # ── Error analysis files and grids ───────────────────────────────────────
    error_files = [
        './Error/mean1_VAC.txt', './Error/mean2_VAC.txt',
        './Error/mean3_VAC.txt', './Error/mean4_VAC.txt',
    ]
    E_error = np.array([
        5.00000000e-02, 6.41142715e-02, 8.34786374e-02, 1.07662418e-01,
        1.38766479e-01, 1.78856616e-01, 2.25271064e-01, 2.97129582e-01,
        3.82971391e-01, 4.82354943e-01, 6.36219849e-01, 8.20026062e-01,
        1.05693455e+00, 1.36228676e+00, 1.75585633e+00, 2.26312957e+00,
        2.85042632e+00, 3.75967497e+00, 4.84585864e+00, 6.24584471e+00,
        8.05029181e+00, 1.03760502e+01, 1.33737286e+01, 1.72374471e+01,
        2.22174078e+01, 2.86360972e+01, 3.60673495e+01, 4.75723614e+01,
        6.13161884e+01, 7.90306568e+01, 1.01862899e+02, 1.31291459e+02,
        1.69222035e+02, 2.18110892e+02, 2.81123917e+02, 3.62341632e+02,
        4.56371628e+02, 6.01948197e+02, 7.75853206e+02, 9.77192128e+02,
    ])
    ddv = np.logspace(-10, 0, 1000)  # Normalised signal intensity axis for error interpolation

    # ── Second species (electrons) ───────────────────────────────────────────
    N_guess_e = 5
    E_guess_range_e = (np.log10(90), np.log10(300))
    E_guess_e = np.logspace(*E_guess_range_e, N_guess_e)
    folder_path_e = "./RM/Response_matrix_double_population/Response_matrix_e/"
    smooth_factor_e = 1.3e-4


# ═══════════════════════════════════════════════════════════════════════════════
# Species configuration dataclass
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpeciesConfig:
    """
    Container for one particle species' response matrix and energy grid.

    After construction, call :meth:`resolve` once to snap ``E_guess`` onto the
    RM energy axis and cache nearest-neighbour indices.

    Parameters
    ----------
    label : str
        Human-readable identifier, e.g. 'photon', 'electron'.
    E : np.ndarray, shape (N_data,)
        Energy axis of the response matrix (sorted ascending, MeV).
    E_guess : np.ndarray, shape (N_guess,)
        Desired spectral bin centroids.
    FLUKA_tot : np.ndarray, shape (N_data, N_FLUKA)
        Normalised detector response matrix (each row divided by its peak).
    FLUKA_fact : np.ndarray, shape (N_data, 1)
        Peak normalization factors stripped during RM import.
    smooth_factor : float
        Weight of the smoothness penalty for this species.
    smooth_transition_start : float
        Logistic onset fraction (0–1). Used when adaptive smoothing is active.
    smooth_min_weight : float
        Floor weight at the high-energy tail of the logistic window.
    smooth_steepness : float
        Logistic steepness parameter k.
    """
    label: str
    E: np.ndarray
    E_guess: np.ndarray
    FLUKA_tot: np.ndarray
    FLUKA_fact: np.ndarray
    smooth_factor: float = 1.3e-5

    smooth_transition_start: float = 0.85
    smooth_min_weight: float = 0.01
    smooth_steepness: float = 30.0

    # Set by resolve(); do not initialise manually.
    nearest_indices: np.ndarray = field(init=False, default=None)
    N_guess: int = field(init=False, default=0)

    def resolve(self):
        """Map ``E_guess`` onto the RM energy axis and cache nearest indices."""
        self.nearest_indices = DataProcessor.find_nearest(self.E, self.E_guess)
        self.E_guess = self.E[self.nearest_indices]  # snap to actual RM energies
        self.N_guess = len(self.E_guess)


# ═══════════════════════════════════════════════════════════════════════════════
# Smoothness penalty functions
# ═══════════════════════════════════════════════════════════════════════════════

def _smoothness_simple(spectrum: np.ndarray, E_guess: np.ndarray) -> float:
    """
    Tikhonov-like penalty: sum of squared second finite differences.

    Penalises curvature in the reconstructed spectrum to suppress
    high-frequency oscillations from noise amplification.
    """
    if spectrum.size < 3:
        return 0.0
    d2 = spectrum[2:] - 2.0 * spectrum[1:-1] + spectrum[:-2]
    return np.sum(d2 ** 2)


def _smoothness_logistic(
    spectrum: np.ndarray,
    transition_start: float = 0.85,
    min_weight: float = 0.01,
    steepness: float = 30.0,
) -> float:
    """
    Logistic-weighted first-difference smoothness penalty.

    The weight decreases sigmoidally from ~1 at low-energy bins to
    ``min_weight`` at the high-energy tail, reducing over-regularisation
    near the spectral cutoff where RM sensitivity drops.

    w(x) = min_weight + (1 - min_weight) / (1 + exp(k*(x - x0)))
    x   = bin_index / (N_bins - 1)
    """
    norm_spec = spectrum / np.max(np.abs(spectrum))
    n = norm_spec.shape[0]
    x = np.arange(n) / (n - 1)
    w = min_weight + (1.0 - min_weight) / (1.0 + np.exp(steepness * (x - transition_start)))
    w_diff = 0.5 * (w[:-1] + w[1:])
    w_diff /= w_diff.max()
    diff = np.diff(norm_spec)
    return float(np.sum((diff * w_diff) ** 2))


# ═══════════════════════════════════════════════════════════════════════════════
# Loss helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def _huber_loss(residuals: np.ndarray, delta: float, weights: np.ndarray) -> np.ndarray:
    """
    Element-wise Huber loss.

    Quadratic for |r| < delta, linear beyond — provides robustness against
    outlier detector channels without fully discarding them.
    """
    abs_r = np.abs(residuals)
    q = np.minimum(abs_r, delta)
    lin = abs_r - q
    return weights * (0.5 * q ** 2 + delta * lin)


def _softplus(x: np.ndarray, k: float) -> np.ndarray:
    """Smooth approximation to ReLU: (1/k) * log(1 + exp(k*x))."""
    return (1.0 / k) * np.log1p(np.exp(k * x))


def _facts_penalty(facts: np.ndarray, k: float = 300.0) -> float:
    """
    Smooth quartic penalty keeping calibration factors close to 1.

    Applies a softplus-based barrier on both sides of unity so that
    the optimiser is gently discouraged from drifting far from the
    nominal detector calibration.
    """
    lower = _softplus(1.0 - facts, k) ** 4
    upper = _softplus(facts - 1.0, k) ** 4
    return float(np.sum(lower + upper))


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-species CMA-ES optimizer
# ═══════════════════════════════════════════════════════════════════════════════

class MultiSpeciesOptimizer:
    """
    CMA-ES optimizer supporting 1–3 particle species with optional
    calibration factor tuning and adaptive smoothing.

    The optimisation vector ``x`` is structured as::

        x = [ S_0 | S_1 | ... | S_{K-1} | facts (optional) ]

    where ``S_k`` has shape ``(N_guess_k,)`` and contains log10(dN/dE),
    and ``facts`` has shape ``(N_FLUKA,)`` initialised near 1.

    Parameters
    ----------
    species_list : List[SpeciesConfig]
        One to three species, each with its own RM.
    Exp_FLUKA : np.ndarray, shape (N_FLUKA,) or (1, N_FLUKA)
        Normalised experimental detector signal.
    tune_calibration : bool
        If True, per-detector calibration factors are appended to the
        optimisation vector and penalised via softplus around 1.
    smoothing : bool
        If True, a smoothness regularisation term is added to the loss.
    adaptive_smoothing : bool
        If True (and smoothing=True), uses logistic-weighted smoothing
        per SpeciesConfig parameters; otherwise uses simple finite-diff.
    lower_bound, upper_bound : float
        Bounds for spectral variables (log10 space).
    facts_bounds : tuple
        Bounds for calibration factor variables (offset from 1).
    huber_delta : float
        Huber loss transition parameter.
    facts_penalty_weight : float
        Weight on the calibration factor constraint penalty.
    cma_options : dict, optional
        Overrides for ``cma.CMAEvolutionStrategy`` options.
    """

    def __init__(
        self,
        species_list: List[SpeciesConfig],
        Exp_FLUKA: np.ndarray,
        tune_calibration: bool = False,
        smoothing: bool = True,
        adaptive_smoothing: bool = False,
        lower_bound: float = -20.0,
        upper_bound: float = 20.0,
        facts_bounds: tuple = (-0.2, 0.2),
        huber_delta: float = 1e-2,
        facts_penalty_weight: float = 1e-2,
        cma_options: Optional[dict] = None,
    ):
        if not 1 <= len(species_list) <= 3:
            raise ValueError("species_list must contain 1, 2, or 3 SpeciesConfig entries.")

        self.species_list = species_list
        for sp in self.species_list:
            sp.resolve()

        self.Exp_FLUKA = Exp_FLUKA.flatten()
        self.N_FLUKA = self.Exp_FLUKA.shape[0]

        self.tune_calibration = tune_calibration
        self.smoothing = smoothing
        self.adaptive_smoothing = adaptive_smoothing

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.facts_bounds = facts_bounds

        self.huber_delta = huber_delta
        self.facts_penalty_weight = facts_penalty_weight

        # Dimension accounting
        self.N_spectrum_vars = sum(sp.N_guess for sp in self.species_list)
        self.N_facts_vars = self.N_FLUKA if tune_calibration else 0
        self.N_total_vars = self.N_spectrum_vars + self.N_facts_vars

        self._cma_options = self._build_cma_options(cma_options)
        self._iter_counter = 0

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _build_cma_options(self, user_options: Optional[dict]) -> dict:
        """Build CMA-ES option dict with species-aware step sizes."""
        # Larger initial step for spectral variables, small for calibration factors
        stds = np.concatenate([
            np.ones(self.N_spectrum_vars) * 5.0,
            np.ones(self.N_facts_vars) * 0.05,
        ])
        defaults = {
            'bounds': [self.lower_bound, self.upper_bound],
            'maxiter': 100000,
            'seed': 123456,
            'verb_disp': 1000,
            'tolx': 1e-8,
            'CMA_stds': stds,
            'popsize': max(17, 4 + int(3 * np.log(self.N_total_vars))),
            'verbose': -8,
            'CMA_active': 1,
        }
        if user_options:
            defaults.update(user_options)
        return defaults

    def _unpack(self, x: np.ndarray):
        """
        Split the flat optimisation vector into per-species spectra and
        optional calibration factors.

        Returns
        -------
        spectra : list of np.ndarray  (log10 scale, one per species)
        facts   : np.ndarray of shape (N_FLUKA,), or None
        """
        spectra = []
        offset = 0
        for sp in self.species_list:
            spectra.append(x[offset: offset + sp.N_guess])
            offset += sp.N_guess
        facts = x[offset:] if self.tune_calibration else None
        return spectra, facts

    def _calc_FLUKA_species(
        self,
        spectrum_log: np.ndarray,
        sp: SpeciesConfig,
        facts: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Forward model for one species.

        Computes the expected detector signal as:
            FLUKA_sim_k = Σ_j [ 10^{S_j} · RM_j · norm_j · (facts) ]

        Parameters
        ----------
        spectrum_log : shape (N_guess_k,), log10(dN/dE)
        sp : SpeciesConfig
        facts : shape (N_FLUKA,) or None — per-detector calibration multipliers

        Returns
        -------
        np.ndarray, shape (N_FLUKA,)
        """
        sel_tot = sp.FLUKA_tot[sp.nearest_indices, :]    # (N_guess_k, N_FLUKA)
        sel_fact = sp.FLUKA_fact[sp.nearest_indices, :]   # (N_guess_k, 1)

        if facts is not None:
            sel_tot = sel_tot * facts[np.newaxis, :]      # broadcast over energy bins

        return np.sum(
            (10.0 ** spectrum_log).reshape(-1, 1) * sel_tot * sel_fact,
            axis=0,
        )

    # ── Objective function variants ──────────────────────────────────────────
    # One per active-flag combination.  The dispatcher _build_objective_fn()
    # selects the appropriate variant once before entering the CMA loop.

    def _objective_base(self, spectra: list, facts: Optional[np.ndarray]) -> float:
        """Fidelity term only (Huber loss on relative residuals)."""
        FLUKA_sim = sum(
            self._calc_FLUKA_species(s, sp)
            for s, sp in zip(spectra, self.species_list)
        )
        residuals = (self.Exp_FLUKA - FLUKA_sim) ** 2 / (self.Exp_FLUKA + 1e-30) ** 2
        return float(np.sum(
            _huber_loss(residuals, self.huber_delta, np.ones_like(residuals))
        ))

    def _objective_smooth(self, spectra: list, facts: Optional[np.ndarray]) -> float:
        """Fidelity + smoothness penalty."""
        fidelity = self._objective_base(spectra, facts)
        smooth_total = 0.0
        for s, sp in zip(spectra, self.species_list):
            if self.adaptive_smoothing:
                st = _smoothness_logistic(
                    s,
                    transition_start=sp.smooth_transition_start,
                    min_weight=sp.smooth_min_weight,
                    steepness=sp.smooth_steepness,
                )
            else:
                st = _smoothness_simple(s, sp.E_guess)
            smooth_total += sp.smooth_factor * st
        return fidelity + smooth_total

    def _objective_calib(self, spectra: list, facts: np.ndarray) -> float:
        """Fidelity + calibration factor penalty (no smoothing)."""
        FLUKA_sim = sum(
            self._calc_FLUKA_species(s, sp, facts)
            for s, sp in zip(spectra, self.species_list)
        )
        residuals = (self.Exp_FLUKA - FLUKA_sim) ** 2 / (self.Exp_FLUKA + 1e-30) ** 2
        fidelity = float(np.sum(
            _huber_loss(residuals, self.huber_delta, np.ones_like(residuals))
        ))
        return fidelity + self.facts_penalty_weight * _facts_penalty(facts)

    def _objective_smooth_calib(self, spectra: list, facts: np.ndarray) -> float:
        """Fidelity + smoothness + calibration factor penalty."""
        FLUKA_sim = sum(
            self._calc_FLUKA_species(s, sp, facts)
            for s, sp in zip(spectra, self.species_list)
        )
        residuals = (self.Exp_FLUKA - FLUKA_sim) ** 2 / (self.Exp_FLUKA + 1e-30) ** 2
        fidelity = float(np.sum(
            _huber_loss(residuals, self.huber_delta, np.ones_like(residuals))
        ))
        smooth_total = 0.0
        for s, sp in zip(spectra, self.species_list):
            if self.adaptive_smoothing:
                st = _smoothness_logistic(
                    s,
                    transition_start=sp.smooth_transition_start,
                    min_weight=sp.smooth_min_weight,
                    steepness=sp.smooth_steepness,
                )
            else:
                st = _smoothness_simple(s, sp.E_guess)
            smooth_total += sp.smooth_factor * st
        return fidelity + smooth_total + self.facts_penalty_weight * _facts_penalty(facts)

    # ── Dispatcher ───────────────────────────────────────────────────────────

    def _build_objective_fn(self):
        """Select the correct objective variant based on active flags (once)."""
        if not self.tune_calibration and not self.smoothing:
            return self._objective_base
        elif not self.tune_calibration and self.smoothing:
            return self._objective_smooth
        elif self.tune_calibration and not self.smoothing:
            return self._objective_calib
        else:
            return self._objective_smooth_calib

    def mmin(self, x: np.ndarray) -> float:
        """CMA-ES callable: unpack x, evaluate pre-selected objective."""
        spectra, facts = self._unpack(x)
        return self._objective_fn(spectra, facts)

    # ── Public API ───────────────────────────────────────────────────────────

    def calc_FLUKA(self, result: 'OptimizationResult') -> np.ndarray:
        """
        Reconstruct the total simulated detector signal from an
        :class:`OptimizationResult`.

        Returns
        -------
        np.ndarray, shape (N_FLUKA,)
        """
        total = np.zeros(self.N_FLUKA)
        for sim_linear, sp in zip(result.spectra_linear, self.species_list):
            total += self._calc_FLUKA_species(
                np.log10(np.clip(sim_linear, 1e-30, None)),
                sp,
                result.facts,
            )
        return total

    def run_CMA(self) -> 'OptimizationResult':
        """
        Run CMA-ES optimisation and return structured results.

        The objective function is dispatched once before entering the
        evolution loop, eliminating conditional overhead per evaluation.
        """
        self._objective_fn = self._build_objective_fn()

        # Initial mean: uniform random in [lower, upper] for spectra,
        # near 1.0 for calibration factors.
        x0_spectra = np.random.uniform(
            self.lower_bound, self.upper_bound, self.N_spectrum_vars
        )
        if self.tune_calibration:
            x0_facts = np.random.uniform(0.98, 1.02, self.N_facts_vars)
            x0 = np.concatenate([x0_spectra, x0_facts])
        else:
            x0 = x0_spectra

        sigma0 = 3.0
        es = cma.CMAEvolutionStrategy(x0, sigma0, self._cma_options)

        while not es.stop():
            solutions = es.ask()
            es.tell(solutions, [self.mmin(sol) for sol in solutions])
            es.logger.add()
            es.disp()

        x_best = es.result.xbest
        spectra_log, facts = self._unpack(x_best)
        spectra_linear = [10.0 ** s for s in spectra_log]

        return OptimizationResult(
            spectra_linear=spectra_linear,
            spectra_log=spectra_log,
            facts=facts,
            species_labels=[sp.label for sp in self.species_list],
            E_guess_list=[sp.E_guess for sp in self.species_list],
            es_result=es.result,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Optimisation result container
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OptimizationResult:
    """
    Structured output of :meth:`MultiSpeciesOptimizer.run_CMA`.

    Attributes
    ----------
    spectra_linear : list of np.ndarray
        Unfolded spectra in linear scale (one per species).
    spectra_log : list of np.ndarray
        Same spectra in log10 space (raw CMA output).
    facts : np.ndarray or None
        Optimised per-detector calibration factors.
    species_labels : list of str
        Labels from ``SpeciesConfig.label``.
    E_guess_list : list of np.ndarray
        Energy grids (snapped to RM axis) for each species.
    es_result : cma result object
        Raw CMA result for diagnostics (convergence, etc.).
    """
    spectra_linear: list
    spectra_log: list
    facts: Optional[np.ndarray]
    species_labels: list
    E_guess_list: list
    es_result: object

    def get(self, label: str) -> tuple:
        """
        Retrieve ``(E_guess, spectrum_linear)`` for a species by label.

        Raises
        ------
        KeyError
            If ``label`` is not found among species.
        """
        for i, lbl in enumerate(self.species_labels):
            if lbl == label:
                return self.E_guess_list[i], self.spectra_linear[i]
        raise KeyError(f"Species '{label}' not found. Available: {self.species_labels}")


# ═══════════════════════════════════════════════════════════════════════════════
# Data processing utilities
# ═══════════════════════════════════════════════════════════════════════════════

class DataProcessor:
    """Handles RM import and experimental image reading."""

    @staticmethod
    def initialize_array(N_data, N_sim):
        """Allocate zeroed arrays for spectrum, FLUKA response, and normalisation."""
        Spectrum_tot = np.zeros((N_data, N_sim))
        FLUKA_tot = np.zeros((N_data, Config.N_FLUKA))
        FLUKA_fact = np.zeros((N_data, 1))
        return Spectrum_tot, FLUKA_tot, FLUKA_fact

    @staticmethod
    def find_nearest(array, values):
        """
        For each value in ``values``, find the index of the nearest element
        in ``array``, ensuring no index is reused (injective mapping).

        Raises
        ------
        ValueError
            If any value falls outside the range of ``array``.
        """
        used_indices = set()
        indices = []

        for val in values:
            index = np.argmin(np.abs(val - array))
            while index in used_indices:
                index += 1
                if index >= len(array):
                    index = 0
                    print('Wrong end of E')
                    return None
            if val > np.amax(array) or val < np.amin(array):
                print('E bin not in RM')
                raise ValueError
            used_indices.add(index)
            indices.append(index)

        return np.array(indices)

    @staticmethod
    def read_image(norm_flag=True, plot_flag=False):
        """
        Read the scintillator image and extract integrated signal per ROI.

        Parameters
        ----------
        norm_flag : bool
            If True, normalise each ROI sum by its pixel count.
        plot_flag : bool
            If True, display each ROI sub-image.

        Returns
        -------
        np.ndarray, shape (N_FLUKA,)
        """
        im = np.array(Image.open(Config.image_path))

        exp_spectrum_image = []
        for y_min, y_max, x_min, x_max in zip(
            Config.ROI[:, 0], Config.ROI[:, 1],
            Config.ROI[:, 2], Config.ROI[:, 3],
        ):
            ROI_image = im[x_min:x_max, y_min:y_max]
            value_scint = np.sum(ROI_image)
            if plot_flag:
                plt.imshow(ROI_image)
                plt.show()
            if norm_flag:
                value_scint = value_scint / ROI_image.size
            exp_spectrum_image.append(value_scint)

        return np.array(exp_spectrum_image)

    @staticmethod
    def sort_numerically(filename):
        """Extract leading number from filename for sorting."""
        num = re.findall(r"\d+\.\d+|\d+", filename)
        return float(num[0]) if num else 0

    @classmethod
    def import_RM(cls):
        """
        Import the response matrix from ``Config.folder_path``.

        Reads triplets of files per mono-energetic simulation:
          - ``<energy>_Energy.txt``   — energy axis
          - ``<energy>_Spectrum.txt`` — (unused, reserved)
          - ``<energy>_FLUKA.txt``    — detector response vector

        The FLUKA response is multiplied by ``Config.factor`` and then
        peak-normalised; the peak is stored in ``FLUKA_fact``.

        Returns
        -------
        Spectrum_tot : np.ndarray, shape (N_data, N_sim)
        FLUKA_tot    : np.ndarray, shape (N_data, N_FLUKA)
        FLUKA_fact   : np.ndarray, shape (N_data, 1)
        E            : np.ndarray, shape (N_data,), sorted energies in MeV
        """
        Spectrum_tot, FLUKA_tot, FLUKA_fact = cls.initialize_array(Config.N_data, Config.N_sim)
        name_list = np.zeros(Config.N_data, dtype=float)

        all_files = glob.glob(os.path.join(Config.folder_path, '*'))
        spectrum_files = sorted(
            [f for f in all_files if os.path.basename(f).endswith('_Spectrum.txt')],
            key=cls.sort_numerically,
        )

        for idx, spec_path in enumerate(spectrum_files):
            base = os.path.basename(spec_path)
            m = re.match(r'(.+?)_Spectrum', base)
            if not m:
                continue
            name = m.group(1)

            energy_path = os.path.join(os.path.dirname(spec_path), f"{name}_Energy.txt")
            fluka_path = os.path.join(os.path.dirname(spec_path), f"{name}_FLUKA.txt")

            try:
                pd.read_csv(energy_path, header=None)  # validate existence
            except FileNotFoundError:
                continue

            FLUKA = pd.read_csv(fluka_path, header=None).iloc[:, 0].to_numpy()
            FLUKA *= Config.factor
            peak = FLUKA.max()
            if peak == 0:
                continue
            FLUKA /= peak

            FLUKA_tot[idx, :] = FLUKA
            FLUKA_fact[idx] = peak
            name_list[idx] = float(name)

        # Sort all arrays by ascending energy
        order = np.argsort(name_list)
        E = name_list[order]
        Spectrum_tot = Spectrum_tot[order, :]
        FLUKA_tot = FLUKA_tot[order, :]
        FLUKA_fact = FLUKA_fact[order]

        return Spectrum_tot, FLUKA_tot, FLUKA_fact, E


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting utilities
# ═══════════════════════════════════════════════════════════════════════════════

class Plotter:
    """Static methods for standard diagnostic plots."""

    @staticmethod
    def plot_results(Exp_FLUKA, FLUKA_sim):
        """Compare experimental and simulated detector signals (normalised)."""
        plt.figure(figsize=(10, 6))
        plt.plot(Exp_FLUKA, label='Exp')
        plt.plot(FLUKA_sim / np.amax(FLUKA_sim), label='Sim')
        plt.xlabel('Bin')
        plt.ylabel('Normalized Intensity')
        plt.title('Experimental vs Simulated FLUKA')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_spectrum(E_guess, sim):
        """Plot unfolded energy spectrum on log–log axes."""
        plt.figure(figsize=(10, 6))
        plt.plot(E_guess, sim)
        plt.xlabel('Energy (MeV)')
        plt.ylabel(r'dN/dE$_{\log_{10}}$')
        plt.title('Unfolded spectrum')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True)
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Error analysis
# ═══════════════════════════════════════════════════════════════════════════════

class ErrorAnalysis:
    """
    Builds and interpolates a 2-D error matrix over (normalised signal, energy).

    The error matrix encodes the expected relative uncertainty of the unfolding
    procedure as a function of signal level and energy, pre-computed from
    multiple Monte Carlo trials at different noise levels.

    Parameters
    ----------
    E_error : np.ndarray
        Energy grid for the error matrix columns.
    error_files : list of str
        Paths to four files containing error levels at different signal
        thresholds (>0.1, >0.01, >0.001, ≤0.001 of peak).
    ddv : np.ndarray
        Normalised signal intensity axis (rows of the error matrix).
    window_size : int
        Moving-average window for smoothing the error matrix.
    """

    def __init__(self, E_error, error_files, ddv, window_size=200):
        self.E_error = E_error
        self.ddv = ddv
        self.window_size = window_size

        # Load the four error-level files
        self.error1 = pd.read_csv(error_files[0], header=None).values[:, 0]
        self.error2 = pd.read_csv(error_files[1], header=None).values[:, 0]
        self.error3 = pd.read_csv(error_files[2], header=None).values[:, 0]
        self.error4 = pd.read_csv(error_files[3], header=None).values[:, 0]

        # Build and smooth the error matrix
        self.error_matrix = self._create_error_matrix()
        self.error_matrix = self.smooth_2d_array(self.error_matrix, self.window_size)

    def _create_error_matrix(self):
        """
        Fill the error matrix by selecting among four error levels
        depending on the normalised signal value (ddv threshold).

        NaN entries in the source files are replaced with 1 (100% error).
        """
        error_matrix = np.zeros((len(self.ddv), len(self.E_error)))
        for i, xi in enumerate(self.E_error):
            for j, dv in enumerate(self.ddv):
                if dv > 0.1:
                    error_matrix[j, i] = self.error1[i] if not np.isnan(self.error1[i]) else 1
                elif dv > 0.01:
                    error_matrix[j, i] = self.error2[i] if not np.isnan(self.error2[i]) else 1
                elif dv > 0.001:
                    error_matrix[j, i] = self.error3[i] if not np.isnan(self.error3[i]) else 1
                else:
                    error_matrix[j, i] = self.error4[i] if not np.isnan(self.error4[i]) else 1
        error_matrix = self.smooth_2d_array(error_matrix, 200)
        return error_matrix

    @staticmethod
    def smooth_2d_array(data, window_size, axis=0):
        """
        Smooth a 2-D array along ``axis`` using a uniform moving average.
        """
        kernel = np.ones(window_size) / window_size
        return np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode='same'), axis=axis, arr=data,
        )

    def get_errors(self, x_vals, data_vals):
        """
        Interpolate errors from the pre-computed matrix.

        Parameters
        ----------
        x_vals : np.ndarray
            Energy values at which to evaluate the error.
        data_vals : np.ndarray
            Corresponding spectrum values (will be peak-normalised internally).

        Returns
        -------
        np.ndarray
            Interpolated relative errors, same shape as ``x_vals``.
        """
        x_vals = np.array(x_vals).flatten()
        data_vals = np.array(data_vals).flatten()
        data_vals = data_vals / np.amax(data_vals)

        interp_func = RegularGridInterpolator(
            (self.ddv, self.E_error), self.error_matrix,
            bounds_error=True, fill_value=True,
        )

        errors = np.zeros_like(x_vals)
        for i in range(len(x_vals)):
            try:
                errors[i] = interp_func(np.array([[data_vals[i], x_vals[i]]]))
            except Exception:
                errors[i] = np.nan
        return errors

    def plot_error_results(self, E_guess, signal, y_errors, std):
        """
        Plot the unfolded spectrum with combined error bars.

        Error budget includes:
          - ``y_errors``: numerical unfolding uncertainty (from RM)
          - ``std``: stack measurement repeatability
          - Poisson counting noise: sqrt(N)/N per bin

        Parameters
        ----------
        E_guess : np.ndarray, shape (N,)
        signal : np.ndarray, shape (N,), linear dN/dE (unnormalised)
        y_errors : np.ndarray, shape (N,), relative errors from interpolation
        std : float, additional systematic relative error
        """
        plt.rcParams.update({'font.size': 24})
        plt.figure(figsize=(10, 8))

        deltaE = np.diff(E_guess)

        # Convert bin values to true dN/dE (particles/MeV)
        N_tot_exp = np.sum(signal)
        scale_exp = N_tot_exp / np.sum(signal[:-1] * deltaE)
        signal = signal[:-1] * scale_exp

        # Poisson contribution
        dN = signal * deltaE
        N_error = np.sqrt(dN) / dN

        # Total relative error
        relative_error = y_errors[:-1] + std + N_error

        y_err_lower = signal - signal / (1 + relative_error)
        y_err_upper = signal * relative_error
        y_err_lower3 = signal / (1 + 3 * relative_error)

        plt.errorbar(
            E_guess[:-1], signal,
            yerr=[y_err_lower, y_err_upper],
            color='k', ecolor='b', capsize=5, label='Unfolded',
        )
        plt.fill_between(
            E_guess[:-1],
            3 * y_err_upper + signal, y_err_lower3,
            color='gray', alpha=0.2, label=r'3 $\sigma$ error',
        )

        plt.ylabel(r'$dN/dE\;(\mathrm{MeV}^{-1})$')
        plt.xlabel('E (MeV)')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend(loc='upper left', fontsize=19)
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic data generation
# ═══════════════════════════════════════════════════════════════════════════════

def make_synthetic_exp(FLUKA_tot, FLUKA_fact, E, E_guess, known_spectrum):
    """
    Build a synthetic ``Exp_FLUKA`` by running the forward model on a
    known spectrum, exactly as ``_calc_FLUKA_species`` does internally.

    Used for validation / closure tests.

    Parameters
    ----------
    known_spectrum : shape (N_guess,), linear scale (not log10)

    Returns
    -------
    Exp_FLUKA : shape (1, N_FLUKA), unnormalised synthetic detector signal
    """
    nearest_indices = DataProcessor.find_nearest(E, E_guess)
    sel_tot = FLUKA_tot[nearest_indices, :]
    sel_fact = FLUKA_fact[nearest_indices, :]
    FLUKA_sim = np.sum(
        known_spectrum.reshape(-1, 1) * sel_tot * sel_fact, axis=0,
    )
    return FLUKA_sim.reshape(1, -1)


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Initialise error analysis and import primary RM ───────────────────
    error_analysis = ErrorAnalysis(Config.E_error, Config.error_files, Config.ddv)
    data_processor = DataProcessor()
    Spectrum_tot, FLUKA_tot, FLUKA_fact, E = data_processor.import_RM()

    # ── 2. Optimiser feature flags ───────────────────────────────────────────
    TUNE_CALIBRATION = False    # optimise per-detector calibration factors
    SMOOTHING = True            # add smoothness regularisation
    ADAPTIVE_SMOOTHING = False  # True → logistic-weighted; False → simple Tikhonov

    # ── 3. Build synthetic experimental signal (closure test) ────────────────
    E_end = 10  # Temperature parameter for exponential photon spectrum (MeV)
    N_particules_p = 1
    known_spectrum = (1 / Config.E_guess) * np.exp(-Config.E_guess / E_end) * N_particules_p

    Exp_FLUKA = make_synthetic_exp(
        FLUKA_tot, FLUKA_fact, E, Config.E_guess, known_spectrum,
    )

    # ── 4. Build species configs ─────────────────────────────────────────────
    species_p = SpeciesConfig(
        label='photon',
        E=E,
        E_guess=Config.E_guess.copy(),
        FLUKA_tot=FLUKA_tot,
        FLUKA_fact=FLUKA_fact,
        smooth_factor=5e-6,
        smooth_transition_start=0.9,
        smooth_min_weight=0.01,
        smooth_steepness=20.0,
    )
    species_list = [species_p]

    # ── 5. Optionally add electron species ───────────────────────────────────
    if Config.n_species == 2:
        
        _folder_backup = Config.folder_path
        Config.folder_path = Config.folder_path_e
        _, FLUKA_tot_e, FLUKA_fact_e, E_e = data_processor.import_RM()
        
        Config.folder_path = _folder_backup

        species_e = SpeciesConfig(
            label='electron',
            E=E_e,
            E_guess=Config.E_guess_e,
            FLUKA_tot=FLUKA_tot_e,
            FLUKA_fact=FLUKA_fact_e,
            smooth_factor=1.3e-9,
            smooth_transition_start=0.30,
            smooth_min_weight=0.10,
            smooth_steepness=10,
        )
        species_list.append(species_e)

        # Electron contribution to synthetic signal
        E_temp_e = 10       # Electron temperature (MeV)
        N_particules_e = 3e5
        known_spectrum_e = np.exp(-Config.E_guess_e / E_temp_e) * N_particules_e
        FLUKA_e = make_synthetic_exp(
            FLUKA_tot_e, FLUKA_fact_e, E_e, Config.E_guess_e, known_spectrum_e,
        )
        FLUKA_p = np.copy(Exp_FLUKA)
        Exp_FLUKA = Exp_FLUKA + FLUKA_e

        plt.plot(FLUKA_e.flatten(), label='Electrons')
        plt.plot(FLUKA_p.flatten(), label='Photons')

    # Normalise combined signal
    Exp_max = np.amax(Exp_FLUKA)
    Exp_FLUKA = Exp_FLUKA / Exp_max

    plt.plot(Exp_FLUKA.flatten() * Exp_max, label='Total')
    plt.legend()
    plt.show()

    # ── 6. Build and run optimiser ───────────────────────────────────────────
    optimizer = MultiSpeciesOptimizer(
        species_list=species_list,
        Exp_FLUKA=Exp_FLUKA,
        tune_calibration=TUNE_CALIBRATION,
        smoothing=SMOOTHING,
        adaptive_smoothing=ADAPTIVE_SMOOTHING,
        lower_bound=Config.lower_bound,
        upper_bound=Config.upper_bound,
    )

    result = optimizer.run_CMA()

    # ── 7. Post-processing: error analysis and comparison plots ──────────────
    for label, E_out, sim_lin in zip(
        result.species_labels, result.E_guess_list, result.spectra_linear,
    ):
        y_errors = error_analysis.get_errors(E_out.reshape(-1, 1), sim_lin * Exp_max)
        error_analysis.plot_error_results(E_out, sim_lin * Exp_max, y_errors, 1)
        Plotter.plot_spectrum(E_out, sim_lin * Exp_max)

        plt.plot(E_out, sim_lin * Exp_max, label='Unfolded')
        if label == 'photon':
            plt.plot(Config.E_guess, known_spectrum)
        elif label == 'electron':
            plt.plot(Config.E_guess_e, known_spectrum_e)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

    # ── 8. Reconstruct total detector signal from unfolded spectra ───────────
    FLUKA_total = np.zeros(Config.N_FLUKA)
    for sp, spec in zip(species_list, result.spectra_linear):
        FLUKA_sim = optimizer._calc_FLUKA_species(np.log10(spec * Exp_max), sp)
        FLUKA_total += FLUKA_sim
        if len(sp.label) > 1:
            plt.plot(FLUKA_sim.flatten(), label=str(sp.label))

    plt.plot(Exp_FLUKA.flatten() * Exp_max, label='Original')
    plt.plot(FLUKA_total, label='Unfolded')
    plt.legend()
    plt.show()

    if result.facts is not None:
        print(f'Calibration factors: {result.facts}')


if __name__ == "__main__":
    main()
