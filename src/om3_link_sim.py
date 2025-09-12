#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OM3 Link Approximation Demo (No pyMMF required)

This script approximates an OM3 multimode fiber link as a *Gaussian low-pass channel*
whose -3 dB bandwidth is derived from the Effective Modal Bandwidth (EMB). It then
simulates NRZ signaling through the channel and plots eye diagrams for two presets:
- 10 Gb/s over 300 m
- 25 Gb/s over 100 m

Notes
-----
- This is a *system-level* toy model intended for intuition-building. It is NOT an IEEE
  compliance test. It ignores connector reflections, spectral chirp, equalization/FEC, etc.
- If you later have a channel impulse response (e.g., from pyMMF or measurements),
  you can drop it into `custom_impulse_response` to replace the Gaussian model.

Usage
-----
python om3_link_sim.py

Dependencies
------------
numpy, scipy, matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

# ----------------------------
# Utilities
# ----------------------------
def gaussian_impulse_from_3db_bandwidth(B_3dB_Hz, t_span_sigma=8.0, fs=1.0):
    """
    Create a *unit-energy* Gaussian impulse response h(t) whose AMPLITUDE response |H(f)|
    has -3 dB at B_3dB_Hz. For a Gaussian h(t) with std sigma_t,
        |H(f)| = exp(- 2 * pi^2 * sigma_t^2 * f^2)
    Solve |H(B)| = 1/sqrt(2) -> sigma_t = sqrt(ln(2)) / (2*pi*B)

    Parameters
    ----------
    B_3dB_Hz : float
        Amplitude 3 dB bandwidth of the channel in Hz
    t_span_sigma : float
        Truncation span in multiples of sigma_t on each side (total ~ 2*t_span_sigma*sigma_t)
    fs : float
        Sampling frequency (Hz) of the discrete-time implementation

    Returns
    -------
    h : ndarray
        Discrete-time impulse response (unit-energy)
    t : ndarray
        Time axis corresponding to h (seconds)
    sigma_t : float
        Time-domain standard deviation of the Gaussian (seconds)
    """
    sigma_t = np.sqrt(np.log(2.0)) / (2.0 * np.pi * B_3dB_Hz)
    # time vector spanning +/- t_span_sigma * sigma_t
    t_max = t_span_sigma * sigma_t
    dt = 1.0 / fs
    # ensure odd length so that impulse is centered at index mid
    n_half = int(np.ceil(t_max / dt))
    t = np.arange(-n_half, n_half + 1) * dt
    # unit-energy Gaussian
    h = (1.0 / (np.sqrt(2.0 * np.pi) * sigma_t)) * np.exp(-0.5 * (t / sigma_t) ** 2)
    # normalize to unit energy (discrete-time)
    h = h / np.sqrt(np.sum(h ** 2))
    return h, t, sigma_t

def nrz_baseband(bits, sps, pulse_shape="rect"):
    """
    Simple NRZ pulse shaping (rectangular). Output is 0/1-level waveform.
    """
    if pulse_shape != "rect":
        raise NotImplementedError("Only rectangular NRZ is implemented in this demo.")
    up = np.repeat(bits.astype(float), sps)
    return up

def awgn(signal, snr_db):
    """
    Add white Gaussian noise at target SNR (per-sample, based on signal power).
    """
    if snr_db is None:
        return signal
    power = np.mean(signal ** 2)
    snr_lin = 10 ** (snr_db / 10.0)
    noise_power = power / snr_lin
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    return signal + noise

def eye_diagram(y, sps, n_traces=200, ui=2):
    """
    Collect segments of length `ui` (in unit intervals) and overlay.
    Returns a 2D array of shape (n_traces, ui*sps) for plotting.
    """
    seg_len = ui * sps
    starts = np.arange(0, len(y) - seg_len, sps)
    if len(starts) > n_traces:
        starts = np.random.choice(starts, size=n_traces, replace=False)
    eyes = np.stack([y[s:s + seg_len] for s in starts], axis=0)
    return eyes

def hard_decision_ber(waveform, sps, bits_tx, sample_offset=0.5, thresh=None):
    """
    Sample at the eye center and do hard decision. Returns BER and decisions.
    sample_offset : position in UI to sample (0..1), default 0.5 (center)
    """
    n_bits = len(bits_tx)
    center_index = int(np.round(sample_offset * sps))
    sample_points = np.arange(center_index, center_index + n_bits * sps, sps)
    sample_points = sample_points[sample_points < len(waveform)]
    y_s = waveform[sample_points]

    # default threshold: midway between empirical 0/1 clusters
    if thresh is None:
        # crude clustering by using lower/upper quartiles
        low = np.median(y_s[y_s < np.median(y_s)])
        high = np.median(y_s[y_s >= np.median(y_s)])
        thresh = 0.5 * (low + high)
    bits_rx = (y_s > thresh).astype(int)
    n = min(len(bits_rx), len(bits_tx))
    ber = np.mean(bits_rx[:n] != bits_tx[:n])
    return ber, bits_rx[:n], y_s[:n]

# ----------------------------
# Channel model from OM3 EMB
# ----------------------------
def emb_to_bandwidth_hz(emb_mhz_km, length_m):
    """
    Convert Effective Modal Bandwidth (EMB, in MHz·km) and length (m) to approximate
    -3 dB amplitude bandwidth in Hz: B ≈ EMB / L_km (then MHz->Hz)
    """
    L_km = length_m / 1000.0
    B_MHz = emb_mhz_km / max(L_km, 1e-12)
    return B_MHz * 1e6

# ----------------------------
# End-to-end simulation
# ----------------------------
def run_case(bit_rate, length_m, emb_mhz_km=2000.0, snr_db=None, n_bits=2000, sps=64, show_plots=True):
    """
    bit_rate : bit rate in bps (e.g., 10e9 or 25e9)
    length_m : fiber length in meters (e.g., 300 or 100)
    emb_mhz_km : OM3 effective modal bandwidth (typ. 2000 MHz·km at 850 nm)
    snr_db : add AWGN at this SNR (per-sample). Use None to disable noise.
    n_bits : number of bits to simulate
    sps : samples per symbol
    show_plots : if True, display eye diagrams
    """
    print(f"=== Case: {bit_rate/1e9:.1f} Gb/s over {length_m} m (EMB={emb_mhz_km} MHz·km) ===")
    fs = bit_rate * sps
    # Channel bandwidth from EMB
    B = emb_to_bandwidth_hz(emb_mhz_km, length_m)
    h, t, sigma_t = gaussian_impulse_from_3db_bandwidth(B, t_span_sigma=8.0, fs=fs)
    print(f"Approx. channel 3 dB bandwidth: {B/1e9:.3f} GHz; Gaussian sigma_t: {sigma_t*1e12:.2f} ps")

    # Transmit random NRZ bits (0/1 levels). For symmetry, map to {0,1}; DC will be filtered a bit.
    bits_tx = np.random.randint(0, 2, size=n_bits)
    x = nrz_baseband(bits_tx, sps=sps, pulse_shape='rect')

    # Convolution through channel
    y = fftconvolve(x, h, mode='full')

    # Add AWGN if desired
    y = awgn(y, snr_db=snr_db)

    # Eye diagram data
    eyes = eye_diagram(y, sps=sps, n_traces=200, ui=2)

    # Hard decision BER at UI center
    ber, bits_rx, y_samples = hard_decision_ber(y, sps=sps, bits_tx=bits_tx, sample_offset=0.5, thresh=None)
    print(f"Estimated BER (no EQ, hard decision): {ber:.3e}")

    if show_plots:
        # Eye plot
        plt.figure(figsize=(7, 4))
        for row in eyes:
            plt.plot(np.arange(len(row)) / sps, row, linewidth=0.7, alpha=0.3)
        plt.title(f"Eye Diagram: {bit_rate/1e9:.1f} Gb/s, L={length_m} m")
        plt.xlabel("UI")
        plt.ylabel("Amplitude")
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()

        # Sample histogram around eye center
        plt.figure(figsize=(6, 3.5))
        plt.hist(y_samples, bins=80, density=True)
        plt.title("Sample Distribution at Eye Center")
        plt.xlabel("Amplitude")
        plt.ylabel("PDF")
        plt.tight_layout()
        plt.show()

    return ber

# ----------------------------
# Optional: replace Gaussian with custom impulse response
# ----------------------------
def custom_impulse_response(fs):
    """
    Stub for inserting a measured or pyMMF-derived impulse response.
    Return h (1D ndarray) sampled at rate fs [Hz]. Normalize to unit energy.
    """
    return None  # Replace with your own h if available.

# ----------------------------
# Run two presets and print BER
# ----------------------------
if __name__ == "__main__":
    # Typical OM3 EMB ~ 2000 MHz·km at 850 nm
    EMB = 2000.0

    # You can tweak SNR to see noise impact; None -> no noise
    SNR_DB = None  # e.g., 30 for noisy case

    ber_10g_300m = run_case(bit_rate=10e9, length_m=300, emb_mhz_km=EMB, snr_db=SNR_DB, n_bits=4000, sps=64)
    ber_25g_100m = run_case(bit_rate=25e9, length_m=100, emb_mhz_km=EMB, snr_db=SNR_DB, n_bits=4000, sps=64)

    print("\nSummary:")
    print(f"  10G@300m  -> BER ≈ {ber_10g_300m:.3e}")
    print(f"  25G@100m  -> BER ≈ {ber_25g_100m:.3e}")
