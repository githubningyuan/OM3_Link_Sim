# OM3 Link Simulator

A lightweight Python simulator for **multimode fiber (MMF) links** such as OM3/OM4 at 850 nm.  
The tool generates eye diagrams and rough BER estimates using Gaussian channel models or custom pyMMF-derived impulse responses.

## Installation

```bash
# (optional) create and activate a virtual environment
# python -m venv .venv && source .venv/bin/activate   # macOS/Linux
# .\.venv\Scripts\activate                            # Windows PowerShell

pip install -r requirements.txt
```

## Quick Start

Run the built-in demo simulation:

```bash
python om3_link_sim.py
```

Two presets will run:

- **10G @ 300 m** (EMB = 2000 MHz·km)  
- **25G @ 100 m** (EMB = 2000 MHz·km)  

You’ll see **eye diagrams** and a rough **BER** from a hard-decision sampler at 0.5 UI.

## Customization

### Using a pyMMF-Derived Impulse
1. Generate or compute the channel impulse response `h(t)` at sampling frequency `fs = bit_rate * sps`.  
2. Normalize it to unit energy:  
   ```python
   h = h / np.sqrt(np.sum(h**2))
   ```  
3. Return it from `custom_impulse_response(fs)` and replace the default Gaussian in `run_case()`.

### Parameters You Can Tweak
- `EMB` in `om3_link_sim.py` (e.g., 2000 for OM3; try 4700 for OM4 to see clearer eyes)  
- `SNR_DB` (add noise)  
- `n_bits` and `sps` (longer sequences, finer resolution)  
- `bit_rate` and `length_m`

## Background

### Why Gaussian?
For strongly coupled multimode fiber links, the **aggregate effect of modal dispersion** is often modeled as a Gaussian low-pass filter.  
The **EMB** (Effective Modal Bandwidth) published by fiber vendors is commonly interpreted as a *–3 dB bandwidth* for a given length at 850 nm, which serves as a practical proxy for first-order simulations.

## Project Structure

```
.
├── src/             # library code
├── examples/        # runnable scripts
├── notebooks/       # Jupyter notebooks (experiments, tutorials)
├── tests/           # unit tests
├── docs/            # documentation, images
├── om3_link_sim.py  # main simulation script
├── requirements.txt
├── README.md
└── LICENSE
```

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
