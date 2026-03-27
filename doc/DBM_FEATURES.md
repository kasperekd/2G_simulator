# dBm Power Calculation Feature

## Overview

This document describes the new dBm-based power calculation feature added to the 2G simulator. The simulator now supports both traditional relative dB-based calculations and absolute dBm-based physical power calculations.

## Changes Summary

### New Files

1. **`receiver/power_calc.py`** - Utility module for power calculations
   - `dbm_to_watts()`, `watts_to_dbm()` - Convert between dBm and watts
   - `db_to_linear()`, `linear_to_db()` - Convert between dB and linear scale
   - `calculate_received_power_dbm()` - Calculate received power using link budget
   - `calculate_thermal_noise_power_dbm()` - Calculate thermal noise power in dBm
   - `calculate_snr_from_powers_*()` - Calculate SNR from power values
   - `calculate_ci_from_interference_dbm()` - Calculate CI ratio
   - `calculate_sinr_from_powers_dbm()` - Calculate SINR ratio

### Modified Files

| File | Changes |
|------|---------|
| `config/validator.py` | Added `PowerParametersConfig` class with dBm parameters |
| `config/extract_parameters.py` | Extract dBm parameters from config |
| `receiver/scaling_and_combing.py` | Added `scaling_combining_and_noise_dbm()` function |
| `core_simulation.py` | Added dBm parameter extraction and output display |
| `visualisation/result.py` | Added dBm metadata to CSV output |
| `core/iterations.py` | Updated `single_burst_iteration()` to accept dBm parameters |

## Configuration

### Power Parameters in settings.json

```json
{
  "power_parameters": {
    "bs_tx_power_dbm": 43.0,
    "bs_antenna_gain_dbi": 18.0,
    "ms_antenna_gain_dbi": 2.0,
    "path_loss_db": 120.0,
    "channel_bandwidth_hz": 200000.0
  }
}
```

#### Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `bs_tx_power_dbm` | Base Station transmit power in dBm | 43.0 | 0-60 |
| `bs_antenna_gain_dbi` | BS antenna gain in dBi | 18.0 | - |
| `ms_antenna_gain_dbi` | Mobile Station antenna gain in dBi | 2.0 | - |
| `path_loss_db` | Path loss in dB | 120.0 | ≥0 |
| `channel_bandwidth_hz` | Channel bandwidth in Hz | 200000.0 | >0 |

### Enabling dBm Mode

To enable dBm-based power calculations, add the following to your config:

```json
{
  "use_dbm_mode": true
}
```

If not specified, the simulator defaults to the traditional dB-based relative power calculation mode.

## How It Works

### Link Budget Calculation

When `use_dbm_mode` is enabled, the simulator calculates received power using:

```
Rx Power (dBm) = Tx Power (dBm) + Tx Antenna Gain (dBi)
                + Rx Antenna Gain (dBi) - Path Loss (dB)
```

### Thermal Noise Calculation

Thermal noise is calculated based on physical parameters:

```
Noise Power (dBm) = kTB (dBm) + Noise Figure (dB)

Where:
k = Boltzmann constant = 1.380649e-23 J/K
T = Temperature in Kelvin (typically 290K)
B = Channel bandwidth in Hz
```

For GSM (200 kHz bandwidth):
- kTB ≈ -174 + 10*log10(200000) ≈ -121 dBm
- With NF=0dB: Total noise ≈ -121 dBm

### Mode Differences

| Mode | Traditional (dB) | dBm Mode |
|------|------------------|----------|
| **CI** | Uses fixed SNR=20dB | Uses physical thermal noise power |
| **SINR** | Uses fixed SNR=20dB | Uses physical thermal noise power |
| **SNR** | Uses fixed CI=20dB | Uses physical thermal noise power |

## Console Output

When running with dBm parameters, the simulator displays:

```
================================================================================
POWER PARAMETERS (dBm)
================================================================================
BS TX Power:                43.0 dBm
BS Antenna Gain:            18.0 dBi
MS Antenna Gain:            2.0 dBi
Path Loss:                  120.0 dB
Channel Bandwidth:          200 kHz
Received Signal Power:      -57.0 dBm
Received Noise Power:       -121.0 dBm
Resulting SNR (thermal):    64.0 dB
================================================================================
```

## CSV Output

dBm parameters are saved in the metadata section of CSV results:

```csv
# SIMULATION METADATA
...
BS Noise Figure (dB), 0
Temperature (K), 290
BS TX Power (dBm), 43.0
BS Antenna Gain (dBi), 18.0
MS Antenna Gain (dBi), 2.0
Path Loss (dB), 120.0
Channel Bandwidth (kHz), 200.0
...
```

## Usage Examples

### Example 1: Use dBm Mode

Create `config/settings.json` with:

```json
{
  "use_dbm_mode": true,
  "power_parameters": {
    "bs_tx_power_dbm": 43.0,
    "bs_antenna_gain_dbi": 18.0,
    "ms_antenna_gain_dbi": 2.0,
    "path_loss_db": 120.0,
    "channel_bandwidth_hz": 200000.0
  }
  // ... other parameters
}
```

Run:

```bash
python core_simulation.py
```

### Example 2: Traditional Mode (Default)

Simply omit `use_dbm_mode` or set to `false`:

```json
{
  "use_dbm_mode": false
  // ... other parameters
}
```

The simulator will use the original dB-based relative power calculations.

## Important Notes

1. **Backward Compatibility**: The default behavior is unchanged (relative dB mode). Set `use_dbm_mode: true` to enable dBm calculations.

2. **BER Results**: BER should be identical between modes when the effective SNR/CINR ratios are the same, as BER depends on ratios, not absolute power values.

3. **Channel Bandwidth**: The `channel_bandwidth_hz` parameter should match your system's actual channel bandwidth (e.g., 200 kHz for GSM).

4. **Path Loss**: Path loss should include all losses in the link budget (free space path loss, fading margin, penetration loss, etc.).

5. **Sampling Frequency**: `fs_hz` is still used for channel generation and timing, not for thermal noise calculation in dBm mode (uses `channel_bandwidth_hz` instead).

## Testing

To verify the dBm implementation:

```bash
# Test with dBm mode enabled
python core_simulation.py

# Check console output for power information
# Check CSV file for dBm metadata

# Run comparison between old and new modes
# (Expected: Similar BER results for same SINR values)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Power parameters missing" | Add `power_parameters` section to your config |
| "Invalid dBm value" | Ensure `bs_tx_power_dbm` is between 0 and 60 |
| Unexpected BER differences | Verify `channel_bandwidth_hz` matches your system |
| Import errors | Ensure `receiver/power_calc.py` exists in your project |

## Future Enhancements

Potential improvements for the dBm feature:

1. Add interference power specification in dBm
2. Support for frequency-dependent path loss models
3. Add shadowing and fading margins
4. Multi-cell interference modeling with absolute power
5. Uplink/downlink link budget asymmetry

## References

- GSM 05.05: "Radio transmission and reception"
- 3GPP TS 45.005: "Radio transmission and reception"
- Link budget theory: Friis transmission equation
- Thermal noise: kTB noise formula
