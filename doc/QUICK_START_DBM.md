# Quick Start Guide: dBm Mode

## How to Enable dBm Mode

Edit `config/settings.json` and change `use_dbm_mode` to `true`:

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
}
```

## Mode Comparison

### Traditional Mode (`use_dbm_mode: false`)
- Uses relative dB ratios
- No physical power calculations shown
- Backward compatible with previous results

### dBm Mode (`use_dbm_mode: true`)
- Uses absolute power values in dBm
- Displays link budget information
- Calculates thermal noise physically (kTB)
- Shows received signal and noise power

## Console Output Comparison

### Traditional Mode
```
================================================================================
SIMULATION CONFIGURATION
================================================================================
Modulation Type:            BPSK
Channel Model:              TU50
Target Ratio Range:         -5.0 to 21.0 dB (step: 2.0 dB)
================================================================================
STARTING SIMULATION
================================================================================
```

### dBm Mode
```
================================================================================
SIMULATION CONFIGURATION
================================================================================
Modulation Type:            BPSK
Channel Model:              TU50
Target Ratio Range:         -5.0 to 21.0 dB (step: 2.0 dB)
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
STARTING SIMULATION
================================================================================
```

## Recommended Settings

### For GSM 900MHz:
```json
"power_parameters": {
  "bs_tx_power_dbm": 43.0,
  "bs_antenna_gain_dbi": 18.0,
  "ms_antenna_gain_dbi": 2.0,
  "path_loss_db": 120.0,
  "channel_bandwidth_hz": 200000.0
}
```

### For GSM 1800MHz:
```json
"power_parameters": {
  "bs_tx_power_dbm": 39.0,
  "bs_antenna_gain_dbi": 18.0,
  "ms_antenna_gain_dbi": -1.0,
  "path_loss_db": 126.0,
  "channel_bandwidth_hz": 200000.0
}
```

### For Laboratory Testing:
```json
"power_parameters": {
  "bs_tx_power_dbm": 0.0,
  "bs_antenna_gain_dbi": 0.0,
  "ms_antenna_gain_dbi": 0.0,
  "path_loss_db": 0.0,
  "channel_bandwidth_hz": 200000.0
}
```

## Parameter Descriptions

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `bs_tx_power_dbm` | Base Station transmit power | 30-43 dBm (GSM macro) |
| `bs_antenna_gain_dbi` | BS antenna directional gain | 0-18 dBi |
| `ms_antenna_gain_dbi` | Mobile station antenna gain | -1 to 3 dBi |
| `path_loss_db` | Total path loss including fading | 90-140 dB |
| `channel_bandwidth_hz` | Channel RF bandwidth | 200 kHz (GSM) |

## Path Loss Estimation

For quick estimation, you can use:

```
Path Loss (dB) = 128.1 + 37.6 * log10(distance_km)
```

Example: 1 km distance → 128.1 + 37.6 ≈ 166 dB

## Switching Back

To switch back to traditional mode, simply change:
```json
"use_dbm_mode": false
```

No other changes needed!
