import scipy.io as spio
import numpy as np
from typing import Dict, Tuple

def load_channel_matrix(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MIMO channel matrix H from MATLAB .mat file.
    
    The file should contain 4 submatrices: h11, h12, h21, h22.
    
    Args:
        file_path: Path to the MATLAB .mat file
        
    Returns:
        Tuple containing four submatrices (h11, h12, h21, h22)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        KeyError: If required matrices are not found in the file
        ValueError: If matrices have incompatible dimensions
    """
    try:
        # Load MATLAB file
        mat_data = spio.loadmat(file_path)
        
        # Extract matrices (adjust keys according to your MATLAB variable names)
        h11 = mat_data['h11']  # Replace with actual key names if different
        h12 = mat_data['h12']
        h21 = mat_data['h21']
        h22 = mat_data['h22']
        
        # Validate matrix shapes
        if h11.shape != h12.shape or h11.shape != h21.shape or h11.shape != h22.shape:
            raise ValueError("All H matrix components must have the same dimensions")
            
        return h11, h12, h21, h22
        
    except FileNotFoundError:
        raise FileNotFoundError(f"MATLAB file not found at {file_path}")
    except KeyError as e:
        raise KeyError(f"Required matrix not found in MATLAB file: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error loading channel matrix: {str(e)}")