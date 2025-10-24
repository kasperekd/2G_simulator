import scipy.io as spio
def load_quadriga_channel(file_path: str):
    # FIXME: This function and its usage should be replaced by a proper 3GPP channel model generator.
    mat_data = spio.loadmat(file_path)
    return mat_data['Ht11'], mat_data['Ht12'], mat_data['Ht21'], mat_data['Ht22']