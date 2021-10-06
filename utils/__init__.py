import os
import numpy as np



def yield_data_file(data_dir):
    for file_name in os.listdir(data_dir):
        yield os.path.join(data_dir, file_name)


def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()
