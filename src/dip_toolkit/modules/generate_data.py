# modules/generate_data.py

import numpy as np


class GenerateData:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def generate_halfmoon(
        self,
        n_samples=1000,
        rad_min=50,
        rad_max=100,
        ang_min=45,
        ang_max=220,
        dist_type="normal",
        rad_std=10,
        ang_std=30.0,
        x0=0.0,
        y0=0.0,
        return_as_array=False,
    ):
        """
        Generate 2D data in a halfmoon shape using polar coordinate sampling.

        Parameters:
            n_samples (int): Number of points to generate.
            rad_min (float): Minimum radius.
            rad_max (float): Maximum radius.
            ang_min (float): Minimum angle in degrees.
            ang_max (float): Maximum angle in degrees.
            dist_type (str): 'uniform' or 'normal'.
            rad_std (float): Std dev for radius (used if dist_type='normal').
            ang_std (float): Std dev for angle (used if dist_type='normal').
            x0 (float): X offset of the moon.
            y0 (float): Y offset of the moon.
            return_as_array (bool): If True, returns an Nx2 array.
                Otherwise, returns an (x, y) tuple.

        Returns:
            np.ndarray or tuple of np.ndarray: Data points (x, y).
        """

        dist_type = dist_type.lower()
        if dist_type not in ["uniform", "normal"]:
            raise ValueError("dist_type must be either 'uniform' or 'normal'.")

        if dist_type == "uniform":
            r = np.random.uniform(rad_min, rad_max, size=n_samples)
            t = np.random.uniform(ang_min, ang_max, size=n_samples)
        else:  # normal
            r_mean = (rad_max + rad_min) / 2.0
            t_mean = (ang_max + ang_min) / 2.0
            r = np.random.normal(loc=r_mean, scale=rad_std, size=n_samples)
            t = np.random.normal(loc=t_mean, scale=ang_std, size=n_samples)
            r = np.clip(r, a_min=0, a_max=None)  # avoid negative radii

        x = x0 + r * np.cos(np.deg2rad(t))
        y = y0 + r * np.sin(np.deg2rad(t))

        if return_as_array:
            return np.vstack((x, y)).T
        else:
            return x.reshape(-1, 1), y.reshape(-1, 1)
