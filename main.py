"""
This is a file containing the main execution for the calculation of optical injected laser parameters
based on the paper from Murakami 2003
link:
10.1109/JQE.2003.817583

author: Damiano Massella
date 18/03/2021
"""

import functions as fn
import numpy as np
import matplotlib.pyplot as plt

c_light = 299792458  # light speed m/s
nu = 2  # percentage of external field enetering the cavity

parameters = {
    "g": 8.4 * 1e-6,  # linear gain [m3/ns]
    "Nth": 2e8,  # Carrier density at threshold []
    "alpha": 3,  # Linewidth enhancement factor [] # from Yao phd thesis
    "gammaN": 1 / 2.04,  # Carrier recombination rate[1/ns]
    "gammaP": 1 / (1.927 * 1e-3),  # Photon decay rate[1/ns]
    "k": c_light / 3.6 / 2e-3 * 1e-9,
    # Coupling factor [1/ns]  or longitudinal mode spacing # from 10.1109/JQE.1985.1072760
    "Sinj": nu
    * 600,  # Injected field [J/ns] # find a way to express this in normalized field
    "phi_inj": 0,  # phase of the injected beam
    "J": 3e8,  # injected current in the slave laser [m3/ns]
    "det_fr": -20,  # detuning on frequency [Ghz] = [1/ns]
}
# TODO find out the parameters for our ring resonator laser
# TODO find how gammaN and gammaP influence the outcome
# TODO implement a sinusoidal J inj current to study frequency response

if __name__ == "__main__":
    print("run")
    a = fn.solve_rate_equations(parameters, time=10)
    print(fn.is_steady_state(a, False))
    print(fn.is_steady_state_fft(a, 90000, plot=True))
    fn.pretty_plotting(a)
    variable_sweep = "J"
    print(
        parameters["k"]
        * parameters["Sinj"]
        / 600
        * np.sqrt(1 + parameters["alpha"] ** 2)
    )
    # sweep_res = fn.sweep_steady_state(
    #     parameters, variable_sweep, np.logspace(8, 10, 40)
    # )
    # fn.pretty_plotting(sweep_res, variable_sweep)
    # result_2d = fn.create_2d_sweep(
    #     parameters,
    #     "det_fr",
    #     "Sinj",
    #     np.linspace(-100, 80, 10),
    #     np.logspace(-2, 1, 10) * 600,
    #     filep=".//fft_peaks_test",
    # )

    # fn.plot_2d_pickle(".//real_ring_fine_sweep", "phi", direct_plot=True)
    plt.show()
