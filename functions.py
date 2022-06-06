"""
This is a file containing functions for the calculation of optical injected laser parameters
based on the paper from Murakami 2003
link:
10.1109/JQE.2003.817583

author: Damiano Massella
date 18/03/2021
"""

from unittest import result
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy import signal


def solve_rate_equations(params, time=1):
    """
    This fuction solves teh equation system of the laser in steady state conditions.

    :param params: dictionary containing all the parameters of the laser,
    :param time: simulation time in ns
    :return:

    """
    Ntr = params["Nth"]
    st1 = 1
    phit1 = 0
    nt1 = 0
    dt = 0.0001  # time step for integration
    i = 0

    def calculate_derivatives(st_start, phi_start, n0):
        """This function calculate the derivatives"""

        gain = params["g"] * (n0 - Ntr)
        stprime = 0.5 * gain * st_start + params["k"] * params["Sinj"] * np.cos(
            phi_start - params["phi_inj"]
        )
        phi_prime = (
            params["alpha"] / 2 * gain
            - params["k"]
            * params["Sinj"]
            / st_start
            * np.sin(phi_start - params["phi_inj"])
            - params["det_fr"]
        )
        dn_prime = (
            params["J"]
            - params["gammaN"] * n0
            - (params["gammaP"] + gain) * st_start**2
        )
        return stprime, phi_prime, dn_prime

    results_lists = {"S": [], "phi": [], "n": []}
    while i < time / dt:
        # appending the values
        results_lists["S"].append(st1)
        results_lists["phi"].append(phit1)
        results_lists["n"].append(nt1)
        # time to update variables
        st0 = st1
        phit0 = phit1
        nt0 = nt1
        # calculate_new
        stp, phip, dnp = calculate_derivatives(st0, phit0, nt0)
        st1 = dt * stp + st0
        phit1 = dt * phip + phit0
        nt1 = dt * dnp + nt0
        i += 1
    results_lists["time"] = np.arange(0, time, dt)
    return results_lists


def pretty_plotting(results, variable="time"):
    """
    This function is used to have some plots that look a bit more decent
    :param results: is a dictionary with all the iteration steps of the integration
    :param variable: is the variable repect to you want to plot the results

    :return none

    """
    fig, ax = plt.subplots(3, 1, sharex=True)
    i = 0
    for key in ["S", "phi", "n"]:
        if key == "S":
            ax[i].plot(results[variable], np.array(results[key]) ** 2)
        else:
            ax[i].plot(results[variable], results[key])
        ax[i].set_ylabel(key)
        i += 1
    ax[2].set_xlabel(variable)


def sweep_steady_state(params, key_sweep, values, time_sim=10):
    """This funciton makes a sweep on the variable key_sweep

    :param params: is a dictionary contaninig all the variables of the laser
    :param key_sweep: is one of the key of teh paramas dictionary that will be used to sweep
    :param values: array containing the values to sweep on
    :param time_sim: how long before reaching the steady state?

    :return dictionary with results
    """
    original_val = params[key_sweep]
    results = {"S": [], "phi": [], "n": [], key_sweep: []}  # results

    for val in tqdm(values):
        params[key_sweep] = val
        r = solve_rate_equations(params, time_sim)
        if is_steady_state_fft(r):
            results[key_sweep].append(val)
            for key in ["S", "phi", "n"]:
                results[key].append(r[key][-1])
        else:
            pass
    params[key_sweep] = original_val
    return results


def is_steady_state(result_vector, plot=False, vocal=True):
    """This function checks if the final state is a steady state
    :param result_vector: vector of results from solve_rate_equations
    :param plot: set to True if you want to se a plot of the solution
    :param vocal: gives a reason for non steady state
    :return bool"""
    precision = 0.001
    if plot:
        pretty_plotting(result_vector)
    dt = result_vector["time"][-1] - result_vector["time"][-2]
    dS = (
        np.abs(result_vector["S"][-1] - result_vector["S"][-2])
        / dt
        / result_vector["S"][-2]
    )
    dphi = (
        np.abs(result_vector["phi"][-1] - result_vector["phi"][-2])
        / dt
        / result_vector["phi"][-2]
    )
    dn = (
        np.abs(result_vector["n"][-1] - result_vector["n"][-2])
        / dt
        / result_vector["n"][-2]
    )
    if dS > precision:
        if vocal:
            print("not stable due to S")
        return False
    elif dphi > precision:
        if vocal:
            print("not stable due to phi")
        return False
    elif dn > precision:
        if vocal:
            print("not stable due to N")
        return False
    else:
        return True


def create_2d_sweep(
    params, var1, var2, values_var1, values_var2, time_sim=10, filep=None
):
    """ ""This funciton creates the map of locking range for the two variables
    :param params: contains all the laser variables
    :param var1: x variable to sweep
    :param var2: y variable to sweep
    :param values_var1: x variable values to test for
    :param values_var2: y variable values to test for
    :param time_sim: is the time of each simulation before considering it finished in [ns]
    :param filep: is the filepath of the file to save the result if not specified the results are not saved


    :return results2D: matrix contaning bool to indicate the reached steady state
    """
    result2D = []
    # TODO make this calculation multiprocess
    for val1 in tqdm(values_var1):
        result1D = []
        for val2 in values_var2:
            params[var1] = val1
            params[var2] = val2
            r1 = solve_rate_equations(params, time_sim)
            r = is_steady_state_fft(r1, vocal=False)
            output = {"S": r1["S"][-1], "phi": r1["phi"][-1], "n": r1["n"][-1], "st": r}
            result1D.append(output)
        result2D.append(result1D)
    result2D = np.array(result2D)
    # TODO add final values to the output
    if filep is not None:
        save_2d_map(result2D, values_var1, values_var2, filep)
    return result2D


def save_2d_map(results, var1, var2, filepath):
    """ ""This function is used to save the results of teh 2D map into a txt file

    :param results :(2d list) is a numpy array containing the result of the 2d sweep
    :param var1 :(array) contains the value of the var1 in the results array
    :param var2 :(array) contains the values of the var2 in the results array
    :param filepath:(str) filepath and name of the file to save the data

    :return None
    """
    df = pd.DataFrame(results, columns=var2)
    df["var1"] = var1
    print(df)
    df.to_pickle(filepath)

    return None


def plot_2d_pickle(path_pick, key=None, direct_plot=False, norm=1):
    """This function load a pickle object made from the previous equation solver
    and plots it in a 2D graph

    :param path_pick: is the filepath of the pickle
    :param key: plot only this key, if not given only stability is plotted
    :param direct_plot: flag if you want to visualize the plot as output

    :return [X,Y,s_results,key_results]: grids of x,y and stability grid, key values grid
    """
    df = pd.read_pickle(path_pick)
    x_values = []
    for column in df:
        if type(column) == float:
            print(column)
            x_values.append(float(column))
            if "s_results" not in locals():
                s_results = np.array([val["st"] for val in df[column]], ndmin=2)
                if key is not None:
                    key_results = np.array([val[key] for val in df[column]], ndmin=2)

            else:
                res = np.array([val["st"] for val in df[column]], ndmin=2)
                s_results = np.append(s_results, res, axis=0)
                if key is not None:
                    res = np.array([val[key] for val in df[column]], ndmin=2)
                    key_results = np.append(key_results, res, axis=0)

    # meshing grid
    X, Y = np.meshgrid(df["var1"].values, x_values / norm)
    if direct_plot:
        plt.figure()
        plt.pcolormesh(X, Y, s_results)
        plt.yscale("log")
    else:
        pass
    return [X, Y, s_results, key_results]


def is_steady_state_fft(result_vector, nstart=50000, plot=False, vocal=False):
    """This function checks if the final state is a steady state
    :param result_vector: vector of results from solve_rate_equations
    :param plot: set to True if you want to se a plot of the solution
    :param vocal: gives a reason for non steady state
    :return bool"""

    dt = result_vector["time"][1] - result_vector["time"][0]
    S = np.array(result_vector["S"][nstart::])
    phi = np.array(result_vector["phi"][nstart::])
    E = S * np.exp(1j * phi)
    E_comp = np.fft.fft(E)
    freq = np.fft.fftfreq(len(result_vector["time"][nstart::]), dt)
    E_comp = 20.0 * np.log10(np.abs(E_comp))
    if plot:
        plt.figure()
        plt.plot(freq, E_comp)
    peaks, _ = signal.find_peaks(E_comp, height=0)

    if len(peaks) > 1:
        # print(peaks)
        return False
    else:
        return True
