## To execute this script, QETpy package should be installed.
## Please follow instructions on this page -> QETpy: https://qetpy.readthedocs.io/en/latest/install.html

## For more information on QETpy package:
# https://github.com/ucbpylegroup/QETpy

import numpy as np
import qetpy as qp

def process_data(trace,recordlength):
    """
    Function to reshape data into (number of traces, record length)
    Parameters
    ----------
    trace : ndarray
        Raw ADC counts

    recordlength: int
        input from the corresponding config file
    Returns
    -------
    data : ndarray
        Reshaped data to put into OF and to calculate PSD
    """
    try:
        traceR = trace.reshape(int(np.floor(len(trace)/recordlength)),recordlength)
    except ValueError:
        trace = trace[:int(np.floor(len(trace)/recordlength))*recordlength,]
        traceR = trace.reshape(int(np.floor(len(trace)/recordlength)),recordlength)

    data= traceR-np.expand_dims(traceR[:,100:2000].mean(axis=1), axis=1)
    return data

def calculate_psd(traces, processed):
    """
    Function to calculate Pulse Spectral Density (PSD)
    Parameters
    ----------
    trace : ndarray
        Raw ADC counts
    processed : bool
        Set it to True if the trace is processed i.e. reshaped
    Returns
    -------
    psd : ndarray
        PSD of the input trace , unfolded
    """
    if processed:
        psd = qp.calc_psd(traces, 1/2e-9, False)
    else:
        psd = qp.calc_psd(process_data(traces), 1/2e-9, False)
    return psd

def create_template(traces):
    """
    Function to create template for the OF.
    Parameters
    ----------
    trace : ndarray
        Raw ADC counts that are t0 aligned and preferrably without pileups and glitches.
        The trace must be processed i.e. reshaped. Use process_data(trace) if the traces are not reshaped.
    Returns
    -------
    template_norm : ndarray
        Normalized template
    """
    template = np.mean(traces, axis = 0)
    template_norm= -template/min(template)
    return template

def OF_calc(signal, template_norm, psd_noise, fs):
    """
    Function to calculate Optimum filter with time delay
    Attributes
    ----------
    signal : ndarray
        The signal that we want to apply the Optimum Filter to.
    psd_noise : ndarray
        The two-sided psd that will be used to describe the noise in the signal
    fs : float
        The sampling rate of the data (in Hz).

    Returns
    -------
    amp : float
        The optimum amplitude calculated for the trace with respect to template's amplitude.
    t0 : float
        The time shift calculated for the pulse with respect to the template's t0 in s.
    chi2 : float
        The chi^2 value calculated from the optimum filter.
    pileup_params : list
        Optimum amplitude, time shift with respect to the main peak and chi^2 value calculated for the pileup.
    """

    amp, t0, chi2 = [], [], []
    amp_pileup, t0_pileup, chi2_pileup = [], [], []
    for trace in np.arange(len(signal)):
        OF = qp.OptimumFilter(signal[trace], template_norm, psd_noise[1], fs) # initialize the OptimumFilter class
        amp_withdelay, t0_withdelay, chi2_withdelay= OF.ofamp_withdelay() # t0_withdelay : time shift calculated for the pulse (in s).
        amp.append(amp_withdelay)
        t0.append(t0_withdelay)
        chi2.append(chi2_withdelay)

        # pileup
        amp_pu, t0_pu, chi2_pu = OF_.ofamp_pileup_iterative(amp_withdelay, t0_withdelay)
        amp_pileup.append(amp_pu)
        t0_pileup.append(t0_pu)
        chi2_pileup.append(chi2_pu)
    pileup_params = [amp_pileup, t0_pileup, chi2_pileup]

    return amp, t0, chi2, pileup_params
