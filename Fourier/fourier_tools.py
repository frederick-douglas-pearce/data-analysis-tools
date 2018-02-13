#!/usr/bin/env python3
# Code name: fourier_tools.py
# Brief description: Provides a set of Fourier transform-related data analysis
# tools that are applied to an input numpy array
#
# Requirements: Python (3.4+?)
# numpy ()
# matplotlib (optional,)
#
# Start Date: 1/26/17
# Last Revision:
# Current Version:
# Notes:
#
# Copyright (C) 2017, Frederick D. Pearce, fourier_tools.py

## Import modules
#
import numpy as np
import matplotlib.pyplot as plt

## Define functions
# Calculate Discrete Fourier Transform (DFT) of numpy array
# i.e. arbitrary frequency(ies) and their corresponding amplitude(s)
def print_orthonormal_test(basis):
    """Print test of ortho-normality applied to input numpy array, basis"""
    print("\nOrtho-normal Basis test:")
    print("Inner product of matrix of basis vectors with itself = \n{}" \
            .format((basis.T).dot(basis))
    )
    print("If result is identity matrix, then basis vectors are ortho-normal"
    )

def calc_orthonormal_basis(basis):
    """Returns numerically accurate orthonormal basis vectors given input
    array, basis, using the modified Gram-Schmidt process
    See details at https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
    """
    # Number of sample, m, Number of basis vectors, n
    # Start by dividing first basis vector by its L2 norm
    # For each of the subsequent ii vectors, subtract its projection onto
    # each of the previous jj vectors
    # Finally, divide the ii vectors by their L2 norm
    m, n = basis.shape
    otb = np.zeros((m, n))
    otb[:, 0] = basis[:, 0] / np.linalg.norm(basis[:, 0], ord=2)
    for ii in range(1, n):
        otb[:, ii] = basis[:, ii]
        for jj in range(0, n-1):
            otb[:, ii] = otb[:, ii] - otb[:, ii].dot(otb[:, jj]) / \
                    np.linalg.norm(otb[:, jj], ord=2) * otb[:, jj]
        otb[:, ii] /= np.linalg.norm(otb[:, ii], ord=2)
    return otb

def calc_basis_periodic(ns, freq, delta, orthnorm_test=False):
    """Returns an orthonormal pair of periodic basis functions with a frequency
    of freq in Hz, and ns samples at a sampling rate of delta, as a [ns, 2] 
    numpy array. A test of ortho-normality is run if orthnorm_test=True, with
    its results printed to screen.
    """
    # ToDo: Improve orthonormality using modified Gram-Schmidt process
    t = np.linspace(0, delta*(ns-1), ns)
    basis = np.vstack((np.cos(2*np.pi*freq*t), -np.sin(2*np.pi*freq*t))).T
    #basis /= np.linalg.norm(basis, ord=2, axis=1, keepdims=True)
    otb = calc_orthonormal_basis(basis)
    if orthnorm_test:
        print_orthonormal_test(otb)
    return otb

def calc_rdft_rmsamp(inparr, freq, delta, axis=-1):
    """Returns the amplitude and phase (in radians) of the real-valued input
    numpy array, inparr, using a Discrete-Fourier Transform, where 
    freq is the discrete frequency value in cycles/ax_unit, and
    delta is the sampling interval along the axis dimension of the input array,
    measured in the units defined along axis, ax_unit.
    """
    orthnormbas = calc_basis_periodic(inparr.shape[axis], freq, delta, \
            orthnorm_test=True
    )
    onb_amps = inparr.dot(orthnormbas)
    amplitude = np.sqrt((2/ns)*sum(onb_amps**2))
    phase = np.arctan2(onb_amps[1], onb_amps[0])
    #print(phase*180/np.pi)
    return (amplitude, phase)

def print_rdft_tests(pm_amp, pm_pha, ft_amp, ft_pha):
    """Print tests to check the difference between the amplitude and phase
    parameters used to build the model data, pm_amp and pm_pha, respectively,
    and the amplitude and phase parameters calculated using the discrete
    Fourier transform, ft_amp and ft_pha, respectively.  All input phase values
    should be in degrees!
    """
    print("\nDiscrete Fourier Transform test results:")
    print("Model Input Amplitude - DFT Output Amplitude = {}".format( \
            pm_amp-ft_amp
    ))
    print("Model Input Phase - DFT Output Phase = {}\n".format(pm_pha-ft_pha))

# Calculate Fast-Fourier Transform (FFT) of numpy array, 
# i.e. equally-spaced frequencies and their corresponding amplitudes
def calc_rfft_frqamp( \
        inparr, outkeys, rescale=True, axis=-1, delta=1, zero_pad=2
):
    """ Computes the Fast-Fourier transform of the real-valued input numpy
    array, inparr, along the dimension axis. Optional input parameters:
    delta is the sampling interval along the axis dimension of the input array,
    zero_pad is an integer, length-multiplier that determines the number of
    zeros appended to the end of the input numpy array prior to applying the
    FFT, such that the total number of zeros appended is equal to
    ns*(zero_pad-1).
    FFT output can be returned in rescaled form (rescale=True), such that
    the amplitude values match up with the actual amplitude in the input
    array; or, the FFT output can be returned exactly as produced by numpy's
    FFT algorithm (rescale=False).
    Returns a dictionary, fftout, with each of its keys defined by the valid
    keys provided in the input list, outkeys, and each value is the 
    corresponding 1D numpy array with ns*zero_pad rows.
    Valid strings for outkeys are as follows:
    "frequency" for frequency vector,
    "complex" for full, complex-valued FFT output
    "amplitude" for amplitude spectrum,
    "phase" for phase spectrum in radians
    If 'complex' is in outkeys, then only 'complex is returned, while 'amplitude' 
    and 'phase' are ignored, if present.  Otherwise, "amplitude" and/or "phase"
    are returned, if present.
    Thus, the resulting output dictionary will have at most three keys:
        ['frequency' AND/OR ('complex' OR ('amplitude' AND/OR 'phase))']
    """
    # ToDo: Make an option to zero pad inparr up to the nearest power of 2, 
    # for speed!!!
    ns = inparr.shape[axis]
    nt = ns * int(zero_pad)
    fftout = {}
    # Compute frequency vector
    if 'frequency' in outkeys:
        fftout['frequency'] = np.fft.rfftfreq(nt, d=delta)
    # Compute raw FFT output, then rescale values if rescale=True
    rfft = np.fft.rfft(inparr, n=nt, axis=axis)
    if rescale:
        rfft /= ns
        rfft[1:] *= 2
    # Append FFT output as either complex ("raw"), amplitude, and/or phase
    if 'complex' in outkeys:
        # Append "raw", complex-valued output from FFT
        fftout['complex'] = rfft
    else:
        if 'amplitude' in outkeys:
            fftout['amplitude'] = np.abs(rfft)
        if 'phase' in outkeys:
            fftout['phase'] = np.angle(rfft)
    return fftout

def plot_spectrum(fft_dict, savename, k1='frequency', k2='amplitude'):
    """Plot the Fourier spectrum obtained from applying the FFT to an input
    numpy array, i.e. from the output of calc_rfft_frqamp defined above.
    """
    # Fix 'complex' key scenario to plot real or imaginary part, and change
    # label, etc.
    ax_labels = {'amplitude': 'Amplitude', 'frequency': 'Frequency [Hz]', \
            'phase': 'Phase [degrees]', 'complex': 'Complex Amplitude'}
    plot_params = {'linestyle': '-', 'linewidth': 2, 'color': 'k', \
            'marker': '*', 'markersize': 8
    }
    fig = plt.figure()
    fig.suptitle('FFT Output: {} vs {}'.format( \
            k2.capitalize(), k1.capitalize()
    ))
    ax = fig.add_subplot(111, xlabel=ax_labels[k1], ylabel=ax_labels[k2])
    ax.plot(fft_dict[k1], fft_dict[k2], **plot_params)
    plt.savefig(savename)


## III) If run from command line, execute script below here
if __name__ == "__main__":
    # Testing parameters
    ns = 2000
    delta = 0.01
    pm_freq = 2.7
    pm_phase = 50
    pm_amp = 8
    pm_x0 = 5
    fft_rescale = True
    fft_zeropad = 1
    fft_savename = 'test_fft.png'

    # Build independent, t, and dependendent, x, variables for testing
    # assuming a periodic model that contains a single frequency component at
    # pm_freq in cycles/t_units, with a phase angle of pm_phase in degrees, an
    # amplitude of pm_amp, and an added constant (i.e. DC) value of pm_x0
    t = np.linspace(0, delta*(ns-1), ns)
    x = pm_amp * np.cos(2*np.pi*(pm_freq*t+pm_phase/360)) + pm_x0

    # Use the Discrete Fourier Transform (DFT) to calculate the amplitude of
    # the test data at the frequency value, pm_freq, used to produce x.
    # This should output a value, rdft, that is roughly equal to the input 
    # amplitude value, pm_amp, specified above
    rdft_a, rdft_p = calc_rdft_rmsamp(x, pm_freq, delta)
    print_rdft_tests(pm_amp, pm_phase, rdft_a, 180*rdft_p/np.pi)

    # Use the Fast-Fourier Transform (FFT) to calculate the amplitude spectrum
    # of the test data at a discrete set of frequency values, i.e. values in
    # fft_out['frequency']
    fft_out = calc_rfft_frqamp(x, ['frequency', 'amplitude', 'phase'], \
            rescale=fft_rescale, delta=delta, zero_pad=fft_zeropad
    )
    maxampind = fft_out['amplitude'].argmax()
    print("Amplitude at f=0 = {}".format(fft_out['amplitude'][0]))
    print("Frequency at max. amplitude = {} cycles/t_units".format( \
            fft_out['frequency'][maxampind]
    ))
    print("Maximum amplitude = {}".format(fft_out['amplitude'][maxampind]))
    print("Phase at max. amplitude = {} degrees".format( \
            180*fft_out['phase'][maxampind]/np.pi
    ))
    plot_spectrum(fft_out, fft_savename)

