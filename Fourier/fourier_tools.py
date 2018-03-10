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

def calc_cos(t, f, a, p):
    """Calculate cosine function"""
    return a * np.cos(2*np.pi*(f*t+p/360))

def calc_periodic_model(t, freq, amp, pha, x0):
    """Calculate periodic model using input "time" array, t, to recursively add
    together cosine functions built from each sequential value in the 
    frequency array, freq, amplitude array, amp, and phase array, pha, with
    the last set of array values including the addition of the x-intercept, a
    single float value, x0.
    """
    if len(freq) == 1 and len(amp) == 1 and len(pha) == 1:
        return calc_cos(t, freq[0], amp[0], pha[0]) + x0
    elif len(freq) == len(amp) and len(freq) == len(pha):
        return calc_cos(t, freq[0], amp[0], pha[0]) + \
                calc_periodic_model(t, freq[1:], amp[1:], pha[1:], x0)
    else:
        print("Error: freq, amp, and, pha arrays MUST ALL be same length")
        print("Returning numpy array of zeros")
        return np.zeros(t.shape)

def get_ind_linspace(step, num, t0=0):
    """Return independent variable array using np.linspace"""
    return np.linspace(0, step*(num-1), num) + t0

def get_model_ind(**parind):
    """Generate independent variable.  Currently only a 1-D, linearly-spaced
    array is supported via np.linspace.  Add new methods, e.g. random, etc.
    """
    if 'linspace' in parind:
        return get_ind_linspace(**parind['linspace'])
    else:
        print("Unrecognized method for generating independent variable")

def get_model_dep(t, **pardep):
    """Generate dependent variable."""
    if 'periodic' in pardep:
        return calc_periodic_model(t, **pardep['periodic'])

def get_input_model(**parmod):
    """Generate model data.  Independent variable, t, is built first as it is
    required to build the dependent variable, x.
    """
    model = {}
    if 't' in parmod:
        model['t'] = get_model_ind(**parmod['t'])
        if 'x' in parmod:
            model['x'] = get_model_dep(model['t'], **parmod['x'])
        return model

def get_input_data(**parinp):
    """Load and/or generate input data for analysis."""
    data = {}
    data['input'] = {}
    if 'model' in parinp:
        data['input']['model'] = get_input_model(**parinp['model'])
    return data

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

def calc_basis_periodic(freq, t, orthnorm_test=False):
    """Returns an orthonormal pair of periodic basis functions with a frequency
    of freq in Hz using the independent variable, t, a numpy array. Resulting
    array has dimensions = [t.shape[0], 2]. The modified Gram-Schmidt process
    is used to ensure orthonormality is accurate to within numerical precision,
    which can be tested by setting orthnorm_test=True.
    """
    basis = np.vstack((np.cos(2*np.pi*freq*t), -np.sin(2*np.pi*freq*t))).T
    otb = calc_orthonormal_basis(basis)
    if orthnorm_test:
        print_orthonormal_test(otb)
    return otb

def calc_rdft_ampphs(t, x, freq):
    """Returns the amplitude and phase (in radians) of the real-valued
    dependent variable, a numpy array, x, using a Discrete-Fourier Transform,
    where freq is the discrete frequency value to analyze in cycles/ax_unit,
    and t is the independent variable corresponding to x, a numpy array.
    """
    # Add ability to handle multiple frequencies
    ampphs = {}
    orthnormbas = calc_basis_periodic(freq[0], t, orthnorm_test=True)
    onb_amps = x.dot(orthnormbas)
    ampphs['amp'] = np.sqrt((2/t.shape[0])*sum(onb_amps**2))
    ampphs['pha'] = np.arctan2(onb_amps[1], onb_amps[0])
    return ampphs

def isarray_complex(xa):
    """Returns true if input array, xa, contains at least one complex value;
    otherwise returns false.
    """
    return any([isinstance(xv, complex) for xv in xa])

def calc_fft(x, outkeys=['freq', 'amp'], rescale=True, \
        axis=-1, step=1, zeropad=2
):
    """Computes the Fast-Fourier transform of the input numpy array, x, along
    the dimension, axis, using either rfft or fft, depending on whether the
    input array, x, contains all real values or not, respectively.
    Optional input parameters:
    step is the sampling interval along the axis dimension of the input array,
    zeropad is an integer, length-multiplier that determines the number of
    zeros appended to the end of the input numpy array prior to applying the
    FFT, such that the total number of zeros appended is equal to
    ns*(zeropad-1).
    FFT output can be returned in rescaled form (rescale=True), such that
    the amplitude values match up with the actual amplitude in the input
    array; or, the FFT output can be returned exactly as produced by numpy's
    FFT algorithm (rescale=False).
    Returns a dictionary, fftout, with each of its keys defined by the valid
    keys provided in the input list, outkeys, and each value is the
    corresponding 1D numpy array with ns*zeropad rows.
    Valid strings for outkeys are as follows:
    "freq" for frequency vector,
    "spec" for full, complex-valued FFT output
    "amp" for amplitude spectrum,
    "pha" for phase spectrum in radians
    If 'spec' is in outkeys, then only the full spectrum is returned, while
    'amp' and 'pha' are ignored, if present.  Otherwise, "amp" and/or "pha"
    are returned, if present.
    Thus, the resulting output dictionary will have at most three keys:
        ['freq' AND/OR ('spec' OR ('amp' AND/OR 'pha))']
    """
    # ToDo: Make an option to zero pad x up to the nearest power of 2, 
    # for speed!!!
    spec['ns'] = x.shape[axis]
    spec['nf'] = spec['ns'] * int(zeropad)
    spec = {'iscomplex': isarray_complex(x)}
    if spec['iscomplex']:
        if 'freq' in outkeys:
            spec['freq'] = np.fft.fftfreq(spec['nf'], d=step)
        fftout = np.fft.fft(x, n=spec['nf'], axis=axis)
    else:
        if 'freq' in outkeys:
            spec['freq'] = np.fft.rfftfreq(spec['nf'], d=step)
        fftout = np.fft.rfft(x, n=spec['nf'], axis=axis)
    if rescale:
        fftout /= spec['ns']
        fftout[1:] *= 2
    if 'spec' in outkeys:
        spec['spec'] = fftout
    else:
        if 'amp' in outkeys:
            spec['amp'] = np.abs(fftout)
        if 'pha' in outkeys:
            spec['pha'] = np.angle(fftout)
    return spec

def calc_invfft(spec, ns, iscomplex, axis=-1):
    """Computes the inverse Fast-Fourier transform of the input numpy array,
    spec, along the dimension, axis, using either irfft or ifft, depending on
    whether the original input array, x, contained all real values or not,
    respectively.  In other words, if iscomplex=True, use ifft, otherwise use
    irfft.
    Returns an array, invfft, containing the output from irfft or ifft
    """
    if iscomplex:
        ifftout = np.fft.ifft(x, n=ns, axis=axis)
    else:
        ifftout = np.fft.irfft(x, n=ns, axis=axis)
    return ifftout

def get_analysis_fourier(datinp, **parana):
    """Get fourier analysis data, including real discrete fourier transform
    (rdft) and fast-fourier transform (fft).
    """
    ditype = parana['input']
    datana = {}
    if 'rdft' in parana:
        datana['rdft'] = calc_rdft_ampphs(datinp[ditype]['t'], \
                datinp[ditype]['x'], \
                **parana['rdft']
        )
    if 'fft' in parana:
        datana['fft'] = calc_fft(datinp[ditype]['x'], \
                **parana['fft']
        )
    return datana

def print_rdft_tests(datdft, **parmod):
    """Print tests to check the difference between the amplitude and phase
    parameters used to build the model data, pm_amp and pm_pha, respectively,
    and the amplitude and phase parameters calculated using the discrete
    Fourier transform, datdft['amp'] and datdft['pha'], respectively.
    All phase values should be converted to degrees!
    """
    pm_amp = parmod['x']['periodic']['amp'][0]
    pm_pha = parmod['x']['periodic']['pha'][0]
    df_pha = 180 * datdft['pha'] / np.pi
    print("\nDiscrete Fourier Transform test results:")
    print("Model Input Amplitude - DFT Output Amplitude = {}".format( \
            pm_amp-datdft['amp']
    ))
    print("Model Input Phase - DFT Output Phase = {}\n".format(pm_pha-df_pha))

def print_fft_tests(datana):
    """Print tests to check the output of the real Fast-Fourier Transform"""
    maxampind = data['analysis']['fft']['amp'].argmax()
    print("Amplitude at f=0 = {}".format( \
            data['analysis']['fft']['amp'][0]
    ))
    print("Frequency at max. amplitude = {} cycles/t_units".format( \
            data['analysis']['fft']['freq'][maxampind]
    ))
    print("Maximum amplitude = {}".format( \
            data['analysis']['fft']['amp'][maxampind]
    ))
    print("Phase at max. amplitude = {} degrees".format( \
            180*data['analysis']['fft']['pha'][maxampind]/np.pi
    ))

def plot_spectrum(fft_dict, **parout):
    """Plot the Fourier spectrum obtained from applying the FFT to an input
    numpy array, i.e. from the output of calc_fft defined above.
    """
    # Fix 'complex' key scenario to plot real or imaginary part, and change
    # label, etc.
    ax_labels = {'amp': 'Amplitude', 'freq': 'Frequency [Hz]', \
            'pha': 'Phase [degrees]', 'spec': 'Real Part of Spectrum'}
    fig = plt.figure()
    fig.suptitle('FFT Output: {} vs {}'.format( \
            parout['y_key'].capitalize(), parout['x_key'].capitalize()
    ))
    ax = fig.add_subplot(111, xlabel=ax_labels[parout['x_key']], \
            ylabel=ax_labels[parout['y_key']]
    )
    ax.plot(fft_dict[parout['x_key']], fft_dict[parout['y_key']], \
            **parout['params']
    )
    if 'save' in parout:
        plt.savefig(parout['save']['name'])

def generate_output(datana, parmod, **parout):
    if parout['rdft_tests']:
        print_rdft_tests(datana['rdft'], **parmod)
    if parout['fft_tests']:
        print_fft_tests(data['analysis']['fft'])
    if parout['figure']['type'] == 'spectrum':
        plot_spectrum(datana['fft'], **parout['figure'])

## III) If run from command line, execute script below here
if __name__ == "__main__":
    # Testing parameters
    # Model data
    params = { \
        'input': { \
            'model': { \
                't': { \
                    'linspace': { \
                        'num': 2000, \
                        'step': 0.01
                    }
                }, \
                'x': { \
                    'periodic': { \
                        'freq': [2.7, 16], \
                        'pha': [50, 100], \
                        'amp': [8, 4], \
                        'x0': 5
                    }
                }
            }
        }, \
        'analysis': { \
            'input': 'model', \
            'rdft': {'freq': [2.7, 16]}, \
            'fft': { \
                'outkeys': ['freq', 'amp', 'pha'], \
                'step': 0.01, \
                'rescale': 1, \
                'zeropad': 2
            }
        }, \
        'output': { \
            'rdft_tests': 1, \
            'fft_tests': 1, \
            'figure': { \
                'type': 'spectrum', \
                'x_key': 'freq', \
                'y_key': 'amp', \
                'params': { \
                    'linestyle': '-',\
                    'linewidth': 2,\
                    'color': 'k', \
                    'marker': '*',\
                    'markersize': 8
                }, \
                'save': { \
                    'name': 'test_fft.png'
                }
            }
        }
    }

    # Input: build data dictionary, including independent, t, and dependendent,
    # x, variables for testing
    data = get_input_data(**params['input'])

    # Analysis: calculate the Discrete Fourier Transform (DFT) and/or the
    # Fast-Fourier Transform (FFT) to dependent variable
    data['analysis'] = get_analysis_fourier(data['input'], **params['analysis'])

    # Output: print results, build/save figures, etc.
    generate_output(data['analysis'], params['input']['model'],
            **params['output']
    )
