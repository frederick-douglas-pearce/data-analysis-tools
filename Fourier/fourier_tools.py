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
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

## Define functions
# Calculate Discrete Fourier Transform (DFT) of numpy array
# i.e. arbitrary frequency(ies) and their corresponding amplitude(s)
def print_orthonormal_test(f, basis):
    """Print test of ortho-normality applied to input numpy array, basis, that
    has a frequency, f.
    """
    print("\nOrtho-normal Basis test for f = {}".format(f))
    print("Inner product of basis vector matrix with itself = \n{}" \
            .format((basis.T).dot(basis))
    )
    print("Basis vectors are ortho-normal if result is identity matrix."
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
        print_orthonormal_test(freq, otb)
    return otb

def calc_rdft_ampphs(t, x, freq):
    """Returns a dict with lists containing the amplitude and phase (radians)
    of the real-valued dependent variable, a numpy array, x, using a
    Discrete-Fourier Transform, where freq is a list of discrete frequency
    values to analyze in cycles/t_unit and t, a numpy array, is the independent
    variable corresponding to x.
    """
    # Vectorize implementation for multiple frequencies
    ampphs = defaultdict(list)
    for f in freq:
        orthnormbas = calc_basis_periodic(f, t, orthnorm_test=True)
        onb_amps = x.dot(orthnormbas)
        ampphs['amp'].append(np.sqrt((2/t.shape[0])*sum(onb_amps**2)))
        ampphs['pha'].append(np.arctan2(onb_amps[1], onb_amps[0]))
    ampphs['amp'] = np.array(ampphs['amp'])
    ampphs['pha'] = np.array(ampphs['pha'])
    return ampphs

def isarray_complex(xa):
    """Returns true if input array, xa, contains at least one complex value;
    otherwise returns false.
    """
    return any([isinstance(xv, complex) for xv in xa])

def calc_array_amppha(arr):
    """Return amplitude and phase of complex input array, arr."""
    return (np.abs(arr), np.angle(arr))

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
    spec = {'iscomplex': isarray_complex(x)}
    spec['ns'] = x.shape[axis]
    spec['nf'] = spec['ns'] * int(zeropad)
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

def calc_ifft(spec, ns, iscomplex, axis=-1):
    """Computes the inverse Fast-Fourier transform of the input numpy array,
    spec, along the dimension, axis, using either irfft or ifft, depending on
    whether the original input array, x, contained all real values or not,
    respectively, i.e., if iscomplex=True, use ifft, otherwise use irfft.
    The input parameter, ns, specifies the length of the ifft output, n, see
    ifft or irfft documentation for details.
    Returns an array, invfft, containing the output from irfft or ifft.
    """
    if iscomplex:
        return np.fft.ifft(spec, n=ns, axis=axis)
    else:
        return np.fft.irfft(spec, n=ns, axis=axis)

def get_analysis_fourier(datinp, **parana):
    """Get fourier analysis data, including real discrete fourier transform
    (rdft) and fast-fourier transform (fft).
    """
    ditype = parana['input']
    datana = {}
    if 'rdft' in parana:
        datana['rdft'] = calc_rdft_ampphs( \
                datinp[ditype]['t'], datinp[ditype]['x'], **parana['rdft']
        )
    if 'fft' in parana:
        datana['fft'] = calc_fft(datinp[ditype]['x'], **parana['fft'])
        if 'ifft' in parana['fft']['outkeys'] and 'spec' in datana['fft']:
            datana['fft']['ifft'] = calc_ifft(datana['fft']['spec'],
                    datana['fft']['ns'], datana['fft']['iscomplex']
            )
    #print(datana)
    return datana

def print_rdft_tests(datdft, **parmod):
    """Print tests to check the difference between the amplitude and phase
    parameters used to build the model data, 'amp' and 'pha' in
    parmod['x']['periodic']['amp'], respectively, and the amplitude and phase
    parameters calculated using the discrete Fourier transform, datdft['amp']
    and datdft['pha'], respectively.
    All phase values should be converted to degrees!
    """
    for i, f in enumerate(parmod['x']['periodic']['freq']):
        print("\nDiscrete Fourier Transform test results for f = {}".format(f))
        print("Model Input Amplitude - DFT Output Amplitude = {}".format( \
                parmod['x']['periodic']['amp'][i]-datdft['amp'][i]
        ))
        df_pha = 180 * datdft['pha'][i] / np.pi
        print("Model Input Phase - DFT Output Phase = {}".format( \
                parmod['x']['periodic']['pha'][i]-df_pha
        ))
    print("")

def print_fft_tests(data):
    """Print tests to check the output of the real Fast-Fourier Transform"""
    if 'spec' in data['analysis']['fft']:
        amp, pha = calc_array_amppha(data['analysis']['fft']['spec'])
    else:
        amp = data['analysis']['fft']['amp']
        pha = 180 * data['analysis']['fft']['pha'] / np.pi
    print(amp,pha)
    maxampind = amp[1:].argmax()
    # Improve this test, e.g. test multiple frequencies, compare with model
    # values, etc.
    print("Amplitude at f=0 = {}".format(amp[0]))
    print("Frequency at max. amplitude = {} cycles/t_unit".format( \
            data['analysis']['fft']['freq'][maxampind]
    ))
    print("Maximum amplitude = {}".format(amp[maxampind]))
    print("Phase at max. amplitude = {} degrees\n".format(pha[maxampind]))

def plot_data(fig, data, **parsub):
    """Plot the data, both before and (optionally) after Fourier analysis."""
    ax_labels = {'t': 'Indepedent (t)', 'x': 'Dependent (x)', 'ifft': 'Inverse FFT'}
    ax = fig.add_subplot(parsub['axid'], xlabel=ax_labels[parsub['x_key']], \
            ylabel=ax_labels[parsub['y_key']], title='Data'
    )
    if parsub['y_key'] == 'x':
        ax.plot(data['input']['model'][parsub['x_key']], \
                data['input']['model'][parsub['y_key']], **parsub['params']
        )
    if 'ifft' in data['analysis']['fft']:
        parsub['params'].update({'color': 'r', 'linestyle': '--'})
        ax.plot(data['input']['model'][parsub['x_key']], \
                data['analysis']['fft']['ifft'], **parsub['params']
        )

def plot_spectrum(fig, fft_dict, **parsub):
    """Plot the Fourier spectrum obtained from applying the FFT to an input
    numpy array, i.e. from the output of calc_fft defined above.
    """
    # Fix 'complex' key scenario to plot real or imaginary part, and change
    # label, etc.
    ax_labels = {'amp': 'Amplitude', 'freq': 'Frequency [Hz]', \
            'pha': 'Phase [degrees]', 'spec': 'Real (b) + Imaginary (r)'}
    ax = fig.add_subplot(parsub['axid'], xlabel=ax_labels[parsub['x_key']], \
            ylabel=ax_labels[parsub['y_key']], title='Spectrum'
    )
    # If y_key is 'spec', then line colors are overwritten in parsub['params']
    # so that the real part of FFT output is blue and the imaginary part is red
    if parsub['y_key'] == 'spec':
        parsub['params'].update({'color': 'b'})
        ax.plot(fft_dict[parsub['x_key']], np.real(fft_dict[parsub['y_key']]),\
                **parsub['params']
        )
        parsub['params'].update({'color': 'r'})
        ax.plot(fft_dict[parsub['x_key']], np.imag(fft_dict[parsub['y_key']]),\
                **parsub['params']
        )
    else:
        ax.plot(fft_dict[parsub['x_key']], fft_dict[parsub['y_key']], \
                **parsub['params']
        )

def make_figure_subplots(fig, data, **parfig):
    """Make subplots by first, identifying keys in parfig that define subplots,
    then looping through the keys to build each subplot
    """
    spkey = [spk for spk in parfig if spk.startswith("subplot")]
    for spk in spkey:
        if parfig[spk]['type'] == 'data':
            plot_data(fig, data, **parfig[spk])
        elif parfig[spk]['type'] == 'spectrum':
            plot_spectrum(fig, data['analysis']['fft'], **parfig[spk])

def save_or_show_figure(**parfig):
    """If "save" is a key in parfig, then save the figure;
    otherwise, show the figure."""
    if 'save' in parfig:
        plt.savefig(parfig['save']['name'], **parfig['save']['params'])
    else:
        plt.show()

def make_figure(data, **parfig):
    """Make figure"""
    fig = plt.figure(**parfig['params'])
    #fig.suptitle('FFT Analysis')
    make_figure_subplots(fig, data, **parfig)
    fig.tight_layout()
    save_or_show_figure(**parfig)

def generate_output(data, parmod, **parout):
    if parout['rdft_tests']:
        print_rdft_tests(data['analysis']['rdft'], **parmod)
    if parout['fft_tests']:
        print_fft_tests(data['analysis']['fft'])
    if 'figure' in parout:
        make_figure(data, **parout['figure'])

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
                #'outkeys': ['freq', 'amp', 'pha'], \
                'outkeys': ['freq', 'spec', 'ifft'], \
                'step': 0.01, \
                'rescale': 0, \
                'zeropad': 1
            }
        }, \
        'output': { \
            'rdft_tests': 1, \
            'fft_tests': 1, \
            'figure': { \
                'params': { \
                    'figsize': (10, 6)
                }, \
                'subplot1': { \
                    'axid': 211, \
                    'type': 'data', \
                    'x_key': 't', \
                    'y_key': 'x', \
                    'params': { \
                        'linestyle': '-',\
                        'linewidth': 1,\
                        'color': 'k', \
                        'marker': '*',\
                        'markersize': 4
                    }
                }, \
                'subplot2': { \
                    'axid': 212, \
                    'type': 'spectrum', \
                    'x_key': 'freq', \
                    'y_key': 'spec', \
                    'params': { \
                        'linestyle': '-',\
                        'linewidth': 1,\
                        'color': 'k', \
                        'marker': '*',\
                        'markersize': 4
                    }
                }, \
                'save': { \
                    'name': 'test_x_fft_real.png', \
                    'params': { \
                        'dpi': 300
                    }
                }
            }
        }
    }

    # Input: build data dictionary, including independent, t, and dependendent,
    # x, variables for testing
    # Analysis: calculate the Discrete Fourier Transform (DFT) and/or the
    # Fast-Fourier Transform (FFT) to dependent variable
    # Output: print results, build/save figures, etc.
    data = get_input_data(**params['input'])
    data['analysis'] = get_analysis_fourier(data['input'], **params['analysis'])
    generate_output(data, params['input']['model'],
            **params['output']
    )
