import pandas as pd
import numpy as np
import os
from collections import Counter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from library.methods import make_filelist, find_maxima, find_closest
from library.functions import pseudo_voigt_asymmetry_coeff, polynome_7 as polynome


def pseudo_voigt_multiprofile_asymmetry_coeff_with_background(x, *args):
    polynome_power = 7
    intensity = np.zeros_like(x)
    pv_args = args[:-(polynome_power + 1)]
    poly_args = args[-(polynome_power + 1):]
    for j in range(len(pv_args) // 5):
        intensity += pseudo_voigt_asymmetry_coeff(x, *pv_args[5 * j:5 * (j + 1)])
    intensity = intensity + polynome(x, *poly_args)
    return intensity


def onpick1(event):
    if isinstance(event.artist, Line2D):
        thisline = event.artist
        xdata = thisline.get_xdata()
        ind = event.ind
        point = np.take(xdata, ind)
        point_theta, point_intensity = point[0], point[1]
        print('2theta = %.2f' % point_theta)
        print('intensity = %.2f' % point_intensity)
        angles_positions.append(np.take(xdata, ind)[0])


hkl_file = '/Users/ivanivanov/Yandex.Disk.localized/PhD/databases/sg_indexes/fm3m_hkl.csv'
hkl_phase = pd.read_csv(hkl_file)

workdir = '/Users/ivanivanov/Yandex.Disk.localized/DESY_2021/HEAs/heating/simulated_undeformed'
loaddir = os.path.join(workdir, 'integrated')
savedir = os.path.join(workdir, 'peaks_data')
filelist = make_filelist(loaddir, 'txt')

filelist = filelist[:]

automatic_peaks_positions = True
pick_peaks_points = False
peaks_positions_plot = False

automatic_background_points = True
pick_background_points = False
background_plot = False

oneframe_fit_plot = False
flow_fit_plot = False

savefile = True

pattern_boundaries = [0, 100]  # 2 thetas

# for automatic peaks search
gauss_smooth_coeff = 1.4  #
intensity_peaks_boundary = 0.008  #

# for background prefitting
intensity_low_boundary = None  # normalized intensity
intensity_low_most_common = 7  # lower than n-th most common of normalized intensity values (rounded by 3)
# will be as the background

# for peaks fitting
delta_angle = 0.1  # 2thetas
delta_intensity = 20  # percents of intensity
remove_peaks = []  # indexes of peaks to remove from savedata
remove_hkl = []  # hkl of necessary phase to remove from savedata

power_of_backround_polynome = 7

if not automatic_peaks_positions and not pick_peaks_points:
    found_peaks_indexes = [759, 1060, 1994, 2543, 2709, 3315, 3720, 3847, 4331, 4665]
if not automatic_background_points and not pick_background_points:
    found_background_points = [213, 1379, 2230, 2896, 3401, 4011, 4391, 4814]

for i, file in enumerate(filelist[::]):
    filename = file[:file.find('.')]

    loadfile = os.path.join(loaddir, file)
    theta, intensity = np.loadtxt(loadfile, unpack=True)

    if pattern_boundaries:
        pattern_lb = find_closest(theta, pattern_boundaries[0])
        pattern_rb = find_closest(theta, pattern_boundaries[1])
        theta = theta[pattern_lb:pattern_rb]
        intensity = intensity[pattern_lb:pattern_rb]

    normalized_intensity = intensity / np.max(intensity)
    x_axis = np.linspace(0, len(normalized_intensity), len(normalized_intensity))

    '''FIND THE PEAKS POSITIONS'''
    if i == 0:
        if pick_peaks_points:
            angles_positions = []
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.set_title('choose necessary peaks', picker=True)
            ax.set_ylabel('Intensity', picker=True)
            line, = ax.plot(theta, np.log(intensity), linestyle='--', color='darkgreen', alpha=0.7, picker=5)

            fig.canvas.mpl_connect('pick_event', onpick1)
            plt.show()
            plt.clf()
            plt.close()

            peaks_indexes = [find_closest(theta, position) for position in angles_positions]
            print(*peaks_indexes, sep=', ')

            plt.subplots(figsize=(12, 5))
            plt.plot(x_axis, intensity, color='darkred')
            plt.scatter(x_axis[peaks_indexes], intensity[peaks_indexes])
            plt.xlabel('indexes')
            plt.ylabel('normalized_intensity')
            plt.show()
            plt.close()

        elif automatic_peaks_positions:
            n_peaks, peaks_indexes = find_maxima(normalized_intensity, gauss_smooth_coeff,
                                                 intensity_peaks_boundary)

        elif not automatic_peaks_positions and not pick_peaks_points:
            peaks_indexes = found_peaks_indexes

    else:
        renewed_peaks_indexes = []
        for ind in peaks_indexes:
            lb = find_closest(theta, theta[ind] - delta_angle)
            rb = find_closest(theta, theta[ind] + delta_angle)
            renewed_peaks_indexes.append(lb + np.argmax(normalized_intensity[lb:rb]))
        peaks_indexes = renewed_peaks_indexes

    if peaks_positions_plot:
        plt.subplots(figsize=(12, 5))
        plt.plot(theta, normalized_intensity)
        plt.scatter(theta[peaks_indexes], normalized_intensity[peaks_indexes])
        plt.xlabel(r'$2\Theta$')
        plt.ylabel(r'$I/I_{max}$')
        plt.suptitle('peaks positions for %s' % file)
        plt.show()
        plt.close()

    '''FIND THE BACKGROUND POINTS'''
    if i == 0:
        if pick_background_points:
            angles_positions = []
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.set_title('choose backgrounds points', picker=True)
            ax.set_ylabel('Intensity', picker=True)
            line, = ax.plot(theta, np.log(intensity), linestyle='--', color='darkgreen', alpha=0.7, picker=5)

            fig.canvas.mpl_connect('pick_event', onpick1)
            plt.show()
            plt.clf()
            plt.close()

            background_ind = [find_closest(theta, position) for position in angles_positions]
            print(*background_ind, sep=', ')

            popt, pcov = curve_fit(polynome, theta[background_ind], normalized_intensity[background_ind])

            plt.subplots(figsize=(12, 5))
            plt.plot(x_axis, intensity, color='darkred')
            plt.plot(x_axis, polynome(theta, *popt))
            plt.scatter(x_axis[background_ind], intensity[background_ind])
            plt.xlabel('indexes')
            plt.ylabel('normalized_intensity')
            plt.show()
            plt.clf()
            plt.close()

        elif automatic_background_points:
            background_intensity_criteria = \
            Counter(normalized_intensity.round(4)).most_common(intensity_low_most_common)[-1][0]
            background_ind = np.where(np.array(normalized_intensity) < background_intensity_criteria)[0]

        elif automatic_background_points and intensity_low_boundary != None:
            background_intensity_criteria = intensity_low_boundary

        else:
            background_ind = found_background_points

        background_popt, background_pcov = curve_fit(polynome, theta[background_ind],
                                                     normalized_intensity[background_ind])

    else:
        background_intensity_criteria = \
        Counter(normalized_intensity.round(4)).most_common(intensity_low_most_common)[-1][0]
        background_ind = np.where(np.array(normalized_intensity) < background_intensity_criteria)[0]

        background_popt, background_pcov = curve_fit(polynome, theta[background_ind],
                                                     normalized_intensity[background_ind], p0=background_popt)
    if background_plot:
        plt.subplots(figsize=(12, 5))
        plt.plot(theta, normalized_intensity, color='seagreen')
        plt.plot(theta, polynome(theta, *background_popt), color='coral')
        plt.scatter(theta[background_ind], normalized_intensity[background_ind], marker='.', color='darkred')
        plt.xlabel(r'$2\Theta$')
        plt.ylabel(r'$I/I_{max}$')
        plt.suptitle('background of %s' % file)
        plt.show()
        plt.clf()
        plt.close()

    '''GUESS AND BOUNDARIES FOR FITTING'''
    if i == 0:
        p0 = []
        left_boundary = []
        right_boundary = []
        for ind in peaks_indexes:
            p0.append([normalized_intensity[ind], theta[ind], 0.25, 0.05, 0.05])
            left_boundary.append(
                [normalized_intensity[ind] * (1 - delta_intensity), theta[ind] - delta_angle, 0, 0, -1])
            right_boundary.append(
                [normalized_intensity[ind] * (1 + delta_intensity), theta[ind] + delta_angle, 1, 1, 1])

        p0.append(list(background_popt))
        p0 = sum(p0, [])

    else:
        p0 = fit_popt
        left_boundary = []
        right_boundary = []
        for ind in peaks_indexes:
            left_boundary.append(
                [normalized_intensity[ind] * (1 - delta_intensity), theta[ind] - delta_angle, 0, 0, -1])
            right_boundary.append(
                [normalized_intensity[ind] * (1 + delta_intensity), theta[ind] + delta_angle, 1, 1, 1])

    left_boundary.append([val - (np.abs(val) * 1e6) for val in background_popt])
    right_boundary.append([val + (np.abs(val) * 1e6) for val in background_popt])
    left_boundary = sum(left_boundary, [])
    right_boundary = sum(right_boundary, [])

    fit_popt, fit_pcov = curve_fit(pseudo_voigt_multiprofile_asymmetry_coeff_with_background, theta,
                                   normalized_intensity,
                                   p0=p0,
                                   bounds=(left_boundary, right_boundary),
                                   sigma=np.full_like(theta, 1e-6),
                                   maxfev=1e6)

    fit_function = pseudo_voigt_multiprofile_asymmetry_coeff_with_background(theta, *fit_popt)
    fit_error = (normalized_intensity - fit_function)

    ss_res = np.sum(
        (normalized_intensity - pseudo_voigt_multiprofile_asymmetry_coeff_with_background(theta, *fit_popt)) ** 2)
    ss_tot = np.sum((normalized_intensity - np.mean(normalized_intensity)) ** 2)
    r2 = 1 - ss_res / ss_tot

    background_popt = fit_popt[-(power_of_backround_polynome + 1):]
    peaks_popt = np.round(fit_popt[:-(power_of_backround_polynome + 1)], 7)
    peaks_perr = np.round(np.sqrt(np.diag(fit_pcov[:-(power_of_backround_polynome + 1)])), 7)

    peaks_parameters = list(np.reshape(peaks_popt, (len(peaks_popt) // 5, 5)))
    peaks_standart_deviation = list(np.reshape(peaks_perr, (len(peaks_perr) // 5, 5)))

    if len(remove_peaks) != 0:
        for ind in remove_peaks[::-1]:
            peaks_parameters.pop(ind)
            peaks_standart_deviation.pop(ind)

    peaks_parameters = np.array(peaks_parameters)
    peaks_standart_deviation = np.array(peaks_standart_deviation)

    if i == 0:
        phase_hkl = []
        for hkl_i in range(len(peaks_parameters) + len(remove_hkl)):
            hkl = ''.join([str(elem) for elem in list(hkl_phase.loc[hkl_i])])
            phase_hkl.append(hkl[1:])

        if len(remove_hkl) != 0:
            for hkl in remove_hkl:
                phase_hkl.remove(hkl)

    peaks_parameters = np.vstack((phase_hkl, peaks_parameters.T))
    peaks_parameters = np.vstack((peaks_parameters, peaks_standart_deviation.T))

    pattern_peaks_parameters = {}
    parameters_names = ['hkl',
                        'intensity', 'theta', 'nu', 'fwhm', 'A',
                        'intensity_std', 'theta_std', 'nu_std', 'fwhm_std', 'A_std', ]
    for j, param in enumerate(parameters_names[:]):
        pattern_peaks_parameters[param] = peaks_parameters[j]
        peaks_data = pd.DataFrame(pattern_peaks_parameters)
        peaks_data['hkl'] = peaks_data['hkl'].astype(str)
        if savefile:
            peaks_data.to_csv(os.path.join(savedir, filename + '.csv'),
                              index=False)

    print('\n%s\t\tR2 = %.6f' % (file, r2))
    print(peaks_data.iloc[:, :6])

    if oneframe_fit_plot or flow_fit_plot:
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        ax1.plot(theta, normalized_intensity, linestyle='--', color='seagreen')
        ax1.plot(theta, fit_function, color='darkred', alpha=0.9)
        ax2.plot(theta, fit_error, color='darkblue', alpha=0.9)

        plt.suptitle('\n%s      R2=%.6f' % (file, r2))

        if oneframe_fit_plot:
            plt.show()
            plt.clf()
            plt.close()

        elif flow_fit_plot:
            plt.pause(0.1)
            plt.clf()
            plt.close()
