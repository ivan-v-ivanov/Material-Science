import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack
import os
import pandas as pd
from library.methods import make_filelist, make_directorylist, find_closest

workdir = '/Users/ivanivanov/Yandex.Disk.localized/DESY_2021/HEAs/heating/Al03CoCrFeNi80'
loaddir = os.path.join(workdir, 'peaks')
savedir = os.path.join(workdir, 'peaks_fft_simulated_deconvolution')
chklfile = os.path.join(workdir, 'results/chkl.csv')
logfile = os.path.join(workdir, 'logfile.csv')
temperature = pd.read_csv(logfile)['temperature1']
temperature[temperature.isna()] = 25

directorylist = make_directorylist(loaddir)

fft_plot = False
should_save = True
should_deconvolute = True
simulated_patterns_deconvolution = True

if simulated_patterns_deconvolution:
    instrumental_workdir = '/Users/ivanivanov/Yandex.Disk.localized/DESY_2021/HEAs/heating/simulated_undeformed'
    instrumental_logfile = pd.read_csv('/Users/ivanivanov/Yandex.Disk.localized/DESY_2021/HEAs/heating/Al03CoCrFeNi00/logfile.csv')
    instrumental_temperature = instrumental_logfile['temperature1']


wavelength = 1.4234695572124026e-2  # nm
chkl = pd.read_csv(chklfile)

for i, directory in enumerate(directorylist):
    print(directory)
    peaks_directory = os.path.join(loaddir, directory)
    filelist = make_filelist(peaks_directory, 'csv')


    for j, file in enumerate(filelist):
        print(file)
        fft_savedir = os.path.join(savedir, directory)

        if not os.path.exists(fft_savedir):
            os.makedirs(fft_savedir)

        peak_hkl = file[file.find('_') + 1:file.rfind('.')]
        peak_chkl = chkl[peak_hkl].iloc[[i]]

        loadfile = os.path.join(peaks_directory, file)
        df = pd.read_csv(loadfile)
        theta = np.deg2rad(df['theta'] / 2)
        intensity = np.array(df['intensity'])

        theta0 = theta[np.argmax(intensity)]

        K = 2 * np.sin(theta) / wavelength
        K0 = 2 * np.sin(theta0) / wavelength
        a3 = 1 / (np.max(K) - np.min(K))

        K2Chkl = K0 ** 2 * peak_chkl

        fy = scipy.fftpack.fft(intensity)
        n_array = np.linspace(0, len(fy) - 1, len(fy))
        fx = n_array * a3
        fy_deconvoluted = np.full((len(fx),), np.nan)

        if should_deconvolute and simulated_patterns_deconvolution:

            instrumental_temperature_index = find_closest(instrumental_temperature, temperature[i])
            instrumental_filenumber = instrumental_logfile['filenumber'].loc[instrumental_temperature_index]

            instrumental_directory = os.path.join(instrumental_workdir, 'peaks', '%.5d'%instrumental_filenumber)
            instrumental_peaks = make_filelist(instrumental_directory, 'csv')

            instrumental_file = os.path.join(instrumental_directory, instrumental_peaks[j])

            instrumental_fy = scipy.fftpack.fft(np.array(pd.read_csv(instrumental_file)['intensity']))

            if len(fy_deconvoluted)<len(instrumental_fy):
                instrumental_fy = instrumental_fy[:-1]

            elif len(fy_deconvoluted)>len(instrumental_fy):
                fy = fy[1:]
                fx = fx[1:]

            fy_deconvoluted = fy / instrumental_fy



        # if should_deconvolute and not simulated_patterns_deconvolution:
        #     instrumental_file = os.path.join(instrumental_directory, instrumental_peaks[j])
        #     instrumental_fy = pd.read_csv(instrumental_file)['FY']
        #     print(pd.read_csv(instrumental_file)['FX'])
        #     print(fx)
        #     fy_deconvoluted = fy / instrumental_fy

        if fft_plot:
            plt.plot(fx, np.log(np.abs(fy_deconvoluted)), linestyle='None', marker='.', color='navy')
            plt.title('%s %s' % (directory, peak_hkl), loc='right')
            plt.xlabel(r'$L, nm$')
            plt.ylabel(r'$lnA(L)$')
            plt.show()
            plt.close()


        if should_save:
            savefile = os.path.join(fft_savedir, '%i_%s_%.6f.csv' % (j, peak_hkl, K2Chkl))
            fft_df = pd.DataFrame(data={'FX': fx, 'FY': fy, 'FY_deconvoluted': fy_deconvoluted})
            fft_df.to_csv(savefile, index=False)
