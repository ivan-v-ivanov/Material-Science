import numpy as np
import matplotlib.pyplot as plt
import os
from library.methods import make_filelist, find_closest
import pandas as pd
from library import profile_methods

wg_coeff = np.array([np.sqrt(3) / 4, 1, np.sqrt(2) / 2, 3 / 2 / np.sqrt(11), np.sqrt(3) / 4, 1])
wavelength = 1.4234695572124026e-2  # nm
ch00 = 0.27392


should_save = True
stackings_analyze = True
deconvolution = False
simulated_patterns_deconvolution = True

if deconvolution:
    instrumental_workdir = '/Users/ivanivanov/Yandex.Disk.localized/DESY_2021/HEAs/heating/Al03CoCrFeNi00'
    instrumental_logfile = os.path.join(instrumental_workdir, 'logfile.csv')
    instrumental_temperature = pd.read_csv(instrumental_logfile)['temperature1']

    instrumental_directory = os.path.join(instrumental_workdir, 'peaks_data')
    instrumental_filelist = make_filelist(instrumental_directory, 'csv')

    instrumental_temperature = instrumental_temperature[:len(instrumental_filelist)]

elif simulated_patterns_deconvolution:
    instrumental_workdir = '/Users/ivanivanov/Yandex.Disk.localized/DESY_2021/HEAs/heating/simulated_undeformed'
    instrumental_logfile = pd.read_csv('/Users/ivanivanov/Yandex.Disk.localized/DESY_2021/HEAs/heating/Al03CoCrFeNi00/logfile.csv')
    instrumental_temperature = instrumental_logfile['temperature1']

    instrumental_directory = os.path.join(instrumental_workdir, 'peaks_data')
    instrumental_filelist = make_filelist(instrumental_directory, 'csv')


workdir = '/Users/ivanivanov/Yandex.Disk.localized/DESY_2021/HEAs/heating'
directories = ['Al03CoCrFeNi80']#, 'Al03CoCrFeNi00', 'Al03CoCrFeNi40', 'Al03CoCrFeNi60']

for directory in directories[::]:
    print(directory)

    mwh_data = {
        'temperature': [],
        'q': [],
        'beta': [],
        'crystalline_size': [],
        'r2': []
    }

    chkl_data = []

    if not stackings_analyze:
        mwh_data.pop('beta', None)

    loaddir = os.path.join(workdir, directory)
    logfile = os.path.join(loaddir, 'logfile.csv')
    temperature = pd.read_csv(logfile)['temperature1']

    temperature[temperature.isna()] = 25

    lattice_parameter = pd.read_csv(os.path.join(loaddir, 'results/lattice_parameter.csv'))['lattice_parameter']

    loaddir = os.path.join(loaddir, 'peaks_data')
    filelist = make_filelist(loaddir, 'csv')

    for i, file in enumerate(filelist[::]):
        print(file)
        loadfile = os.path.join(loaddir, file)
        data = pd.read_csv(loadfile)
        #data = data.drop([0, 5, 8, 9], axis='index')

        if deconvolution or simulated_patterns_deconvolution:

            instrumental_temperature_index = find_closest(instrumental_temperature, temperature[i])
            instrumental_filenumber = instrumental_logfile['filenumber'].loc[instrumental_temperature_index]
            instrumental_file = os.path.join(instrumental_directory,
                                             '%.5d.csv'%instrumental_filenumber)

            instrumental_data = pd.read_csv(instrumental_file)
            data['fwhm'] = data['fwhm'] - instrumental_data['fwhm']
            #data.loc[data['fwhm'] < 0, 'fwhm'] = 0

            if stackings_analyze:
                instrumental_data = instrumental_data.head(len(wg_coeff))

        if stackings_analyze:
            data = data.head(len(wg_coeff))
            MWH = profile_methods.mwh_stackings(data['theta'], data['fwhm'], data['hkl'], wg_coeff,
                                                lattice_parameter[i], wavelength, ch00=ch00)
            q, beta, d, r2 = MWH.output_data()
            mwh_data['beta'].append(beta)

        else:
            MWH = profile_methods.mwh(data['theta'], data['fwhm'], data['hkl'], wavelength, ch00=ch00)
            q, d, r2 = MWH.output_data()

        chkl = MWH.chkl()

        #MWH.plotting(file)

        mwh_data['temperature'].append(temperature[i])
        mwh_data['q'].append(q)
        mwh_data['crystalline_size'].append(d)
        mwh_data['r2'].append(r2)

        chkl_data.append(chkl)


    if should_save:
        save_dataframe = pd.DataFrame(mwh_data)
        #chkl_data = pd.DataFrame(chkl_data, columns=data['hkl'])
        #chkl_data.insert(0, 'temperature', mwh_data['temperature'])

        #chkl_data.to_csv(os.path.join(workdir, directory, 'results/chkl.csv'), index=False)
        save_dataframe.to_csv(os.path.join(workdir, directory,'results/mwh_beta.csv'), index=False)

    x_axis = mwh_data['temperature']
    # x_axis = np.linspace(0, len(filelist), len(filelist))

    if stackings_analyze:
        figures_num = 4
        plt.subplot(figures_num, 1, figures_num - 1)
        plt.scatter(x_axis, mwh_data['beta'], alpha=0.5, label=directory)
        plt.ylabel(r'$\beta$')
        plt.legend()
    else:
        figures_num = 3

    plt.subplot(figures_num, 1, 1)
    plt.scatter(x_axis, mwh_data['crystalline_size'], alpha=0.5, label=directory)
    plt.ylabel('D, nm')
    plt.ylim(bottom=0, top=200)
    plt.legend()

    plt.subplot(figures_num, 1, 2)
    plt.scatter(x_axis, mwh_data['q'], alpha=0.5, label=directory)
    plt.ylabel(r'q')
    plt.ylim(bottom=-10, top=10)
    plt.legend()

    plt.subplot(figures_num, 1, figures_num)
    plt.scatter(x_axis, mwh_data['r2'], alpha=0.5, label=directory)
    plt.ylabel(r'$R^2$')
    plt.xlabel(r'Temperature, $\degree$C')
    plt.legend()

plt.show()
plt.close()
