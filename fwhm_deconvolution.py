import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from methods import make_filelist

instrumental_directory = '/Users/ivanivanov/Yandex.Disk.localized/DESY_2021/HEAs/heating/Al03CoCrFeNi00'
instrumental_logfile = os.path.join(instrumental_directory, 'logfile.csv')
instrumental_filelist = make_filelist(os.path.join(instrumental_directory, 'peaks_data'), 'csv')
instrumental_temperature = pd.read_csv(instrumental_logfile)['temperature1']
instrumental_temperature = instrumental_temperature[:353]

for temp, file in zip(instrumental_temperature, instrumental_filelist):
    print(temp, file)

workdir = '/Users/ivanivanov/Yandex.Disk.localized/DESY_2021/HEAs/heating'
directories = ['Al03CoCrFeNi40', 'Al03CoCrFeNi60', 'Al03CoCrFeNi80']

x_axis = np.linspace(25, 995, len(instrumental_temperature))
plt.scatter(x_axis, instrumental_temperature, label = 'Al03CoCrFeNi00')

for directory in directories:
    loaddir = os.path.join(workdir, directory)
    logfile = os.path.join(loaddir, 'logfile.csv')
    temperature = pd.read_csv(logfile)['temperature1']

    loaddir = os.path.join(loaddir, 'peaks_data')
    filelist = make_filelist(loaddir, 'csv')

    temperature = temperature[:len(filelist)]

    x_axis = np.linspace(25, 995, len(temperature))
    plt.scatter(x_axis, temperature, label=directory)

plt.legend()
plt.show()



