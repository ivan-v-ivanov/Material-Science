import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit
import pandas as pd
from library.methods import make_filelist


def quadr(x, a, b, c):
    return a + b * x# + c*x**2


def slope_line(x, a, b):
    return -b * (a - x)


# def line (x, a, b):
#     return a - b*x
#
# def exp (m, s):
#     return m*np.exp(2.5*s**2)*2/3


wl = 1.4234695572124026e-2  # nm

mwa_plot = False
slope_plot = True
should_save = True

workdir = '/Users/ivanivanov/Yandex.Disk.localized/DESY_2021/HEAs/heating/Al03CoCrFeNi80'

loaddir = os.path.join(workdir, 'peaks_fft_simulated_deconvolution')
savefile = os.path.join(workdir, 'results/mwa_dislocations.csv')

lattice_file = os.path.join(workdir, 'results/lattice_parameter.csv')
lattice_parameter = np.array(pd.read_csv(lattice_file)['lattice_parameter'])

logfile = os.path.join(workdir, 'logfile.csv')
temperature = pd.read_csv(logfile)['temperature1']

directorylist = [path[0][path[0].rfind('/') + 1:] for path in os.walk(loaddir)]
directorylist.sort()
directorylist = directorylist[:-1]

savedataframe = {'temperature':[],
                 'po':[],
                 'Re':[],
                 'M':[]}
# po_list = []
# Re_list = []
# Re_star_list = []
# M_list = []
# M_star_list = []

text = 14
mark = ['o', 'v', 'P', 's', '^', 'h', 'X']
approx = ['black', 'orangered', 'mediumblue', 'crimson', 'green', 'maroon', 'rebeccapurple']
style = ['--', '-.', ':']
mark = mark * 50
approx = approx * 50
style = style * 50

number_of_mwa_lines = 35
points_to_approximate = 4
points_to_cut = 2

for m, directory in enumerate(directorylist[:]):
    reversedir = os.path.join(loaddir, directory)

    filelist = make_filelist(reversedir, 'csv')

    allLlist = []
    alist = []
    blist = []

    fig, ax1 = plt.subplots(figsize=(10, 4))
    for i in range(number_of_mwa_lines):

        #ALlist = []
        #hkdotl = []

        Xlist = []
        Llist = []
        lnALlist = []

        for j, file in enumerate(filelist):
            data = pd.read_csv(os.path.join(reversedir, file))


            fx = np.array(data['FX'])
            fy = np.abs(np.array(data['FY_deconvoluted'], dtype=np.cdouble).real)

            if j == 0:
               allLlist.append(fx[i])

            L = fx[i]
            AL0 = fy[0]
            AL = fy[i]
            K = float(file[file.find('_') + 1:file.rfind('.')])
            X = K * (np.sqrt(2) / 2 * lattice_parameter[m]) ** 2

            Llist.append(L)
            Xlist.append(X)
            lnAL = np.log(AL / AL0)
            lnALlist.append(lnAL)

        Xlist = np.array(Xlist)
        lnALlist = np.array(lnALlist)

        popt, pcov = curve_fit(quadr, Xlist, lnALlist, bounds=(-np.inf, [np.inf, np.inf, 0]))
        alist.append(popt[0])
        blist.append(popt[1])

        if mwa_plot:
            plt.scatter(Xlist, lnALlist, marker='%s' % mark[i], edgecolor='indianred', color='white',
                        label='L = %.2f' % L)
            Xlist = np.linspace(np.min(Xlist) - 0.25, np.max(Xlist) + 0.25, 50)
            plt.plot(Xlist, quadr(Xlist, *popt), linestyle='%s' % style[i], color='%s' % approx[i], alpha=0.8)
            plt.ylabel(r'$lnA(L)$', size=text)
            plt.xlabel(r'$\overline{g^2 C_{hkl} b^2}$', size=text)
            # plt.xlim(0, 0.1)
            # plt.ylim(-5, 1)
            plt.yticks(size=text * 0.9)
            plt.xticks(size=text * 0.9)
            # plt.title('%s' % object[num], loc='right')
            plt.legend(edgecolor='white', fontsize=text * 0.9)

    # plt.savefig(savelnAL, dpi=150)
    # plt.savefig(savelnALpdf, dpi=150)
    fig.tight_layout()
    #plt.show()
    plt.close()

    #cut = [4] * len(directorylist)  # [4, 4, 1, 1, 2, 1]
    allLlist = np.array(allLlist[points_to_cut:])
    alist = np.array(alist[points_to_cut:])
    blist = np.array(blist[points_to_cut:])

    #lenght = [11] * len(directorylist)  # [6, 7, 8, 8, 8, 7]##[7, 10, 8, 7, 8, 7, 6, 6, 7, 5]
    Ylist = []
    for i in range(len(allLlist)):
        Y = blist[i] / (allLlist[i]) ** 2
        Ylist.append(Y)

    Y_list = Ylist[:points_to_approximate]
    allL_list = allLlist[:points_to_approximate]
    b_list = blist[:points_to_approximate]

    ln_L = np.log(allL_list)
    lnL = np.log(allLlist[:])

    Ylist = np.array(Ylist)

    popt, pcov = curve_fit(slope_line, ln_L, Y_list, maxfev=10000)
    perr = np.sqrt(np.diag(pcov)) / len(ln_L)

    Re = np.exp(popt[0])
    Re_star = Re / np.exp(2)

    po = np.abs(popt[1]) * 2 / np.pi * 10 ** 14
    po_star = (popt[1]) * 2 / np.pi

    M = Re * np.sqrt(po_star)
    M_star = Re_star * (po_star) ** 0.5

    savedataframe['temperature'].append(temperature[m])
    savedataframe['po'].append(po)
    savedataframe['Re'].append(Re_star)
    savedataframe['M'].append(M_star)
    #M_star_list.append(M_star)

    print('\n%s' % directory)
    #print('\tT = %i deg C'%temperature[m])
    print('\tRe* = %.2f nm \n\tpo = %.2e sm-2 \n\tM* = %.3f' % (Re_star, po, M_star))
    print('\tRe  = %.2f nm \n\tpo = %.2e sm-2 \n\tM  = %.3f' % (Re, po, M))

    if slope_plot:
        # text=14
        # fig, ax = plt.subplots()#figsize=(8,6))
        plt.scatter(lnL, Ylist * 10 ** 3, marker='o', edgecolor='darkred', color='white', alpha=0.7)
        plt.plot(ln_L, slope_line(ln_L, *popt) * 10 ** 3, linestyle='--', color='midnightblue', alpha=0.0,
                 label='Re = %.2f nm\npo = %.2e sm-2\nM = %.3f' % (Re, po, M))
        plt.plot(ln_L, slope_line(ln_L, *popt) * 10 ** 3, linestyle='--', color='midnightblue', alpha=0.7)

        plt.legend(edgecolor='white', loc=4)
        # plt.title(directory)

        plt.ylabel(r'$C/L^2 \cdot 10^3$')  # ,size=text)
        plt.xlabel(r'$lnL$')  # ,size=text)
        plt.title(directory)
        # plt.yticks(size=text * 0.9)
        # plt.xticks(size=text * 0.9)

        # fig.tight_layout()
        plt.pause(0.02)
        #plt.show()
        plt.close()

if should_save:
    savedataframe = pd.DataFrame(data=savedataframe)
    savedataframe.to_csv(savefile, index=False)

    # savedata = x_axis + po_list + Re_star_list + M_star_list
    # savedata = np.reshape(savedata, (4, len(po_list)))
    #
    # np.savetxt(savefile, savedata.T, fmt='%.5f',
    #            header=f'Dislocations data according to mWA analysis\n'
    #                   f'eps (%)\tpo (sm-2)\tRe (nm)\tM\n')

# x_axis = np.linspace(0, len(po_list), len(po_list))



plt.subplot(311)
plt.scatter(savedataframe['temperature'], savedataframe['po'], color='white', edgecolor='darkred',
            marker='s', s=14)
plt.ylabel(r'$Density\/[sm^{-2}]$')

plt.subplot(312)
plt.scatter(savedataframe['temperature'], savedataframe['Re'], color='white', edgecolor='darkgreen',
            marker='^', s=14)
plt.ylabel(r'$Radius\/[nm]$')

plt.subplot(313)
plt.scatter(savedataframe['temperature'], savedataframe['M'], color='white', edgecolor='darkblue',
            marker='v', s=14)
plt.ylabel(r'$Wilkens\/constant$')

plt.show()
plt.close()
