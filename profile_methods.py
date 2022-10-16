import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from library.functions import *


def mwh_surface_function(xy, A, B, C):
    return A * (xy[:, 1] ** 2 - B * xy[:, 0] ** 2 * xy[:, 1] ** 2) + C


def mwh_surface_function_stacking(xy, A, B, C, D):
    return A * (xy[:, 1] ** 2 - B * xy[:, 0] ** 2 * xy[:, 1] ** 2) + C + D * xy[:, 2]


class mwh():
    def __init__(self, theta, fwhm, hkl, wavelength, ch00=0.5):
        self.K = 2 * np.sin(np.deg2rad(theta / 2)) / wavelength
        self.dK = np.deg2rad(fwhm) * np.cos(np.deg2rad(theta / 2)) / wavelength
        self.H = np.array([np.sqrt((int(str(plane_index)[0]) ** 2 * int(str(plane_index)[1]) ** 2 +
                                    int(str(plane_index)[0]) ** 2 * int(str(plane_index)[2]) ** 2 +
                                    int(str(plane_index)[1]) ** 2 * int(str(plane_index)[2]) ** 2) /
                                   (int(str(plane_index)[0]) ** 2 + int(str(plane_index)[1]) ** 2 +
                                    int(str(plane_index)[2]) ** 2) ** 2) for plane_index in hkl])
        self.ch00 = ch00

    def input_data(self):
        dat = np.column_stack((self.H, self.K, self.dK ** 2))
        return dat

    def surface_fitting(self):
        xy = self.input_data()

        #p0 = (1, 2, 0.1)  # beta q alpha**2
        #bounds = ((-np.inf, -10, 1e-06), (np.inf, 10, 0.81))
        params, pcov = curve_fit(mwh_surface_function, xy[:, :2], xy[:, 2],
                                 #p0=p0, bounds=bounds,
                                 maxfev=10000)
        q = params[1]
        d = 0.9 / np.sqrt(np.abs(params[2]))
        return q, d

    def chkl(self):
        q = self.surface_fitting()[0]
        chkl = self.ch00 * (1 - q * self.H ** 2)
        return chkl

    def line_fitting(self):
        x = self.K ** 2 * self.chkl()
        y = self.dK ** 2
        popt, pcov = curve_fit(polynome_1, x, y)
        ss_res = np.sum((y - polynome_1(x, *popt))**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / ss_tot
        return popt, r2

    def plotting(self, *plot_parameters):
        labelname, color, shape, textsize, markersize, linewidth = plot_parameters
        popt, error = self.line_fitting()
        x = self.K ** 2 * self.chkl()
        y = self.dK ** 2
        plt.scatter(x, y*1e2, label=r'%s' % labelname, edgecolor=color, color=color, alpha=0.8, marker=shape, s=markersize)
        plt.plot(x, polynome_1(x, *popt)*1e2, alpha=0.9, color=color, linewidth=linewidth)
        plt.ylabel(r'$\Delta K^2 \cdot 10^{-2}$ (1/nm$^2$)', size=textsize)
        plt.xlabel(r'$K^2 \bar{C}_{hkl}$ (1/nm$^2$)', size=textsize)
        plt.xticks(size=textsize)
        plt.yticks(size=textsize)

    def output_data(self):
        q, d = self.surface_fitting()
        _, r2 = self.line_fitting()
        return q, d, r2






class mwh_stackings():
    def __init__(self, theta, fwhm, hkl, wg_coeff, lattice_parameter, wavelength, ch00=0.5):
        self.K = 2 * np.sin(np.deg2rad(theta / 2)) / wavelength
        self.dK = np.deg2rad(fwhm) * np.cos(np.deg2rad(theta / 2)) / wavelength
        self.H = np.array([np.sqrt((int(str(plane_index)[0]) ** 2 * int(str(plane_index)[1]) ** 2 +
                                    int(str(plane_index)[0]) ** 2 * int(str(plane_index)[2]) ** 2 +
                                    int(str(plane_index)[1]) ** 2 * int(str(plane_index)[2]) ** 2) /
                                   (int(str(plane_index)[0]) ** 2 + int(str(plane_index)[1]) ** 2 +
                                    int(str(plane_index)[2]) ** 2) ** 2) for plane_index in hkl])
        self.Wg = wg_coeff / lattice_parameter
        self.ch00 = ch00

    def input_data(self):
        dat = np.column_stack((self.H, self.K, self.Wg, self.dK ** 2))
        return dat

    def surface_fitting(self):
        xy = self.input_data()

        #p0 = (1, 2, 0.1, 1e-04)  # beta q alpha**2 stackings_probability
        #bounds = ((-np.inf, -10, 1e-06, 0), (np.inf, 10, 0.81, 1))
        params, pcov = curve_fit(mwh_surface_function_stacking, xy[:, :3], xy[:, 3],
                                 #p0=p0, bounds=bounds,
                                 maxfev=10000)
        ss_res = np.sum((xy[:, 3] - mwh_surface_function_stacking(xy[:, :3], *params)) ** 2)
        ss_tot = np.sum((xy[:, 3] - np.mean(xy[:, :3])) ** 2)

        r2 = 1 - ss_res / ss_tot
        q = params[1]
        beta = params[3]
        d = 0.9 / np.sqrt(np.abs(params[2]))
        return q, beta, d, r2

    def chkl(self):
        q = self.surface_fitting()[0]
        chkl = self.ch00 * (1 - q * self.H ** 2)
        return chkl

    def line_fitting(self):
        x = self.K ** 2 * self.chkl()
        y = self.dK ** 2
        popt, pcov = curve_fit(polynome_1, x, y)
        ss_res = np.sum((y - polynome_1(x, *popt)) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        return popt, r2

    def plotting(self, *plot_parameters):
        labelname, color, shape, textsize, markersize, linewidth = plot_parameters
        popt, error = self.line_fitting()

        x = self.K ** 2 * (self.chkl())
        y = self.dK ** 2

        plt.scatter(x, y * 1e2, label=r'%s' % labelname, edgecolor=color, color=color, alpha=0.8, marker=shape,
                    s=markersize)
        plt.plot(x, polynome_1(x, *popt) * 1e2, alpha=0.9, color=color, linewidth=linewidth)
        plt.ylabel(r'$\Delta K^2 \cdot 10^{-2}$, 1/nm$^2$', size=textsize)
        plt.xlabel(r'$K^2 \bar{C}_{hkl}$, 1/nm$^2$', size=textsize)
        plt.xticks(size=textsize)
        plt.yticks(size=textsize)

    def output_data(self):
        q, beta, d, _ = self.surface_fitting()
        _, r2 = self.line_fitting()
        return q, beta, d, r2
