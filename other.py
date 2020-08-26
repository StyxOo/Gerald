# -*- coding: utf-8 -*-

"""

Created on Mon Apr 20 23:45:57 2020



@author: Phillip

"""

# Packages einlesen
import math
import os

import matplotlib.pyplot as plt  # für die ganzen Plots zuständig

import numpy as np  # für die Rechnung mit Arrays, Vektoren und Matrizen

from numpy import exp  # E-Funktion

from lmfit.models import Model  # für das Erstellen EIGENER FIT-Funktionen

import pandas as pd

# Daten auslesen

data_files = ''

asc_name = '20-08-02_spec_energy_axe_17.csv'

spec = pd.read_csv(f"{os.path.join(data_files, asc_name)}", skiprows=2, delimiter='\t', decimal='.', header=None)

x = spec[0].values

y = spec[1].values

print(x)

print(y)

# Lädt Daten aus einer Textfile (Name der File (Dateiendung angeben), Trennzeichen, Zeilen überspringen, Spalte wählen, unpack = If True, the returned array is transposed )


# Verschiebung des Abszissendatensatzes über null

y_min = float(abs(min(y)))  # Bestimmt das Minimum des Ordinatendatensatzes und gibt es als rationale Zahl aus

d = y + y_min


# Lineare Funktion definieren für exponentiellen Hintergrund (basierend auf die Interbandübergänge der Goldnanopartikel, näherungsweise)

def linear(x, grad,cut):
    """
    grad = Steigung der lineare Funktion, shift = Verschiebung der linearen Funktion auf der Abszissenachse, cut = Schnitt auf der um den shift verschobenen Ordinatenachse
    :param x: Unabhängige variable
    :param grad: Steigung
    :param cut: Y-Achsen abschnitt
    :return: linear wert
    """

    return grad * x + cut  # Funktion


# Startwerte Paramater der Exponentialfunktion für den Fit festlegen

lin_grad_guess = float((d[np.argwhere(x == max(x))] - d[np.argwhere(x == min(x))]) / (max(x) - min(x)))

lin_cut_guess = float(d[np.argwhere(x == max(x))] - lin_grad_guess * max(x))


# Gaussfunktion definieren für Plasmonenpeaks

def gauss(x, amp, cen,
          wid):  # (amp=Maximum des Gausspeaks, cen = Erwartungswert bzw. Argument des lokalen Extremums, wid = Standardabweichung, näherungsweise HFHM)

    return (amp * exp(-(x - cen) ** 2 / (2 * wid ** 2)))  # Funktion


# Grenzen für die Bestimmung des ersten Gauss-Maximums festlegen

g1_li = 2.0  # linke Grenze erstes Gauss-Maximum

g1_ri = 2.1  # rechte Grenze erstes Gauss-Maximum

# Startwerte Paramter der ersten Gaussfunktion für den Fit festlegen


def closest_value(array, vergleichswert):
    closest_dst = math.inf
    closest_index = None
    for i in range(len(x)):
        dst = array[i] - vergleichswert
        if abs(dst) < closest_dst:
            closest_dst = abs(dst)
            closest_index = i
    return closest_index


def max_value_index(interval_min, interval_max):
    index_min = closest_value(x, interval_min)
    index_max = closest_value(x, interval_max)

    max_index = index_min
    max_value = 0
    for i in range(index_min, index_max + 1):
        if d[i] > max_value:
            max_value = d[i]
            max_index = i
    return max_index


# g1_li_mid = int(np.argwhere(index_min))
# g1_ri_mid = int(np.argwhere(index_max))

g1_max_index = max_value_index(g1_li, g1_ri)

g1_cen_guess = float(x[g1_max_index])  # Energie an der Indexstelle des Maximums von d im Bereich vom Index der linken Gauss-Grenze bis zum Index der rechten Gauss-Grenze der Energie



g1_amp_guess = float(d[g1_max_index] - float(linear(x[g1_max_index], lin_grad_guess, lin_cut_guess)))  # Maximums von d im Bereich vom Index der linken Gauss-Grenze bis zum Index der rechten Gauss-Grenze der Energie, abzüglich vom Funktionswert der Exponentialfunktion an der Stelle des Energiemaximums der Gaussfunktion

g1_wid_guess = 0.05  # HWHM oder Standardabweichung

# Grenzen für die Bestimmung des zweiten Gauss-Maximums festlegen

g2_li = 2.1  # linke Grenze zweites Gauss-Maximum

g2_ri = 2.2  # rechte Grenze zweites Gauss-Maximum



g2_max_index = max_value_index(g2_li, g2_ri)



g2_cen_guess = float(x[g2_max_index])  # Energie an der Indexstelle des Maximums von d im Bereich vom Index der linken Gauss-Grenze bis zum Index der rechten Gauss-Grenze der Energie

g2_amp_guess = float(d[g2_max_index] - float(
    linear(x[g2_max_index], lin_grad_guess,
           lin_cut_guess)))  # Maximums von d im Bereich vom Index der linken Gauss-Grenze bis zum Index der rechten Gauss-Grenze der Energie, abzüglich vom Funktionswert der Exponentialfunktion an der Stelle des Energiemaximums der Gaussfunktion

g2_wid_guess = 0.06  # HWHM oder Standardabweichung

# Fitfunktion der Exponentialfunktion erstellen

linmodel = Model(linear,
                 prefix='lin_')  # erster Slot: Funktion, prefix = Benennung der Funktion, relevant für Komponentendarstellung im Diagramm

pars = linmodel.make_params()

pars['lin_grad'].set(value=lin_grad_guess)

pars['lin_cut'].set(value=lin_cut_guess)

# Fitfunktion der ersten Gaussfunktion erstellen

gauss1 = Model(gauss,
               prefix='g1_')  # erster Slot: Funktion, prefix = Benennung der Funktion, relevant für Komponentendarstellung im Diagramm

pars.update(gauss1.make_params())  # Array um die Parameter des neuen Fitmodells erweitern

pars['g1_cen'].set(
    value=g1_cen_guess)  # Startwert für den jeweiligen Paramater benennen. Name des Parameters im Array = Prefix + Parameter

pars['g1_amp'].set(
    value=g1_amp_guess)  # Startwert für den jeweiligen Paramater benennen. Name des Parameters im Array = Prefix + Parameter

pars['g1_wid'].set(
    value=g1_wid_guess)  # Startwert für den jeweiligen Paramater benennen. Name des Parameters im Array = Prefix + Parameter

# Fitfunktion der zweiten Gaussfunktion erstellen

gauss2 = Model(gauss,
               prefix='g2_')  # erster Slot: Funktion, prefix = Benennung der Funktion, relevant für Komponentendarstellung im Diagramm

pars.update(gauss2.make_params())  # Array um die Parameter des neuen Fitmodells erweitern

pars['g2_cen'].set(
    value=g2_cen_guess)  # Startwert für den jeweiligen Paramater benennen. Name des Parameters im Array = Prefix + Parameter

pars['g2_amp'].set(
    value=g2_amp_guess)  # Startwert für den jeweiligen Paramater benennen. Name des Parameters im Array = Prefix + Parameter

pars['g2_wid'].set(
    value=g2_wid_guess)  # Startwert für den jeweiligen Paramater benennen. Name des Parameters im Array = Prefix + Parameter

# Fitfunktion des Gesamtmodells. Einzelne, extra Benennung der Komponenten relevant, damit diese im Diagramm getrennt dargestellt werden können

mod = gauss1 + gauss2 + linmodel  # Summe der drei Komponenten zum Gesamtmodell. Liefert den theoretischen Verlauf der optischen Eigenschaften von Goldnanopartikeln

# Durchführung der Fitrechenprozedur

init = mod.eval(pars,
                x=x)  # Initialfunktion: mod-Funktion wird "evaluiert",d.h. sie berechnet die Start-Funktion mit den Startparametern. Verwendert dabei für die Parameter das Array pars, welches die Startparamater enthält. Als Abszissenwerte werden die x-Variablen der mod-Funktion verwendet, die wiederum den Energiewerten Enew zugeordnet werden.

out = mod.fit(d, pars,
              x=x)  # Fitfunktion: Bei gegebenem Datensatz d wird die bestmögliche mod-Funktion, die durch Start-Parameternim Array pars gegeben sind, erstellt. Als Abszissenwerte werden die x-Variablen der mod-Funktion verwendet, die wiederum den Energiewerten Enew zugeordnet werden.

dely = out.eval_uncertainty(x=x,
                            sigma=2)  # Bestimmen der Konfidenzintervalle für die Fitfunktion (nicht für die einzelnen Parameter). Sigma gibt das Signifikanzniveau an

# Erstellt eine Umgebung für die Resultate des Fits, bestehend aus dem Datensatz, Initial-Funktion, bester Fit, und deren Komponenten des besten Fits

fig, axes = plt.subplots(1, 2, figsize=(12.8,
                                        4.8))  # erstellt 1x2=2 Diagrammen in Form einer 1x2 Matrix an Diagrammen (Anzahl Zeilen, Anzahl Spalten). Die Matrix besitzen die gegebene Gesamtbreite und -höhe

# Erstes Diagramm (Position 0)

axes[0].plot(x, d, 'b',
             label='Daten')  # Plot im nullten Diagramm, Abszissendatensatz = Enew, stellt den zu fittenden Datensatz grafisch dar, Farbe = blau, betitelt mit "Daten"

axes[0].plot(x, init, 'k--',
             label='Startfit')  # Plot im nullten Diagramm, Abszissendatensatz = Enew, stellt die Funktion mit Startparametern grafisch dar, Farbe = schwarz, gestrichelt, betitelt mit "Startfit"

axes[0].plot(x, out.best_fit, 'r-',
             label='Bester Fit')  # Plot im nullten Diagramm, Abszissendatensatz = Enew, stellt die beste Fit-Funktion mit den gegebenen Startparametern durch den Ausdruck out.best_fit grafisch dar, Farbe = rot, betitelt mit "bester Fit"

axes[0].fill_between(x, out.best_fit - dely, out.best_fit + dely,
                     color='#888888')  # Füllt die Fläche zwischen zwei Funktionen mit einer Farbe deiner Wahl aus. Die zwei Funktionen stellen in diesem Fall die Konfidenzintervallsgrenzen für die Best-Fit-Funktion dar.

axes[0].legend(
    loc='best')  # erstellt eine Legende mit den betitelten Funktionen, positioniert sich automatisch da, wo die Plots am wenigsten durch die Legende überdeckt werden

# Zweites Diagramm (Position 1)

comps = out.eval_components(
    x=x)  # gibt die Komponenten der erstellten Fitfunkten ("out") in einem Array an. Array besteht aus exp_ ,g1_ ,g2_ mit Abszissendatensatz Enew

axes[1].plot(x, d,
             'b')  # Plot im ersten Diagramm, Abszissendatensatz = Enew, stellt den zu fittenden Datensatz grafisch dar, Farbe = blau, betitelt mit "Daten"

axes[1].plot(x, comps['lin_'], 'g--', label='Lineare Komponente')

axes[1].plot(x, comps['g1_'], 'm--',
             label='Gauß-Komponente 1')  # Plot im ersten Diagramm, Abszissendatensatz = Enew, stellt erste Gausskomponente grafisch dar, Farbe = violett, gestrichelt, betitelt mit "Gauß-Komponente 1"

axes[1].plot(x, comps['g2_'], 'k--',
             label='Gauß-Komponente 2')  # Plot im ersten Diagramm, Abszissendatensatz = Enew, stellt zweite Gausskomponente grafisch dar, Farbe = schwarz, gestrichelt, betitelt mit "Gauß-Komponente 2"

axes[1].legend(
    loc='best')  # erstellt eine Legende mit den betitelten Funktionen, positioniert sich automatisch da, wo die Plots am wenigsten durch die Legende überdeckt werden

plt.setp(axes[:], xlabel='Energie (eV)')  # betitelt alle Abszissenachsen einzeln

plt.setp(axes[:], ylabel='Transmission/ Extinktion (rel. Einheit)')  # betitelt alle Ordinatenachsen einzeln

plt.show()  # zeigt den Plot

plt.close  # schließt die Figure-Umgebung zum Einsparen von Computerspeicher

# Statistik

print(out.fit_report(
    min_correl=0.5))  # Statistik über die Fitfunktion und deren Initial- und Finalparametern sowie dessen Konfidenzintervalle

outfile = open(f"./{os.path.join(data_files, asc_name)[:-4]}_statistic.txt", "w+")
outfile.writelines(out.fit_report(min_correl=0.5))
