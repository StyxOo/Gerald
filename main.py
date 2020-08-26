# -*- coding: utf-8 -*-

"""

Created on Mon Apr 20 23:45:57 2020



@author: Phillip

"""

# Packages einlesen

import matplotlib.pyplot as plt  # für die ganzen Plots zuständig

import numpy as np  # für die Rechnung mit Arrays, Vektoren und Matrizen

from numpy import exp  # E-Funktion

from lmfit.models import Model  # für das Erstellen EIGENER FIT-Funktionen

import scipy.constants as const  # für das Einbinden von Naturkonstanten

import \
    scipy.interpolate as intr  # für die Interpolation von Datensätze zur Bestimmung von Funktionswerte, die zwischen Datenpunkten sich befinden.

# Daten auslesen

q_1 = 't.csv'  # Variable für:Welche Datei soll geladen werden (inkl. Dateiendung)?

q_2 = 2  # Variable für: Wie viele Zeilen hat der Tabellenkopf?

q_2_int = int(q_2)  # int = ganze Zahl

q_3 = 11  # Variable für: Welche Daten sollen betrachtet werden (Tabellenspalte)?

q_3_int = int(q_3)  # int = ganze Zahl

x, y = np.loadtxt(q_1, delimiter=',', skiprows=q_2_int, usecols=(0, q_3_int), unpack=True)

# Lädt Daten aus einer Textfile (Name der File (Dateiendung angeben), Trennzeichen, Zeilen überspringen, Spalte wählen, unpack = If True, the returned array is transposed )


# Eingelesene Daten plotten

fig = plt.figure(1,
                 figsize=(8, 4))  # Erstellt eine Umgebung für (mehrere) Plots (Nummerierung im Speicher, Breite & Höhe)

plt.plot(x, y)  # erstellt ein Diagramm mit den ausgewählten Datensätzen (Abszisse, Ordinate)

plt.xticks(np.arange(min(x), max(x) + 25,
                     step=25))  # setzt feste Argumente auf die Abszissenachse (np.arange = Array erstellen mit Start- und Endwert, Endwert nicht angezeigt, Schrittweite)

plt.xlabel('Energie (eV)')  # Beschriftung Abszissen-Achse

plt.ylabel('Transmission/ Extinktion (rel. Einheit)')  # Beschriftung Ordinaten-Achse

plt.grid(True)  # erstellt ein Gitter

plt.show()  # zeigt den Plot

plt.close  # schließt die Figure-Umgebung zum Einsparen von Computerspeicher

# Umwandlung des Abszissendatensatz von Wellenlänge zu Energien und Interpolation des Ordinatendatensatzes auf Basis der Energien für GLEICHMÄßIGE Abstände der Abszissenwerte für spätere Fitfunktionen

E = const.h * const.c / (x * 10 ** (
    -9) * const.elementary_charge)  # Zusammenhang von Wellenlänge (Einheit: Nanometer) und Energie (Einheit: Elektronenvolt). Der Hyperbelansatz bewirkt, dass äquidistante Werte nicht äquidistant werden

f = intr.interp1d(E,
                  y)  # interpoliert den Ordinatendatensatz unter gegebenem Abszissendatensatz mit dem Zweck, Daten zwischen Datenpunkten durch Interpolationsfunktion "vorherzusagen". Dient dem Zweck, dass später äquidistante Abszissenwerte für die Fitfunktion bereitstehen.

Enew = np.linspace(1.55, 3.09,
                   617)  # Array, welche 617 äquidistante Werte von Start- bis Endwert (standardmäßig Endwert mit eingeschlossen) angibt. Die "schiefe" Anzahl bezieht sich auf die späteren Eingaben der Grenzen für die Bestimmung der Maxima für die Gaussfits, damit man bspw. 1.95 als Datenpunkt überhaupt guessen kann. np.arange ist nicht passend, da rationale Schrittweiten Probleme verursachen

ynew = f(Enew)  # Neuer Ordinatendatensatz, basierend auf den neuen Abszissendatensatz

# Invertierung der Daten

q_i = 0  # Variable für: Sollen die Daten invertiert werden (Transmission/ Extinktion)(ja=0, nein=1)?

q_i_int = int(q_i)  # ganzzahlige Ausgabe des Eingabewertes

c = (1 - q_i_int) * (
            100 - ynew) + q_i_int * ynew  # Invertierung des Ordinatendatensatzes, falls dafür entschieden wurde

# Verschiebung des Abszissendatensatzes über null

q_j = 0  # Variable für: Sollen die y-Werte größer mindestens null sein (ja=0, nein=1)? Der Graph wird um das Minimum nach oben verschoben

y_min = float(abs(min(c)))  # Bestimmt das Minimum des Ordinatendatensatzes und gibt es als rationale Zahl aus

d = c + y_min * (
    float(1 - int(q_j)))  # ggf. Verschiebung des Ordinatendatensatzes um das Minimum, falls dafür entschieden wurde


# Lineare Funktion definieren für exponentiellen Hintergrund (basierend auf die Interbandübergänge der Goldnanopartikel, näherungsweise)

def linear(x, grad,
           cut):  # (grad = Steigung der lineare Funktion, shift = Verschiebung der linearen Funktion auf der Abszissenachse, cut = Schnitt auf der um den shift verschobenen Ordinatenachse)

    return grad * x + cut  # Funktion


# Startwerte Paramater der Exponentialfunktion für den Fit festlegen

lin_grad_guess = float(
    (d[np.argwhere(Enew == max(Enew))] - d[np.argwhere(Enew == min(Enew))]) / (max(Enew) - min(Enew)))

lin_cut_guess = float(d[np.argwhere(Enew == max(Enew))] - lin_grad_guess * max(Enew))


# Gaussfunktion definieren für Plasmonenpeaks

def gauss(x, amp, cen,
          wid):  # (amp=Maximum des Gausspeaks, cen = Erwartungswert bzw. Argument des lokalen Extremums, wid = Standardabweichung, näherungsweise HFHM)

    return (amp * exp(-(x - cen) ** 2 / (2 * wid ** 2)))  # Funktion


# Grenzen für die Bestimmung des ersten Gauss-Maximums festlegen

g1_li = 1.95  # linke Grenze erstes Gauss-Maximum

g1_ri = 2.05  # rechte Grenze erstes Gauss-Maximum

# Startwerte Paramter der ersten Gaussfunktion für den Fit festlegen

g1_cen_guess = float(Enew[np.argwhere(d == max(d[int(np.argwhere(Enew == g1_li)):int(np.argwhere(
    Enew == g1_ri))]))])  # Energie an der Indexstelle des Maximums von d im Bereich vom Index der linken Gauss-Grenze bis zum Index der rechten Gauss-Grenze der Energie

g1_amp_guess = float(max(d[int(np.argwhere(Enew == g1_li)):int(np.argwhere(Enew == g1_ri))]) - float(
    linear(Enew[np.argwhere(d == max(d[int(np.argwhere(Enew == g1_li)):int(np.argwhere(Enew == g1_ri))]))],
           lin_grad_guess,
           lin_cut_guess)))  # Maximums von d im Bereich vom Index der linken Gauss-Grenze bis zum Index der rechten Gauss-Grenze der Energie, abzüglich vom Funktionswert der Exponentialfunktion an der Stelle des Energiemaximums der Gaussfunktion

g1_wid_guess = 0.1  # HWHM oder Standardabweichung

# Grenzen für die Bestimmung des zweiten Gauss-Maximums festlegen

g2_li = 2.3  # linke Grenze zweites Gauss-Maximum

g2_ri = 2.4  # rechte Grenze zweites Gauss-Maximum

g2_cen_guess = float(Enew[np.argwhere(d == max(d[int(np.argwhere(Enew == g2_li)):int(np.argwhere(
    Enew == g2_ri))]))])  # Energie an der Indexstelle des Maximums von d im Bereich vom Index der linken Gauss-Grenze bis zum Index der rechten Gauss-Grenze der Energie

g2_amp_guess = float(max(d[int(np.argwhere(Enew == g2_li)):int(np.argwhere(Enew == g2_ri))]) - float(
    linear(Enew[np.argwhere(d == max(d[int(np.argwhere(Enew == g2_li)):int(np.argwhere(Enew == g2_ri))]))],
           lin_grad_guess,
           lin_cut_guess)))  # Maximums von d im Bereich vom Index der linken Gauss-Grenze bis zum Index der rechten Gauss-Grenze der Energie, abzüglich vom Funktionswert der Exponentialfunktion an der Stelle des Energiemaximums der Gaussfunktion

g2_wid_guess = 0.12  # HWHM oder Standardabweichung

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
                x=Enew)  # Initialfunktion: mod-Funktion wird "evaluiert",d.h. sie berechnet die Start-Funktion mit den Startparametern. Verwendert dabei für die Parameter das Array pars, welches die Startparamater enthält. Als Abszissenwerte werden die x-Variablen der mod-Funktion verwendet, die wiederum den Energiewerten Enew zugeordnet werden.

out = mod.fit(d, pars,
              x=Enew)  # Fitfunktion: Bei gegebenem Datensatz d wird die bestmögliche mod-Funktion, die durch Start-Parameternim Array pars gegeben sind, erstellt. Als Abszissenwerte werden die x-Variablen der mod-Funktion verwendet, die wiederum den Energiewerten Enew zugeordnet werden.

dely = out.eval_uncertainty(x=Enew,
                            sigma=2)  # Bestimmen der Konfidenzintervalle für die Fitfunktion (nicht für die einzelnen Parameter). Sigma gibt das Signifikanzniveau an

# Erstellt eine Umgebung für die Resultate des Fits, bestehend aus dem Datensatz, Initial-Funktion, bester Fit, und deren Komponenten des besten Fits

fig, axes = plt.subplots(1, 2, figsize=(12.8,
                                        4.8))  # erstellt 1x2=2 Diagrammen in Form einer 1x2 Matrix an Diagrammen (Anzahl Zeilen, Anzahl Spalten). Die Matrix besitzen die gegebene Gesamtbreite und -höhe

# Erstes Diagramm (Position 0)

axes[0].plot(Enew, d, 'b',
             label='Daten')  # Plot im nullten Diagramm, Abszissendatensatz = Enew, stellt den zu fittenden Datensatz grafisch dar, Farbe = blau, betitelt mit "Daten"

axes[0].plot(Enew, init, 'k--',
             label='Startfit')  # Plot im nullten Diagramm, Abszissendatensatz = Enew, stellt die Funktion mit Startparametern grafisch dar, Farbe = schwarz, gestrichelt, betitelt mit "Startfit"

axes[0].plot(Enew, out.best_fit, 'r-',
             label='Bester Fit')  # Plot im nullten Diagramm, Abszissendatensatz = Enew, stellt die beste Fit-Funktion mit den gegebenen Startparametern durch den Ausdruck out.best_fit grafisch dar, Farbe = rot, betitelt mit "bester Fit"

axes[0].fill_between(Enew, out.best_fit - dely, out.best_fit + dely,
                     color='#888888')  # Füllt die Fläche zwischen zwei Funktionen mit einer Farbe deiner Wahl aus. Die zwei Funktionen stellen in diesem Fall die Konfidenzintervallsgrenzen für die Best-Fit-Funktion dar.

axes[0].legend(
    loc='best')  # erstellt eine Legende mit den betitelten Funktionen, positioniert sich automatisch da, wo die Plots am wenigsten durch die Legende überdeckt werden

# Zweites Diagramm (Position 1)

comps = out.eval_components(
    x=Enew)  # gibt die Komponenten der erstellten Fitfunkten ("out") in einem Array an. Array besteht aus exp_ ,g1_ ,g2_ mit Abszissendatensatz Enew

axes[1].plot(Enew, d,
             'b')  # Plot im ersten Diagramm, Abszissendatensatz = Enew, stellt den zu fittenden Datensatz grafisch dar, Farbe = blau, betitelt mit "Daten"

axes[1].plot(Enew, comps['lin_'], 'g--', label='Lineare Komponente')

axes[1].plot(Enew, comps['g1_'], 'm--',
             label='Gauß-Komponente 1')  # Plot im ersten Diagramm, Abszissendatensatz = Enew, stellt erste Gausskomponente grafisch dar, Farbe = violett, gestrichelt, betitelt mit "Gauß-Komponente 1"

axes[1].plot(Enew, comps['g2_'], 'k--',
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