# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 21:47:45 2020

@author: QuanMin
"""

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math

#url = r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/archived_data/archived_time_series/time_series_19-covid-Confirmed_archived_0325.csv'
#df = pd.read_csv(url)
#ushist = {}
#for index, row in df.iterrows():
#    l = row.tolist()
#    if l[1] == 'US' and not (',' in l[0]):
#        ushist[l[0]] = l[2:]
#        
#with open('usdata.json', 'w') as f:
#    json.dump(ushist, f)

with open('usdata.json', 'r') as f:
    hist = json.load(f)    
    
def expfunc(x, a, b, c):
    return a*(1 - np.exp(-b*(x-c)))

def jac(x, a, b, c):
    return np.array([1 - np.exp(-b*(x-c)), a*(x-c)*np.exp(-b*(x-c)), -a*b*np.exp(-b*(x-c))]).transpose()

plt.close('all')

cnt = 1
fig = plt.figure()
para = {}
for rec in hist:
    if hist[rec][-1] < 100:
        continue
    y = np.array(hist[rec][2:])
    y = np.log(y[y>0])
    days = len(y)
    x = np.arange(days)
    
    try:
        print(rec)
        ax = fig.add_subplot(9,4, cnt)
        ax.set_title(rec, x=0.5, y=0.0)
#        plt.subplot(6,3, cnt)    
        cnt += 1
        plt.plot(x, np.exp(y),'o')
        sigmafactor = -0.5
        sigma = [float(s+1)**sigmafactor for s in range(days)]
        sigma[0] = sigma[-1]/2
        popt, pcov = curve_fit(expfunc, x, y, p0=(10000, 0.2, 0), jac=jac, sigma = sigma)
        plt.plot(x, np.exp(expfunc(x, *popt )))
        plt.yscale('log')
        a, b, c = popt
        para[rec] = popt
        plt.title(rec + '\n' + str(int(np.exp(a))) + '\n' + str(round(a*b*math.exp(b*c),3)) + ' ' +str(round(1/b)))
#        plt.show()
    except RuntimeError:
#        plt.show()
        coef = np.polyfit(x, y, 1)
        plt.plot(x, np.exp(coef[1] + coef[0]*x))
        plt.title(rec + '\n' + str(round(coef[0],3)))
        plt.yscale('log')
        print('error')
        
        
        
# Rolling horizon
        
cnt = 1
fig = plt.figure()
para = {}
for rec in hist:
    if hist[rec][-1] < 100 or rec in ['Virgin Islands']:
        continue
    y0 = np.array(hist[rec][2:])
    y0 = np.log(y0[y0>0])
    days = len(y0)
    
    print(rec)
    ax = fig.add_subplot(8,4, cnt)
    ax.set_title(rec, x=0.5, y=0.0)
#        plt.subplot(6,3, cnt)    
    cnt += 1
    increase = []
    x = np.arange(7)
    for i in range(days - 6):
    
        try:
            y = y0[i:i+7]
            
            sigmafactor = -0.5
            sigma = [float(s+1)**sigmafactor for s in range(7)]
            sigma[0] = sigma[-1]/2
            popt, pcov = curve_fit(expfunc, x, y, p0=(10000, 0.2, 0), jac=jac, sigma = sigma)

            a, b, c = popt
            para[rec] = popt
            rate = a*b*math.exp(-b*(6-c))
    #        plt.show()
        except RuntimeError:
    #        plt.show()
            coef = np.polyfit(x, y, 1)
            reate = coef[0]
        
        increase.append(math.log(2)/rate)
    plt.plot(increase)
    plt.title(rec)
    plt.ylim([0,15])
        
        