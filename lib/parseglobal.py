# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:41:59 2020

@author: SHAOQM
"""
import time
import requests
import json
# import matplotlib.pyplot as plt
import numpy as np


url = r'https://covid19.tk/by_world.json?t='


# Get Data
try: response
except NameError: response = None

if response is None:
    try:
        with open('globalhist.json', 'r') as f:
            response = json.load(f)    
    except FileNotFoundError:
        response = json.loads(requests.get(url + str(int(time.time()*1000)), verify=False ).text)
        with open('globalhist.json', 'w') as f:
            json.dump(response, f)
            print('saved')
           
over100 = {}            
for res in response:
    if res['confirmedCount']>200 and len(res['countryName'])<7:
        records = []
        for rec in res['records']:
            if rec['confirmedCount'] < 200:
                continue
            records.append(rec['confirmedCount'])
        if len(records)>=4:
            over100[res['countryName']] = records

# for rec in over100:
#     print(rec, over100[rec])
#     plt.plot(np.log10(over100[rec]), '-o')
# plt.legend(over100.keys())