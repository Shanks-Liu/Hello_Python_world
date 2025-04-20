# -*- coding: utf-8 -*-
"""
Created on Tue May  1 21:33:14 2018

@author: 游侠-Speed
"""

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

plt.figure(figsize=(16,8))
m = Basemap(llcrnrlon=77, llcrnrlat=14, urcrnrlon=140, urcrnrlat=51, 
            projection='lcc', lat_1=33, lat_2=45, lon_0=100)
m.readshapefile('CHN_adm1', 'states', drawbounds=True)
m.readshapefile(r"China_zip\TWN_adm_shp\TWN_adm0", "taiwan", drawbounds=True)
#plt.rcParams['axes.facecolor'] = 'blue'
m.drawcoastlines()
m.drawcountries(linewidth=1.5)

plt.show()