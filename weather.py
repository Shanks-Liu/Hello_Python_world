# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 09:39:05 2018

@author: 游侠-Speed
"""

import csv
from matplotlib import pyplot as plt
from datetime import datetime

dates, highs, lows = [], [], []

with open ('sitka_weather_2014.csv') as f:
    reader = csv.reader(f)#用CSV里的reader函数
    row_head = next(reader)
    for row in reader:
        try:
            current_date = datetime.strptime(row[0], "%Y/%m/%d")
            current_high = int(row[1])
            current_low = int(row[3])
        except ValueError:
            print(current_date, "missing data")
        else:
            dates.append(current_date)
            highs.append(current_high)
            lows.append(current_low)

fig = plt.figure(dpi=141, figsize=(10, 6))#尺寸是根据这个而不是Anaconda里的设置
plt.plot(dates, highs, c='red')
plt.plot(dates, lows, c='blue')
plt.fill_between(dates, highs, lows, facecolor='blue', alpha=0.1)
plt.axis()
plt.title("weather", fontsize=24)
plt.xlabel("date", fontsize=14)
fig.autofmt_xdate()
plt.ylabel("high and low", fontsize=14)
plt.tick_params(axis="both", labelsize=14)

plt.show()