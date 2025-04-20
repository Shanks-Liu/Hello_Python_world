# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 10:41:12 2018

@author: 游侠-Speed
"""

with open ("python_practice\guest.text", "w") as file:
    active = True
    while active:
        x = input("Please enter your name: ")
        if x =='quit':
            break
        x=x/2
        y = input("Why do you like programming: ")
        file.write(x + " log in \n")
        print("Hello, " + x)
        with open ("python_practice\programming.text", "a") as file1:
            file1.write(y + "\n")