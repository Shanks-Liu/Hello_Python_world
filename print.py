# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 19:51:24 2018

@author: 游侠-Speed
"""

def print_models(unprint_names, complete_names):
    while unprint_names:
        current_name = unprint_names.pop()
        print('now printing this: ' + current_name)
        complete_names.append(current_name)


def show_models(complete_names):
    print('the fllowing models have been printed: ')
    for complete_name in complete_names:
        print('\t' + complete_name)
        