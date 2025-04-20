# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:39:53 2018

@author: 游侠-Speed
"""

class User():
    def __init__(self, first_name, last_name, *user_infos):
        self.first_name = first_name
        self.last_name = last_name
        self.login_attempts = 0
        self.full_name = first_name +" " + last_name
        self.user_infos = user_infos
    def describe_user(self):
        print("The user'name is: " + self.full_name)
        for self.user_info in self.user_infos:
            print("The user'information are: " + str(self.user_info))
    def greet_user(self):
        print("Hello! " + self.full_name)
    def increment_login_attempts(self):
        self.login_attempts += 1
    def reset_login_attempts(self):
        self.login_attempts = 0
    def read_login_attempts(self):
        print("Now, There have " + 
              str(self.login_attempts) + 
              " users is logging")

