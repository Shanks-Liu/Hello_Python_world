# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 11:09:41 2018

@author: 游侠-Speed
"""

class AnonymousSurvey():
    
    def __init__(self, question):
        self.question = question
        self.responses = []
        
    def store_response(self, *new_responses):
        for new_response in new_responses:
            self.responses.append(new_response)
        
    def show_results(self):
        print("All responses are:" )
        for response in self.responses:
            print("\t" + str(response))
        