# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 17:54:16 2018

@author: 游侠-Speed
"""

import unittest
from survey import AnonymousSurvey

class TestAnonymousSurvey(unittest.TestCase):
    
    def setUp(self):
        question = "What's your name"
        self.my_survey = AnonymousSurvey(question)
        self.responses = ["liu", "wang", "zhao"]
        
    def test_store_single_response(self):
        self.my_survey.store_response(self.responses[0])
        self.assertIn(self.responses[0], self.my_survey.responses)
        
    def test_store_three_response(self):
        self.my_survey.store_response(self.responses)
        self.assertIn(self.responses, self.my_survey.responses)
#        for response in self.responses:
#            self.my_survey.store_response(response)
#        for response in self.responses:
#            self.assertIn(response, self.my_survey.responses)

unittest.main()
        