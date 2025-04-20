# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 11:29:12 2018

@author: 游侠-Speed
"""

import unittest
from survey import AnonymousSurvey

class TestAnonymousSurvey(unittest.TestCase):
    
    def test__store_single_response(self):
        question = "What's your name"
        my_survey = AnonymousSurvey(question)
        my_survey.store_response("Liu")
        self.assertIn("Liu", my_survey.responses)
        
    def test_store_three_response(self):
        question = "What's your name"
        my_survey = AnonymousSurvey(question)
        responses = ["Zhao", "wang", "li"]
        my_survey.store_response(responses)
        self.assertIn(responses, my_survey.responses)
#        for response in responses:
#            my_survey.store_response(response)
#        for response in responses:
#            self.assertIn(response, my_survey.responses)
        
unittest.main()