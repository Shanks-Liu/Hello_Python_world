# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 11:20:49 2018

@author: 游侠-Speed
"""

from survey import AnonymousSurvey

question = "What's your name"
my_survey = AnonymousSurvey(question)
my_survey.store_response("liu", "zhao")
my_survey.show_results()