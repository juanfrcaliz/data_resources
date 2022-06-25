#!/usr/bin/env python

from distutils.core import setup

setup(name='data_resources',
      version='1.0',
      description='Resources for working with data',
      author='juanfrcaliz',
      packages=['corr_limit', 'data_visualize', 'db_connect', 
                'impute_missing_values', 'model_score', 'nlp_tools'],
     )
