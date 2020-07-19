""" A python class that read the topcoder data into pandas with some preprocessing of text."""
import os
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, NavigableString, Tag

from preprocessing_util import remove_url

class TopCoder:
    """ Read the detailed requirements and numeric data of challenges
        into pandas DataFrame.
        Also extract the text from HTML segments and get rid of url links.
    """

    data_path = os.path.join(os.curdir, 'data')
    cbf_path = os.path.join(data_path, 'challenge_basic_info.json') # challenge basic info path
    dreq_path = os.path.join(data_path, 'detail_requirements.json') # detailed requirement path
    dvec_path = os.path.join(data_path, 'document_vec_100D.json')

    develop_challenge_prize_range = {
        'FIRST_2_FINISH': (0, 600),
        'CODE': (250, 2500),
        'ASSEMBLY_COMPETITION': (750, 2750),
        'BUG_HUNT': (0, 750),
        'UI_PROTOTYPE_COMPETITION': (1250, 2750),
        'ARCHITECTURE': (1500, 3000),
        'DEVELOP_MARATHON_MATCH': (1000, 1750),
        'COPILOT_POSTING': (150, 300),
        'TEST_SUITES': (500, 2000),
        'TEST_SCENARIOS': (500, 2000),
        'SPECIFICATION': (1500, 3000),
        'CONTENT_CREATION': (500, 2000),
        'CONCEPTUALIZATION': (1500, 2000)
    }

    def __init__(self):
        self.titles, self.requirements = self.process_detailed_requirements()
        self.challenge_basic_info: pd.DataFrame = self.read_challenge_basic_info()

    def process_detailed_requirements(self) -> (pd.DataFrame, pd.DataFrame):
        """ Process detailed requirments.
            NOTE: All the text are kept case sensitive as is.
        """
        processed_req = {}
        processed_ttl = {}

        with open(self.dreq_path) as fread:
            requirements = json.load(fread)

        for req in requirements:
            processed_req[req['challenge_id']] = self.preprocess_req_txt(req['requirements'])
            processed_ttl[req['challenge_id']] = req['title']

        df_req = pd.DataFrame.from_dict(processed_req, orient='index')
        df_req.index.names = ['challenge_id']
        df_req.columns = ['requirement']

        df_ttl = pd.DataFrame.from_dict(processed_ttl, orient='index')
        df_ttl.index.names = ['challenge_id']
        df_ttl.columns = ['title']

        return df_ttl, df_req

    def preprocess_req_txt(self, req):
        """ Preprocess requirement text, extract text from HTML segments."""
        soup = BeautifulSoup(req, 'html.parser')

        # There are some img tags and a tags that won't be extracted below, do it now.
        if soup.a:
            soup.a.decompose()
        if soup.img:
            soup.img.decompose()

        return ' '.join(remove_url(soup.get_text()).split())

    def create_df_from_json(self, fn, orient='records', index_col=None, convert_dates=None, convert_cat=None):
        """ Read the given json file into a pandas dataframe."""
        with open(fn) as fread:
            df = pd.read_json(fread, orient=orient, convert_dates=convert_dates or False)

        if index_col:
            df.set_index(index_col, inplace=True)
            if 'date' in index_col: # convert the datetime index to datetime object
                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)

        if convert_cat is not None and isinstance(convert_cat, list):
            df[[f'{col}_category' for col in convert_cat]] = df[convert_cat].astype('category')

        return df

    def read_challenge_basic_info(self):
        """ Read challenge basic info and add challenge duration column into the DataFrame"""
        cbi_df = self.create_df_from_json(
            self.cbf_path,
            index_col='challenge_id', 
            convert_dates=['registration_start_date', 'registration_end_date', 'submission_end_date'],
            convert_cat=['track', 'subtrack']
        )
        cbi_df['challenge_duration'] = (cbi_df.submission_end_date - cbi_df.registration_start_date).apply(lambda td: td.days)

        return cbi_df.loc[cbi_df.challenge_duration >= 0].copy()

    def get_filtered_challenge_id(self):
        """ Return filtered challenges IDs for selecting training data.
            This method get identical result of filtered challenges as previous pricing models
        """
        with open(self.dvec_path) as fread:
            doc_vec_id = list(json.load(fread).keys())

        cbi_df = self.challenge_basic_info.loc[self.challenge_basic_info.total_prize > 0]

        hand_pick_cha_id = pd.concat([
            cbi_df.loc[
                (cbi_df.subtrack == subtrack) &
                (low <= cbi_df.total_prize) &
                (cbi_df.total_prize <= high)
            ] for subtrack, (low, high) in self.develop_challenge_prize_range.items() if subtrack in ('FIRST_2_FINISH', 'CODE')
        ]).index

        filtered_index = self.requirements.loc[self.requirements.index.isin(hand_pick_cha_id) & self.requirements.index.isin(doc_vec_id)].index
        return filtered_index

    def get_filtered_challenge_info(self):
        """ Return the copy of filtered challenges."""
        return self.challenge_basic_info.loc[self.challenge_basic_info.index.isin(self.get_filtered_challenge_id())].copy().sort_index()

    def get_filtered_requirements(self):
        """ Return the copy of filtered requirements."""
        return self.requirements.loc[self.requirements.index.isin(self.get_filtered_challenge_id())].copy().sort_index()
