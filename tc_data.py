""" A python class that read the topcoder data into pandas with some preprocessing of text."""
import os
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
from bs4 import BeautifulSoup, NavigableString, Tag
from sklearn import preprocessing

from preprocessing_util import remove_url, remove_punctuation, remove_digits, consine_similarity

def extract_txt_from_node(node, is_nav=False, rm_url=True, rm_punc=False, rm_digits=False, rm_uppercase=False, delimiter=' '):
    """ Extract text from given node of a HTML parse tree, remove url, words with digits, punctuation."""
    text = node.strip() if is_nav else node.get_text()

    if rm_url:
        text = remove_url(text)
    if rm_punc:
        text = remove_punctuation(text)
    if rm_digits:
        text = remove_digits(text)
    if rm_uppercase:
        text = text.lower()

    return delimiter.join(text.split())

def extract_sections_from_html(req):
    """ Extract text from html formatted string.
        Divide text into sections by h-tags
    """
    sectioned_req_dct = defaultdict(list)
    soup = BeautifulSoup(req, 'html.parser')
    
    # There are some img tags and a tags that won't be extracted below, do it now.
    if soup.a:
        soup.a.decompose()
    if soup.img:
        soup.img.decompose()

    all_header_tags = soup.find_all(re.compile(r'^h'))
    
    if len(all_header_tags) == 0:
        return {'no_header_tag': extract_txt_from_node(soup)}
    
    for header in all_header_tags:
        section_name = extract_txt_from_node(header, rm_punc=True, rm_digits=True, rm_uppercase=True, delimiter='_')
        nxt_node = header
        while True:
            nxt_node = nxt_node.nextSibling
            
            if nxt_node is None:
                break
                
            if isinstance(nxt_node, NavigableString):
                sectioned_req_dct[section_name].append(extract_txt_from_node(nxt_node, is_nav=True))
            if isinstance(nxt_node, Tag):
                if nxt_node.name.startswith('h'):
                    break
                sectioned_req_dct[section_name].append(extract_txt_from_node(nxt_node))
    
    return {sec_name: ' '.join(' '.join(sec_reqs).split()) for sec_name, sec_reqs in sectioned_req_dct.items()}

class TopCoder:
    """ Read the detailed requirements and numeric data of challenges
        into pandas DataFrame.
        Also extract the text from HTML segments and get rid of url links.
    """

    data_path = os.path.join(os.curdir, 'data')
    cbf_path = os.path.join(data_path, 'challenge_basic_info.json') # challenge basic info path
    dreq_path = os.path.join(data_path, 'detail_requirements.json') # detailed requirement path
    tech_path = os.path.join(data_path, 'tech_by_challenge.json') # technology listed by challenge
    dvec_path = os.path.join(data_path, 'document_vec_100D.json')
    score_path = os.path.join(data_path, 'challenge_score_stat.json')
    cha_reg_dir = os.path.join(data_path, 'challenge_registration')

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
        self.global_features = self.extract_global_context_features()

    def process_detailed_requirements(self) -> (pd.DataFrame, pd.DataFrame):
        """ Process the detailed requirements from loaded json"""
        processed_reqs = defaultdict(dict)
        processed_ttls = defaultdict(dict)

        with open(self.dreq_path) as fjson:
            detailed_reqs = json.load(fjson)

        for req in detailed_reqs:
            processed_reqs[req['project_id']][req['challenge_id']] = extract_sections_from_html(req['requirements'])
            processed_ttls[req['project_id']][req['challenge_id']] = req['title']

        flatten_reqs = {
                (project_id, challenge_id, sec_name): {'requirements_by_section': sec_text}
                for project_id, challenges in processed_reqs.items()
                for challenge_id, req in challenges.items()
                for sec_name, sec_text in req.items()
        } # multi-for dict comprehension > nested for loops with extra variable declared ;-)

        flatten_ttls = {(project_id, challenge_id): {'title': title} for project_id, challenges in processed_ttls.items() for challenge_id, title in challenges.items()}

        df_requirements = pd.DataFrame.from_dict(flatten_reqs, orient='index')
        df_requirements = df_requirements.loc[df_requirements['requirements_by_section'] != '']
        df_requirements.index.names = ['project_id', 'challenge_id', 'section_name']

        df_titles = pd.DataFrame.from_dict(flatten_ttls, orient='index')
        df_titles.index.names = ['project_id', 'challenge_id']

        return df_titles, df_requirements

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
        cbi_df['sub_reg_ratio'] = (cbi_df.number_of_submitters / cbi_df.number_of_registration).round(2)
        cbi_df = cbi_df.loc[cbi_df.challenge_duration >= 0]

        score_stat_df = self.create_df_from_json(self.score_path, index_col='challenge_id')

        return cbi_df.join(score_stat_df, how='inner')

    def get_filtered_challenge_id(self):
        """ Return filtered challenges IDs for selecting training data.
            This method get identical result of filtered challenges as previous pricing models
        """
        with open(self.dvec_path) as fread:
            doc_vec_id = list(json.load(fread).keys())
        with open(self.tech_path) as fread:
            cha_tech_id = [cha['challenge_id'] for cha in json.load(fread)]

        cbi_df = self.challenge_basic_info.loc[self.challenge_basic_info.total_prize > 0]
        req_index = self.requirements.index.get_level_values(1).unique().to_series()

        hand_pick_cha_id = pd.concat([
            cbi_df.loc[
                (cbi_df.subtrack == subtrack) &
                (low <= cbi_df.total_prize) &
                (cbi_df.total_prize <= high)
            ] for subtrack, (low, high) in self.develop_challenge_prize_range.items() if subtrack in ('FIRST_2_FINISH', 'CODE')
        ]).index

        filtered_index = req_index[req_index.index.isin(hand_pick_cha_id) & req_index.index.isin(doc_vec_id) & req_index.index.isin(cha_tech_id)].index
        return filtered_index

    def get_filtered_challenge_info(self):
        """ Return the copy of filtered challenges."""
        return self.challenge_basic_info.loc[self.challenge_basic_info.index.isin(self.get_filtered_challenge_id())].copy().sort_index()

    def get_challenge_overview(self):
        """ For the challenge requirement text that has some sort of overview section
            extract the text from that section.
        """
        overview_sections = [
            ('overview' in sec_name and 
            'project' not in sec_name and 
            'technology_overview' not in sec_name
            ) for sec_name in self.requirements.index.get_level_values(2)
        ]
        overview_df = self.requirements.loc[overview_sections]

        return overview_df.groupby(level=1).aggregate(lambda sec_strs: sec_strs[np.argmax([len(s) for s in sec_strs])])

    def get_filtered_requirements(self, extract_overview=False):
        """ Return the copy of filtered requirements."""
        filtered_cha_id = self.get_filtered_challenge_id()
        cha_req = self.requirements.groupby(level=1).aggregate(' '.join)

        if extract_overview:
            overview_df = self.get_challenge_overview()
            return pd.concat([
                cha_req.loc[~cha_req.index.isin(overview_df.index) & cha_req.index.isin(filtered_cha_id)],
                overview_df.loc[overview_df.index.isin(filtered_cha_id)]
            ]).rename(columns={'requirements_by_section': 'requirements'}).sort_index()
        else:
            return cha_req.loc[cha_req.index.isin(filtered_cha_id)].rename(columns={'requirements_by_section': 'requirements'}).sort_index()

    def calculate_tech_popularity(self):
        """ Calculate popularity of used technology in filtered challenges"""
        filtered_cha_id = self.get_filtered_challenge_id()
        with open(self.tech_path) as fread:
            cha_tech_dct = {cha['challenge_id']: cha['tech_lst'] for cha in json.load(fread) if cha['challenge_id'] in filtered_cha_id}

        tech_popularity = defaultdict(int)
        clean_cha_tech_dct = {}
        for cha_id, tech_lst in cha_tech_dct.items():
            clean_tech_lst = ['angularjs' if 'angular' in t.lower() else '_'.join(t.lower().split()) for t in tech_lst]
            clean_cha_tech_dct[cha_id] = clean_tech_lst
            for t in clean_tech_lst:
                tech_popularity[t] += 1

        tech_popularity_df = pd.Series(tech_popularity).sort_values(ascending=False).to_frame().reset_index().head(30)
        tech_popularity_df.columns = ['tech', 'popularity']

        tech_pop_norm = (tech_popularity_df['popularity'] - tech_popularity_df['popularity'].mean()) / tech_popularity_df['popularity'].std()
        tech_popularity_df['softmax_popularity'] = np.exp(tech_pop_norm) / np.sum(np.exp(tech_pop_norm)) # get a softmax encoded popularity score

        encoded_tech_dct = {cha_id: {t: int(t in tech_lst) for t in tech_popularity_df['tech']} for cha_id, tech_lst in clean_cha_tech_dct.items()}
        encoded_tech_df = pd.DataFrame.from_dict(encoded_tech_dct, orient='index')

        return tech_popularity_df, encoded_tech_df

    def get_encoded_tech_feature(self):
        """ Calculate 0-1 encoded used tech features as well as softmax popularity score of challenges."""
        tech_pop, encoded_tech = self.calculate_tech_popularity()
        return encoded_tech, encoded_tech.apply(lambda col: col * tech_pop.loc[tech_pop['tech'] == col.name, 'softmax_popularity'].iloc[0]).sum(axis=1).to_frame().rename(columns={0: 'softmax_sum'})

    def get_meta_data_features(self,
        return_tensor=False,
        standardize=False,
        normalize=False,
        encoded_tech=False,
        softmax_tech=False,
        contain_dv=False,
        contain_prize=False,
        return_df=False
        ):
        """ Get meta data as training features.

            :param return_tensor: when True, return a list of Tensor(shape)
            :param normalize: whether to use (df - df.mean()) / df.std() to normalize
            data
        """
        cbi = self.get_filtered_challenge_info()
        metadata_cols = ['number_of_platforms', 'number_of_technologies', 'project_id', 'challenge_duration']\
            + (['total_prize'] if contain_prize else [])
        metadata_df = cbi.reindex(metadata_cols, axis=1).join(cbi['subtrack_category'].cat.codes.rename('subtrack'))

        if encoded_tech or softmax_tech:
            encoded_tech_df, softmax_tech_df = self.get_encoded_tech_feature()
            if softmax_tech:
                metadata_df = pd.concat([metadata_df, softmax_tech_df], axis=1)
            if encoded_tech:
                metadata_df = pd.concat([metadata_df, encoded_tech_df], axis=1)

        if contain_dv:
            metadata_df = metadata_df.join(
                pd.read_json(
                    os.path.join(os.curdir, 'data', 'new_docvec.json'),
                    orient='index'
                ).rename(columns={i: f'dv{i}' for i in range(100)})
            )

        if standardize:
            index, columns = metadata_df.index, metadata_df.columns
            std_data = preprocessing.scale(metadata_df)
            metadata_df = pd.DataFrame.from_records(std_data, index=index, columns=columns)

        if normalize:
            index, columns = metadata_df.index, metadata_df.columns
            norm_data = preprocessing.normalize(metadata_df)
            metadata_df = pd.DataFrame.from_records(norm_data, index=index, columns=columns)

        if return_df:
            return metadata_df

        return [tf.constant(row) for row in metadata_df.itertuples(index=False)] if return_tensor else metadata_df.to_numpy(copy=True)

    def get_bert_encoded_txt_features(self, tokenizer, extract_overview=False, return_tensor=False):
        """ Method that return encoded text from the bert tokenizer"""
        req = self.get_filtered_requirements(extract_overview)
        batch_encoding = tokenizer(req['requirements'].to_list(), padding=True, truncation=True, return_tensors='tf' if return_tensor else None)
        return batch_encoding.data

    def get_target(self, return_tensor=False):
        """ Return the total prize as np.array or Tensor."""
        prz_arr = self.get_filtered_challenge_info()['total_prize'].to_numpy()

        return [tf.constant(prz) for prz in prz_arr] if return_tensor else prz_arr

    def extract_global_context_features(self):
        """ Detect the challenges that are open simutanously
            Count the activate workers who are in the registration of some tasks.
        """
        cha_info = self.get_filtered_challenge_info()
        cha_start_and_end_date = cha_info.reindex(['registration_start_date', 'submission_end_date'], axis=1)

        # The feature we use is a vector of [*metadata, *docvec]
        feature_df = self.get_meta_data_features(
            encoded_tech=True,
            softmax_tech=True,
            contain_dv=True,
            contain_prize=True,
            standardize=True,
            normalize=True,
            return_df=True,
        )

        new_feature_dct = {}
        for cha_id, start_date, end_date in cha_start_and_end_date.itertuples():
            new_feature_dct[cha_id] = self.detect_competing_cha(
                cha_id,
                start_date,
                end_date,
                cha_start_and_end_date.loc[cha_start_and_end_date.index != cha_id],
                feature_df,
                cha_info['project_id'],
            )

        return pd.DataFrame.from_dict(new_feature_dct, orient='index')

    def detect_competing_cha(self,
        challenge_id: int,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        challenges: pd.DataFrame,
        feature_df: pd.DataFrame,
        project_ids: pd.Series,
        ):
        """ Detect a group of stuff."""
        competing_cha = [cha_id for cha_id, sub in challenges['submission_end_date'].iteritems() if start_date <= sub <= end_date]

        target_cha_vec = feature_df.loc[challenge_id].to_numpy()
        cos_sim_result = {}
        active_workers = set()
        for comp_cha_id in competing_cha:
            competing_cha_vec = feature_df.loc[comp_cha_id].to_numpy()
            cos_sim_result[comp_cha_id] = consine_similarity(target_cha_vec, competing_cha_vec)

            with open(os.path.join(self.cha_reg_dir, f'challenge_registration_{comp_cha_id}.json')) as f:
                active_workers.update([cha_reg['username'] for cha_reg in json.load(f)])

        # aggregate the count of competing challenges by similarity intervals.
        cos_sim_sr = pd.Series(cos_sim_result).round(6)
        similarity_intervals = np.linspace(-1, 1, 21)
        num_of_tasks_by_sim_sr = pd.cut(cos_sim_sr, bins=similarity_intervals).value_counts().sort_index()
        num_of_tasks_by_sim_dct = {f'num_of_tasks_sim{intv}': cnt for intv, cnt in num_of_tasks_by_sim_sr.to_dict().items()}

        if len(cos_sim_sr) != num_of_tasks_by_sim_sr.sum():
            raise ValueError(f'Slicing intervals failed, total of {len(cos_sim_sr)}, sliced out {num_of_tasks_by_sim_sr.sum()}')

        # check if there are challenges under the same project as target challenge in competing challenges
        same_proj_indicator = pd.Series({cha_id: int(project_ids[challenge_id] == project_ids[cha_id]) for cha_id in competing_cha})
        ratio_same_proj = same_proj_indicator.mean()

        return {
            'ratio_of_same_project': ratio_same_proj if not np.isnan(ratio_same_proj) else -1,
            'num_of_competing_tasks': len(competing_cha),
            **num_of_tasks_by_sim_dct,
            'num_of_active_workers': len(active_workers),
        }

    def build_final_dataset(self, target: str):
        """ Build the dataset that combines metadata, document vectors and """
        if target not in ('total_prize', 'avg_score', 'number_of_registration', 'sub_reg_ratio'):
            raise ValueError('target is not valid, only options are \'total_prize\', \'avg_score\', \'number_of_registration\', \'sub_reg_ratio\'.')

        contain_prize = not bool(target == 'total_prize')

        metadata_features = self.get_meta_data_features(
            encoded_tech=True,
            softmax_tech=True,
            contain_dv=True,
            contain_prize=contain_prize,
            return_df=True,
        )

        X = pd.concat([self.global_features, metadata_features], axis=1)
        y = self.get_filtered_challenge_info()[target]

        return X, y
