import pandas as pd
import numpy as np
import re
import json
import torch
import random
import os
# from probes import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
# torch.set_printoptions(profile="full")
import re
import pandas as pd
from data_classes import BookData, MovieData
import argparse

# TODO: update build_full_df to include MovieData, change how MF works to pull in
# loaded model from pickle and individually get scores from that model for
# each user item pair.
# TODO: Change output calcs to evaluate ranking by score by user gender
    
class LLMRecs:
    def __init__(self, output_data_path, item_type, n):
        # Load outputs that ran on GPU
        self.outputs = torch.load(os.path.join(output_data_path, f'output_dict_{item_type}_{n}.pt'), map_location=torch.device('cpu'))
        
        # Load original recommendation data
        if item_type == 'book':
            rec_data = BookData()
            self.map_dicts = rec_data.get_map_dicts()
            self.reverse_title_dict = self.map_dicts.book_dict_reverse
            result_df = rec_data.merged()
            self.user_title_dict = rec_data.user_title_dict(result_df)
            self.str1 = "system\n\nCutting Knowledge Date: December 2023\nToday Date: 08 Nov 2025\n\nuser\n\nHi, I\'ve read and enjoyed the following books:"
            self.str1b = "Hi, I\'ve read and enjoyed the following books:"
            # self.str2 = """  Only return the 5 books you recommend in JSON format like {"Books": {\'title\':..., \'author\':...}}, and nothing else.assistant\n\n"""
            # self.str3 = """Please recommend new books based on the user\'s reading preferences and only return the 5 books you recommend in JSON format like {"Books": {\'title\':..., \'author\':...}}, and nothing else.assistant"""
            with open(os.path.join(output_data_path, f'gender_dict_{n}.json'), 'r') as f:
                self.gender_dict = json.load(f)
            verb = 'read'
        elif item_type == 'movie':
            verb = 'watch'
            rec_data = MovieData()
            result_df = rec_data.get_rating_df()
            self.gender_dict = rec_data.get_gender_dict()
            self.reverse_title_dict = rec_data.get_reverse_title_dict()

        self.str3 = f"Please rank the following {item_type}s in order from most to least likely to recommend to them to {verb} next: "

    
    def small_dict(self, df=True):
        """ Returns a version of the outputs without the hidden state repr"""
        small_dict = {}
        for k, v in self.outputs.items():
            inner_dict = {prop:value for prop, value in v.items() if ((prop != 'pre_hidden') and (prop != 'post_hidden'))}
            small_dict[k] = inner_dict
        if df:
            small_df = pd.DataFrame.from_dict(small_dict, orient='index').reset_index().rename(columns={'index':'user_id'})
            return small_df
        else: 
            return small_dict    
        
    def extract_titles_to_list(self, text):
        """ Takes in baseline or steered text and extracts the titles to a list in order of ranking"""
        if self.item_type == 'book':
            return re.findall(r'^\s*n?\d+\.\s*(.+?)\s*$', text, re.MULTILINE)
        elif self.item_type == 'movie':
            return re.findall(r'\d*\.\s*(.*?)\s*\(\d{4}\)', text)
    

    def map_titles(self, df):
        """ Tries to parse LLM output text  and map to titles in existing book dict"""
        # for k, values in self.outputs.items():
        for typ in ('baseline', 'steered'):
            df[f'{typ}_recs'] = df[f'{typ}_text'].str.split(self.str3).str[1].str.replace("\n","")
            df[f'{typ}_titles'] = df[f'{typ}_recs'].apply(self.extract_titles_to_list)
            df[f'{typ}_ids'] = df[f'{typ}_titles'].apply(
                lambda x: [
                    self.reverse_title_dict[item] for item in x if item in self.reverse_title_dict
                    ]
                )
        # And for the original sampled books can use either baseline or steered as they have the same starting string
        df['hist'] = df['baseline_text'].str.split(self.str3).str[0].str.split(self.str1b).str[1].str.replace(self.str1,"").apply(self.extract_books)
        df['hist_titles'] = df['hist'].apply(lambda x: [i['title'] for i in x])


    def count_from_hist(self, df):
        """ Count the instances that books in the users history were recommended to the user by the LLM"""
        df['titles'] = df['user_id'].map(self.user_title_dict)
        for typ in ('baseline','steered'):
            df[f'{typ}_rec_count'] = df[f'{typ}_titles'].apply(lambda row: len(row))
            # Find items that were recommended to the user that are in their history (but not in initial prompt)
            df[f'{typ}_count'] = df[['titles',f'{typ}_titles','hist_titles']].apply(lambda row: len(list(set(row['titles']).intersection(row[f'{typ}_titles']).difference(row['hist_titles']))), axis=1)
            # Count the number of generated recs that can be mapped to the book_dict
            df[f'{typ}_map_count'] = df[f'{typ}_ids'].apply(lambda row: len(row))


    def precision(self, df, k=5):
        """ Calculates and prints the precision of the recs """
        bigger_denom = len(df) * k
        for typ in ('baseline','steered'):
            count = df[f'{typ}_count'].sum()
            denom_count = df[f'{typ}_rec_count'].sum()
            print(
            f'Precision @ k: \n {typ}: ', 
             count/denom_count, 
            ", ",
            count/bigger_denom,
            '\n'
            )


    def user_gender(self, df):
        """ Append annotated user gender to our dict"""
        df['user_gender'] = df['user_id'].astype(str).map(self.gender_dict)     


    def author_gender(self, small_df):
        """ Append author gender where available to our dict"""
        for typ in ('baseline','steered'):
            small_df[f'{typ}_author_list'] = small_df[f'{typ}_ids'].apply(
            lambda row: [
                int(self.map_dicts.book_dict.get(str(item)).get('authors')[0]['author_id'])
                for item in row
                if str(item) in self.map_dicts.book_dict
                ]
            )

            small_df[f'{typ}_gender_list'] = small_df[f'{typ}_author_list'].apply(
                lambda row: [
                    self.map_dicts.author_gender.get(item).get('gender')
                    for item in row
                    if item in self.map_dicts.author_gender
                    ]
                )
        
            small_df[f'{typ}_gender_count'] = small_df[f'{typ}_gender_list'].apply(lambda row: len(row))
        


    def author_gender_count(self, df):
        # sum author genders by baseline and steered
        for value in ['baseline','steered']:
            print('Count by author gender: Baseline vs steered \n')
            print(df[f'{value}_gender_list'].explode().value_counts()/(df[f'{value}_gender_count'].sum()))

    def build_full_df(self):
        """ Append all the variables needed"""
        outputs_df = self.small_dict(df=True)
        # outputs = self.small_dict(df=False)
        if self.item_type == 'book':
            rec_data = BookData()
            df = rec_data.merged()
            mdf = rec_data.merged(agg=False)

        elif self.item_type == 'movie':
            rec_data  = MovieData()

        mfrec = MFRec(mdf, 'goodreads_data', False)
        preds_df = mfrec.do_mf()
        mf_author_gender_df = mfrec.eval_mf_by_author_gender(preds_df)

        self.map_titles(outputs_df)
        self.count_from_hist(outputs_df)
        self.author_gender(outputs_df)
        self.user_gender(outputs_df)
        outputs2 = self.lookup_mf_scores(outputs_df, preds_df, mfrec.user_ids_fromdf)
        # self.append_mf_to_df(outputs_df, outputs)
        outputs3 = self.append_mf_author_gender_counts(outputs2, mf_author_gender_df)

        return outputs3
    

    def lookup_mf_scores(self):
        """Use loaded model from pickle """

        
    def append_mf_to_df(self, small_df, outputs_dict):
        for typ in ('baseline','steered'):
            small_df[f'mf_{typ}_scores'] = small_df['user_id_from_df'].apply(lambda x: outputs_dict.get(x).get(f'mf_{typ}_score'))
            small_df[f'mf_{typ}_denom'] = small_df['user_id_from_df'].apply(lambda x: outputs_dict.get(x).get(f'mf_{typ}_denom'))

    def output_mf_scores(self, df):
        # Overall
        for typ in ('baseline','steered'):
            print(f"{typ} MF Score mean: ", np.sum(df[f'mf_{typ}_score'])/np.sum(df[f'mf_{typ}_denom']))
            result = df.groupby('user_gender').agg(
            sum_value1=(f'mf_{typ}_score', 'sum'),
            sum_value2=(f'mf_{typ}_denom', 'sum')
            )
            print(f'Type: {typ} - MF score, averaged by gender: ',result['sum_value1']/result['sum_value2'])

    def append_mf_author_gender_counts(self, df, ag_df):
        """Merge author genders from MF recs to rec df"""
        return pd.merge(df, ag_df, left_on='user_id_from_df', right_on='user_id',how='left')
    
    def output_mf_author_gender(self, df):
        """ Print the author gender breakdown of the MF recs"""
        print('Author gender breakdown for MF recommendations - using top ranked books from MF model')
        print(df['mf_author_gender'].explode().value_counts()/df['mf_author_gender'].explode().shape[0])
        

class MFRec:
    def __init__(self, item_type):
        """ Load the CF model pipeline that was saved"""
        with open(f'output_data/{item_type}/probes/rec_mf.pkl', 'rb') as file:
            self.pipe = pickle.load(file)

    def cf_predict(self, user_id, item_id):
        """ Get score for user and item_ids from saved model"""
        # Users and items can be lists
        return self.pipe.run(user_id, item_id)

        
    def eval_mf_by_author_gender(self, preds_df):
        """Evaluate the top ranked books for each user returned by the MF model"""
        n = 10 # Number of recommendations to return
        columns_to_consider = preds_df.columns.drop('user_id')
        nlargest_column_names = preds_df[columns_to_consider].apply(lambda row: row.nlargest(n).index.tolist(), axis=1)
        nlargest_column_names.name = 'mf_top_k'
        output = pd.concat([preds_df['user_id'], nlargest_column_names], axis=1)
        # TO DO: Map titles to author ids
        output['author_id'] = output.mf_top_k.apply(
            lambda x: [
            self.map_dicts.book_dict[self.map_dicts.book_dict_reverse[i]]['authors'][0]['author_id']
            for i in x
                ]
            )

        output['mf_author_gender'] = output['author_id'].apply(
            lambda x: [
                self.map_dicts.author_gender.get(int(i),{'gender':'unknown'}).get('gender')
                  for i in x
                  ]
                )
        for gender in ('female','male','unknown'):
            output[f'mf_{gender}_count'] = output['mf_author_gender'].apply(lambda x: len([i for i in x if i==gender]))
        return output


        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--size_of_sample", type=int, default=10, help="Number of users to include")
    parser.add_argument("-i","--item_type",type=str, help='Type of item recommendation')
    parser.add_argument('-m','--model_name', type=str, help ='Model name')
    args = parser.parse_args()

    if args.item_type == 'book':
        folder = '~/Documents/repos/probing_classifiers/goodreads_data'
    elif args.item_type == 'movie':
        folder = '~/Documents/repos/Glocal_K/1/ml-100k'

    llm_recs = LLMRecs(folder, args.item_type, args.size_of_sample)
    outputs_df = llm_recs.build_full_df()
    
    if args.item_type == 'book':
        llm_recs.output_mf_author_gender(outputs_df)

    # # Print summary statistics
    llm_recs.author_gender_count(outputs_df)
    llm_recs.precision(outputs_df)
    llm_recs.output_mf_scores(outputs_df)

    

    
    # rec_data = RecData('goodreads_data')
    # df = rec_data.merged()
    # mdf = rec_data.merged(agg=False)

    # mfrec = MFRec(mdf, 'goodreads_data', True)
    # mfrec.eval_mf_by_author_gender()

