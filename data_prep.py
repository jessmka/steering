import pandas as pd
import numpy as np
import os
import argparse
import random
import json

NUM_SHOT = 5

class MovieData:
    def __init__(self):
        data_path = '/Users/jessicakahn/Documents/repos/Glocal_K/1'
        header = ['user_id', 'item_id', 'rating', 'timestamp']
        self.all_rating_df = pd.read_csv(os.path.join(data_path,'ml-100k/u.data'), sep='\t', names=header)
        
        user = pd.read_csv(os.path.join(data_path,'ml-100k/u.user'), sep="|", encoding='latin-1', header=None)
        user.columns = ['user_id', 'age', 'gender', 'occupation', 'zip code']
        self.gender_dict = user.set_index('user_id')['gender'].to_dict()

        item = pd.read_csv(os.path.join(data_path,'ml-100k/u.item'), sep="|", encoding='latin-1', header=None)
        item.columns = ['movie_id', 'movie_title' ,'release_date','video_release_date', 'IMDb_URL', 'unknown', 'Action', 
                'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 
                'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        item = item.set_index('movie_id')
        self.title_dict = item['movie_title'].to_dict()

    def get_rating_df(self):
        return self.all_rating_df
    
    def get_title_dict(self):
        return self.title_dict
    
    def get_gender_dict(self):
        return self.gender_dict
    
class MusicData:
    def __init__(self):
        splits = {'train': 'train.gz.parquet', 'valid': 'valid.gz.parquet', 'test': 'test.gz.parquet'}
        df = pd.read_parquet("hf://datasets/matthewfranglen/lastfm-1k/" + splits["train"])
        self.title_dict = df.set_index('track_index')['track_name'].to_dict()
        self.gender_dict = df.set_index('user_index')['gender'].to_dict()
        self.artist_dict = df.set_index('track_index')['artist_name'].to_dict()
        self.rating_df = df.groupby(['user_index','track_index'])['timestamp'].count().reset_index()
        
        self.rating_df.columns = ['user_id','item_id','rating']
    
    def get_rating_df(self):
        return self.rating_df

    def get_title_dict(self):
        return self.title_dict
    
    def get_gender_dict(self):
        return self.gender_dict
    
    def get_artist_dict(self):
        return self.artist_dict


class DataClass:
    def __init__(self, item_type):
        self.counts = {'pos':4, 'neut':3, 'neg':3} # Number of ratings per grouping
        self.item_type = item_type
        if item_type == 'movie':
            self.data = MovieData()
        elif item_type == 'music':
            self.data = MusicData()
            self.artist_dict = self.data.get_artist_dict()


    def get_rating_df(self):
        return self.data.get_rating_df()
    
    def get_title_dict(self):
        return self.data.get_title_dict()
    
    def get_gender_dict(self):
        return self.data.get_gender_dict()
            
    def process_rating_df(self, in_df, agg=True):
        """ Output the data for running the ranker"""
        in_df['rating_type'] = np.where((in_df['rating']>=4), 'pos',
                                         np.where(in_df['rating']>=3, 'neut',
                                         'neg'))
        df_wide = in_df.pivot_table(index='user_id',
                                                 columns='rating_type', 
                                                 values='item_id', 
                                                 aggfunc='count').reset_index()
        # Make sure all users in the dataset have enough positive, neutral and negative reviews
        ids = df_wide[(df_wide.pos>=(self.counts['pos'] + NUM_SHOT)) & 
                      (df_wide.neut >= self.counts['neut']) & 
                      (df_wide.neg >= self.counts['neg'])].user_id.unique()
        # Limit to users who satisfy the counts condition
        df = in_df[in_df.user_id.isin(ids)]
        if agg:
            return df[['user_id','rating_type','item_id']].groupby(['user_id','rating_type']).agg({'item_id': lambda x: x.tolist()}).reset_index()
        else:
            return df
        
    def get_prompt(self):
        if self.item_type == 'movie':
            return """
                Hi, I've watched and enjoyed the following movies:
                """
        elif self.item_type == 'music':
            return "Hi, I've listened to and enjoyed the following songs:"
    
    def sample_candidates(self, df):
        """Put together the lists of items and return as a dict"""
        title_dict = self.get_title_dict()
        np.random.seed(123)
        data_dict = {}
        prompt = self.get_prompt()
        # Convert dataframe to dict
        for user_id, row in df.iterrows():
            data_dict[user_id] = dict(pos_ids ={}, neut_ids={}, neg_ids ={}, prompt='')
        
            prompt_ids = random.sample(row.pos, NUM_SHOT)
            neut_ids = random.sample(row.neut, self.counts['neut'])
            neg_ids =  random.sample(row.neg, self.counts['neg'])
            
            pos_ids =random.sample(
                list(set(row.pos) - set(prompt_ids)), 
                self.counts['pos'])

            data_dict[user_id]['prompt_ids'] = prompt_ids
            data_dict[user_id]['pos_ids'] = pos_ids
            data_dict[user_id]['neut_ids'] = neut_ids
            data_dict[user_id]['neg_ids'] = neg_ids
            
            for typ in ('prompt','pos','neut','neg'):
                if self.item_type != 'music':
                    data_dict[user_id][f'{typ}_titles'] = [title_dict[i] for i in data_dict[user_id][f'{typ}_ids']]
                elif self.item_type == 'music':
                    data_dict[user_id][f'{typ}_titles'] = [(title_dict[i] + " by " + self.artist_dict[i]) for i in data_dict[user_id][f'{typ}_ids']]
            
            data_dict[user_id]['prompt'] = prompt + ",".join(data_dict[user_id]['prompt_titles'])
        return data_dict
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--item_type",type=str, default='movie', help='Type of item recommendation')
    args = parser.parse_args()

    data = DataClass(args.item_type)
    data_df = data.get_rating_df()

    df = data.process_rating_df(data_df)
    # Map titles
    title_dict = data.get_title_dict()
    gender_dict = data.get_gender_dict()
    # df['titles'] = df['item_id'].apply(lambda x:[title_dict[i] for i in x])
    df_wide = df.pivot(index='user_id', columns='rating_type', values='item_id')
    data_dict = data.sample_candidates(df_wide)
    # Save the data 
    with open(f'output_data/{args.item_type}/processed_data_dict.json', 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)

    with open(f'output_data/{args.item_type}/title_dict.json', 'w') as json_file:
        json.dump(title_dict, json_file, indent=4)

    with open(f'output_data/{args.item_type}/gender_dict.json', 'w') as json_file:
        json.dump(gender_dict, json_file, indent=4)
    # df_wide.to_csv(f'output_data/{args.item_type}/processed_rating_df.csv', index=False)

