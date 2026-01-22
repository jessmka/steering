import pandas as pd
import numpy as np
import os

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
        self.movie_title_dict = item['movie_title'].to_dict() 

    def get_title_dict(self):
        return self.movie_title_dict
    
    def get_gender_dict(self, new_dict):
        return self.gender_dict
    
    @staticmethod
    def process_rating_df(self, in_df, counts={'pos':4, 'neut':3, 'neg':3}, agg=True):
        """ Output the data for running the ranker"""
        in_df['rating_type'] = np.where((in_df['rating']>=4), 'pos',
                                         np.where(in_df['rating']>=3, 'neut',
                                         'neg'))
        df_wide = in_df.pivot_table(index='user_id',
                                                 columns='rating_type', 
                                                 values='item_id', 
                                                 aggfunc='count').reset_index()
        # Make sure all users in the dataset have enough positive, neutral and negative reviews
        ids = df_wide[(df_wide.pos>=(counts['pos'] + NUM_SHOT)) & (df_wide.neut >= counts['neut']) & (df_wide.neg >= counts['neg'])].user_id.unique()
        df = in_df[in_df.user_id.isin(ids)]
        if agg:
            return df[['user_id','rating_type','item_id']].groupby(['user_id','rating_type']).agg({'item_id': lambda x: x.tolist()}).reset_index()
        else:
            return df

