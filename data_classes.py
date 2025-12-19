import pandas as pd
import os
import json

POS_RATING = 4
NUM_RATINGS = 10

class BookMappingDicts:
    def __init__(self):
        folder = '/Users/jessicakahn/Documents/repos/probing_classifiers/goodreads_data'
        self.user_id_map = pd.read_csv(os.path.join(folder,'user_id_map.csv'))
        self.book_id_map = pd.read_csv(os.path.join(folder,'book_id_map.csv'))
        author_gender_df = pd.read_csv(os.path.join(folder, 'final_dataset.csv'))
        author_gender_df = author_gender_df.set_index('authorid')
        self.author_gender = author_gender_df[['gender']].to_dict('index')
        
        with open(os.path.join(folder,'author_data.json'), 'r') as f:
            self.author_dict = json.load(f)
        with open(os.path.join(folder,'book_data.json'), 'r') as f:
            self.book_dict = json.load(f)
        
        self.title_map = {k: v['title_without_series'] for k, v in self.book_dict.items()}
        self.book_dict_reverse = {v['title']: k for k, v in self.book_dict.items()}

    def get_title_map(self):
        return self.title_map
    
    def get_author_dict(self):
        return self.author_dict
    
    def get_book_dict(self):
        return self.book_dict

    def get_reverse_book_dict(self):
        return self.book_dict_reverse
    
    def get_author_gender(self):
        return self.author_gender
    
class BookData:
    """ Defines a class of Goodreads data and methods"""
    def __init__(self, filename='goodreads_samples2.csv'):
        foldername = '/Users/jessicakahn/Documents/repos/probing_classifiers/goodreads_data'
        self.map_dicts = BookMappingDicts()
        self.df = pd.read_csv(os.path.join(foldername,filename))

    def get_map_dicts(self):
        return self.map_dicts

    def merged(self, agg=True):
        merged_df = pd.merge(
            self.df, 
            self.map_dicts.book_id_map, 
            left_on ='book_id',
            right_on='book_id_csv', 
            how='left'
            )
        totals = merged_df.groupby('user_id')['rating'].agg(lambda x: (x >= POS_RATING).sum()).reset_index()
        totals.columns = ['user_id','rating_count']
        merged_df2 = pd.merge(
            merged_df, 
            totals[totals.rating_count >= NUM_RATINGS], 
            on ='user_id', 
            how='inner'
            )
        if agg:
            return merged_df2.groupby('user_id')['book_id_y'].agg(list).reset_index()
        else:
            return merged_df2
    
    
    def user_title_dict(self, result_df):
        # Returns a mapping dict with the users total historical titles read and liked 
        user_title_dict = {}
        for i, row in result_df.iterrows():
            user_title_dict[row['user_id']] = [self.map_dicts.book_dict[str(i)]['title_without_series'] for i in row['book_id_y']]
        return user_title_dict
    

class MovieData:
    """ Defines a class of MovieLens data with methods"""
    def __init__(self):
        data_path = '/Users/jessicakahn/Documents/repos/Glocal_K/1'
        header = ['user_id', 'item_id', 'rating', 'timestamp']
        self.all_rating_df = pd.read_csv(os.path.join(data_path,'ml-100k/u.data'), sep='\t', names=header)
        user = pd.read_csv(os.path.join(data_path,'ml-100k/u.user'), sep="|", encoding='latin-1', header=None)
        user.columns = ['user_id', 'age', 'gender', 'occupation', 'zip code']
        # Get negatively rated items
        self.rating_df_neg = all_rating_df[all_rating_df.rating <3]
        neg_totals = all_rating_df.groupby('user_id')['rating'].agg(lambda x: (x < 3).sum()).reset_index()
        neg_totals.columns = ['user_id','rating_count']
        self.gender_dict = user.set_index('user_id')['gender'].to_dict()
        rating_df_pos = all_rating_df[all_rating_df.rating>=POS_RATING]
        # Need to filter on users with more than NUM_RATINGS pos ratings
        rating_df_pos['rating_count'] = rating_df_pos.groupby('user_id')['item_id'].transform('count')
        rating_df_pos = rating_df_pos[rating_df_pos.rating_count >= NUM_RATINGS][['user_id','item_id']]
        # Filter on users with at least 2 neg ratings
        rating_df_pos = pd.merge(rating_df_pos, neg_totals[neg_totals.rating_count>=2], on ='user_id', how='inner')
        self.rating_df = (rating_df_pos.groupby(['user_id'])
        .agg({'item_id': lambda x: x.tolist()})
        .reset_index())
        
        item = pd.read_csv(os.path.join(data_path,'ml-100k/u.item'), sep="|", encoding='latin-1', header=None)
        item.columns = ['movie_id', 'movie_title' ,'release_date','video_release_date', 'IMDb_URL', 'unknown', 'Action', 
                'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 
                'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        item = item.set_index('movie_id')
        self.movie_title_dict = item['movie_title'].to_dict() 

    def get_neg_items(self):
        self.rating_df_neg['item'] = self.rating_df_neg['item_id'].map(self.movie_title_dict)
        neg_list_df = self.rating_df_neg.groupby('user_id')['item'].agg(list).reset_index()
        return neg_list_df.set_index('user_id')['item'].to_dict()

    def get_movie_gender_dict(self, new_dict):
        return self.gender_dict
    
    def get_rating_df(self):
        return self.rating_df
    
    def get_all_rating_df(self):
        return self.all_rating_df
    
    def get_title_dict(self):
        return self.movie_title_dict
    
    def get_reverse_title_dict(self):
        return {value: key for key, value in self.movie_title_dict.items()}