import pandas as pd
import numpy as np
import os
import argparse
import random
import json
import re
from collections import defaultdict

NUM_SHOT = 5

class MovieData:
    def __init__(self):
        data_path = '/Users/jessicakahn/Documents/repos/Glocal_K/1'
        header = ['user_id', 'item_id', 'rating', 'timestamp']
        self.all_rating_df = pd.read_csv(os.path.join(data_path,'ml-100k/u.data'), sep='\t', names=header)
        
        user = pd.read_csv(os.path.join(data_path,'ml-100k/u.user'), sep="|", encoding='latin-1', header=None)
        user.columns = ['user_id', 'age', 'gender', 'occupation', 'zip code']

        age_bin_edges = [0, 18, 25, 35, 45, 55, 100]
        age_bin_labels = ['0_17','18_24','25_34','35_44','45_54','55_']
        user['age_group'] = pd.cut(user['age'], bins=age_bin_edges, labels=age_bin_labels, right=False)
        user['age_binary'] = np.where(user['age'] >= 31, 'older', 'younger')
        self.gender_dict = user.set_index('user_id')['gender'].to_dict()
        self.age_dict = user.set_index('user_id')['age_group'].to_dict()
        self.age_binary_dict = user.set_index('user_id')['age_binary'].to_dict()

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
    
    def get_age_dict(self):
        return self.age_dict
    
    def get_age_binary_dict(self):
        return self.age_binary_dict
    
class BookData:
    def __init__(self):
        """ Defines a class of Goodreads data and methods"""
        foldername = '/Users/jessicakahn/Documents/repos/probing_classifiers/goodreads_data'
        filename = 'goodreads_samples2.csv'
        self.book_id_map = pd.read_csv(os.path.join(foldername,'book_id_map.csv'))
        with open('output_data/book/gender_dict_500.json', 'r') as f:
            self.gender_dict = json.load(f)

        gender_df = pd.DataFrame(
                self.gender_dict.items(),
                columns=["user_id", "gender"]
                )
        gender_df["user_id"] = gender_df["user_id"].astype("int64")
        # gender_df.columns = 
        # Load rating df and merge with inferred gender dict 
        df = pd.read_csv(os.path.join(foldername,filename))
        # Merge book id
        self.book_id_map.rename(columns={'book_id':'item_id'}, inplace=True)
        print('book_id_map: ', self.book_id_map.head())
        merged_df = pd.merge(
            df, 
            self.book_id_map, 
            left_on ='book_id',
            right_on='book_id_csv', 
            how='left'
            )
        

        self.rating_df = pd.merge(merged_df, gender_df, on='user_id',how='inner')
        print('COLUMNS: ',self.rating_df.columns )
        self.rating_df['age_group'] = 'no_age'
        self.rating_df['age_binary'] = 0
        self.rating_df['item_id'] = self.rating_df['item_id'].astype('string')
        self.age_dict = self.rating_df.set_index('user_id')['age_group'].to_dict()
        self.age_binary_dict = self.rating_df.set_index('user_id')['age_binary'].to_dict()

        self.user_id_map = pd.read_csv(os.path.join(foldername,'user_id_map.csv'))
        self.book_id_map = pd.read_csv(os.path.join(foldername,'book_id_map.csv'))
        author_gender_df = pd.read_csv(os.path.join(foldername, 'final_dataset.csv'))
        author_gender_df = author_gender_df.set_index('authorid')
        self.author_gender_dict = author_gender_df[['gender']].to_dict('index')
        
        with open(os.path.join(foldername,'author_data.json'), 'r') as f:
            self.author_dict = json.load(f)
        with open(os.path.join(foldername,'book_data.json'), 'r') as f:
            self.book_dict = json.load(f)
        
        self.title_dict = {k: v['title_without_series'] for k, v in self.book_dict.items()}
        self.book_dict_reverse = {v['title']: k for k, v in self.book_dict.items()}

    def get_rating_df(self):
        return self.rating_df[['user_id','item_id','rating']]

    def get_title_dict(self):
        return self.title_dict
    
    def get_gender_dict(self):
        return self.gender_dict
    
    def get_author_dict(self):
        return self.author_dict
    
    def get_age_dict(self):
        return self.age_dict
    
    def get_age_binary_dict(self):
        return self.age_binary_dict

    
class MusicData:
    def __init__(self):
        splits = {'train': 'train.gz.parquet', 'valid': 'valid.gz.parquet', 'test': 'test.gz.parquet'}
        df = pd.read_parquet("hf://datasets/matthewfranglen/lastfm-1k/" + splits["train"])
        df = df.dropna(subset=['gender'])
        self.title_dict = df.set_index('track_index')['track_name'].to_dict()
        self.gender_dict = df.set_index('user_index')['gender'].to_dict()
        sub_age_df = df.dropna(subset=['age'])
        self.age_dict = sub_age_df.set_index('user_index')['age'].to_dict()
        age_bin_edges = [0, 18, 25, 35, 45, 55, 100]
        age_bin_labels = ['0_17','18_24','25_34','35_44','45_54','55_']
        sub_age_df['age_group'] = pd.cut(sub_age_df['age'], bins=age_bin_edges, labels=age_bin_labels, right=False)
        sub_age_df['age_binary'] = np.where(sub_age_df['age'] >= 31, 'older', 'younger')
        self.age_binary_dict = sub_age_df.set_index('user_id')['age_binary'].to_dict()
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
    
    def get_age_dict(self):
        return self.age_dict

    def get_age_binary_dict(self):
        return self.age_binary_dict


class DataClass:
    def __init__(self, item_type):
        self.counts = {'pos':5, 'neut':2, 'neg':3} # Number of ratings per grouping
        self.item_type = item_type
        if item_type == 'movie':
            self.data = MovieData()
        elif item_type == 'music':
            self.data = MusicData()
            self.artist_dict = self.data.get_artist_dict()
        elif item_type == 'book':
            self.data = BookData()
            self.author_gender_dict = self.data.author_gender_dict

    def get_reverse_title_dict(self, title_dict):
        """
        Mapping every possible normalized title variant
        to ALL movie_ids that match it.
        """

        new_dict = defaultdict(list)

        for movie_id, full_title in title_dict.items():
            if full_title is None:
                print('NULL TITLE', movie_id)
                break

            variants = set()

            # Original title
            variants.add(full_title)

            # Strip year
            title_no_year = re.sub(r'\s*\(\d{4}\)\s*$', '', full_title)
            variants.add(title_no_year)

            # Handle inverted articles WITH year
            m = re.search(r'^(.*?),\s*(The|A|An)\s*(\(\d{4}\))$', full_title)
            if m:
                main, article, year = m.groups()

                variants.add(f"{article} {main} {year}")
                variants.add(f"{main} {year}")
                variants.add(f"{main}, {article}")
                variants.add(f"{main}")
                variants.add(f"{article} {main}")

            for v in variants:
                new_dict[v].append(int(movie_id))

        return dict(new_dict)

    def get_counts(self):
        return self.counts
            
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
        
        elif self.item_type == 'book':
            return "Hi, I've read and enjoyed the following books:"
    
    def sample_candidates(self, df):
        """Put together the lists of items and return as a dict"""
        title_dict = self.data.get_title_dict()
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
    counts = data.get_counts()
    counts_str = "".join([str(v) for k,v in counts.items()])
    data_df = data.data.get_rating_df()

    df = data.process_rating_df(data_df)

    # Map titles
    if args.item_type == 'music':
        artist_dict = data.data.get_artist_dict()
        with open(f'output_data/{args.item_type}/artist_dict.json', 'w') as json_file:
            json.dump(artist_dict, json_file, indent=4)
    title_dict = data.data.get_title_dict()
    gender_dict = data.data.get_gender_dict()
    age_dict = data.data.get_age_dict()
    age_binary_dict = data.data.get_age_binary_dict()
    reverse_title_dict = data.get_reverse_title_dict(title_dict)
    print('RATINGDFHEAD: ', df.head())
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

    with open(f'output_data/{args.item_type}/age_dict.json', 'w') as json_file:
        json.dump(age_dict, json_file, indent=4)

    with open(f'output_data/{args.item_type}/age_binary_dict.json', 'w') as json_file:
        json.dump(age_binary_dict, json_file, indent=4)
    
    with open(f'output_data/{args.item_type}/reverse_title_dict.json', 'w') as json_file:
        json.dump(reverse_title_dict, json_file, indent=4)

    
    df_wide.to_csv(f'output_data/{args.item_type}/processed_rating_df.csv', index=False)
    data_df.to_csv(f'output_data/{args.item_type}/orig_rating_df.csv', index=False)

