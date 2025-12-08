import pandas as pd
import numpy as np
import re
import json
import torch
import random
import os
from probes import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
# torch.set_printoptions(profile="full")
import re
import pandas as pd
from scipy.sparse.linalg import svds

# pd.set_option('display.max_columns', None)


class MappingDicts:
    def __init__(self, folder):
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


class RecData:
    def __init__(self, foldername, filename='goodreads_samples2.csv'):
        self.map_dicts = MappingDicts(foldername)
        self.df = pd.read_csv(os.path.join(foldername,filename))

    def merged(self, lo_rating=4, num_ratings=3, agg=True):
        merged_df = pd.merge(
            self.df, 
            self.map_dicts.book_id_map, 
            left_on ='book_id',
            right_on='book_id_csv', 
            how='left'
            )
        totals = merged_df.groupby('user_id')['rating'].agg(lambda x: (x >= lo_rating).sum()).reset_index()
        totals.columns = ['user_id','rating_count']
        merged_df2 = pd.merge(
            merged_df, 
            totals[totals.rating_count >= num_ratings], 
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
    
class LLMRecs:
    def __init__(self, foldername, n):
        # Load outputs that ran on GPU
        self.outputs = torch.load(os.path.join(foldername, f'output_dict_{n}.pt'), map_location=torch.device('cpu'))
        self.map_dicts = MappingDicts(foldername)
        self.book_dict_reverse = self.map_dicts.book_dict_reverse
        # Load original recommendation data
        rec_data = RecData(foldername)
        
        result_df = rec_data.merged()
        self.user_title_dict = rec_data.user_title_dict(result_df)
        self.str1 = "system\n\nCutting Knowledge Date: December 2023\nToday Date: 08 Nov 2025\n\nuser\n\nHi, I\'ve read and enjoyed the following books:"
        self.str1b = "Hi, I\'ve read and enjoyed the following books:"
        self.str2 = """  Only return the 5 books you recommend in JSON format like {"Books": {\'title\':..., \'author\':...}}, and nothing else.assistant\n\n"""
        self.str3 = """Please recommend new books based on the user\'s reading preferences and only return the 5 books you recommend in JSON format like {"Books": {\'title\':..., \'author\':...}}, and nothing else.assistant"""
        with open(os.path.join(foldername, f'gender_dict_{n}.json'), 'r') as f:
            self.gender_dict = json.load(f)

    def extract_all_titles(self, text):
        raw_titles = re.findall(r'"title"\s*:\s*"([^"]+)"', text)
        cleaned = [t.split(" by ")[0].strip() for t in raw_titles]
        return cleaned
    
    def extract_books(self, entry):
        # Case 1: structured JSON-like string
        if isinstance(entry, dict) and "Books" in entry:
            return entry["Books"]
        
        if isinstance(entry, str) and entry.strip().startswith("{"):
            try:
                data = json.loads(entry)
                if "Books" in data:
                    return data["Books"]
            except json.JSONDecodeError:
                pass  # fall through to regex

        # Case 2: free-form text like "Title (Series, #N) by Author"
        pattern = r"([A-Z][^()]*?(?:\([^)]*\))?\s+by\s+[A-Z][^,\.]+)"
        matches = re.findall(pattern, entry)

        books = []
        for m in matches:
            # Split into title and author
            title_part, author = m.split(" by ", 1)
            # try to merge book id in
            if title_part.strip() in self.book_dict_reverse:
                book_id = self.book_dict_reverse[title_part.strip()]
            else:
                book_id = None
            books.append({"title": title_part.strip(), "author": author.strip(), "book_id": book_id})
        
        return books
    
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
    

    def map_titles(self, df):
        """ Tries to parse LLM output text  and map to titles in existing book dict"""
        # for k, values in self.outputs.items():
        for typ in ('baseline', 'steered'):
            df[f'{typ}_recs'] = df[f'{typ}_text'].str.split(self.str3).str[1].str.replace("\n","")
            df[f'{typ}_titles'] = df[f'{typ}_recs'].apply(self.extract_all_titles)
            df[f'{typ}_ids'] = df[f'{typ}_titles'].apply(
                lambda x: [
                    self.book_dict_reverse[item] for item in x if item in self.book_dict_reverse
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
        
        rec_data = RecData('goodreads_data')
        df = rec_data.merged()
        mdf = rec_data.merged(agg=False)

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
    
    def lookup_mf_scores(self, small_df, preds_df, user_ids_fromdf):
        # Append MF title scores to each users baseline and steered titles
        user_id_map_df = pd.DataFrame(user_ids_fromdf, columns=['user_id']).reset_index()
        user_id_map_df = user_id_map_df.rename(columns={'index':'user_id_from_df'})
        small_df2 = pd.merge(small_df, user_id_map_df, on='user_id', how='inner')
        if small_df2.shape[0] < small_df.shape[0]:
            print('Losing some users', small_df.shape, small_df2.shape)

        for typ in ('baseline','steered'):
            small_df2[f'mf_{typ}_score_list'] = small_df2.apply(
                lambda x: [preds_df.iloc[x.user_id_from_df].get(i,np.nan) for i in x[f'{typ}_titles']], 
                axis=1
                )
            small_df2[f'mf_{typ}_score'] = small_df2[f'mf_{typ}_score_list'].apply(np.nansum)
            small_df2[f'mf_{typ}_denom'] = small_df2[f'mf_{typ}_score_list'].apply(lambda x: np.count_nonzero(~np.isnan(x)))
        return small_df2

        
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
    def __init__(self, df, foldername, load, num_users_baseline=10):
        self.map_dicts = MappingDicts(foldername)
        book_count = df.groupby('book_id_y').count().reset_index()
        books = book_count[book_count.user_id>=num_users_baseline]['book_id_y']
        int_df = df[df.book_id_y.isin(books)]
        int_df['title'] = int_df['book_id_y'].astype(str).map(self.map_dicts.title_map)
        self.int_df = int_df

        self.user_ids_fromdf = self.int_df[self.int_df.is_read == 1].user_id.unique()
        self.load = load
        self.save = False
        if load:
            try:
                self.preds_df, self.user_ids_fromdf = self.load_pickles()
                print('Loaded MF data from folder')
                
            except FileNotFoundError:
                print(f'The MF filepath was not found')
                self.load=False
                self.save = True
            

    def load_pickles(self):
        """ Load the MF data that was saved"""
        with open('probe_pickles/MF_df.pkl', 'rb') as file:
            preds_df = pickle.load(file)
        with open('probe_pickles/user_ids_map.pkl', 'rb') as file:
            user_ids_fromdf = pickle.load(file)
        return preds_df, user_ids_fromdf
        

    def do_mf(self):
        if self.load:
            return self.preds_df
        else:
            R_df = (
                self.int_df[self.int_df.is_read == 1]
                .pivot_table(
                    index='user_id',
                    columns='title',
                    values='rating',
                    aggfunc='mean',
                    fill_value=0
                )
            )

            R = R_df.to_numpy()

            user_ratings_mean = np.mean(R, axis = 1)
            R_demeaned = R - user_ratings_mean.reshape(-1, 1)

            U, sigma, Vt = svds(R_demeaned, k = 50)
            sigma = np.diag(sigma)
            all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
            preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)
            preds_df.reset_index(names='user_id', inplace=True)
            if self.save:
                print('Saving MF dataframe to pkl')
                preds_df.to_pickle('probe_pickles/MF_df.pkl')
                with open('probe_pickles/user_ids_map.pkl', 'wb') as f:
                    pickle.dump(self.user_ids_fromdf, f)
            return preds_df
        
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
    
    llm_recs = LLMRecs('goodreads_data', 2500)
    outputs_df = llm_recs.build_full_df()
    # # Print summary statistics
    llm_recs.author_gender_count(outputs_df)
    llm_recs.precision(outputs_df)
    llm_recs.output_mf_scores(outputs_df)
    llm_recs.output_mf_author_gender(outputs_df)
    # rec_data = RecData('goodreads_data')
    # df = rec_data.merged()
    # mdf = rec_data.merged(agg=False)

    # mfrec = MFRec(mdf, 'goodreads_data', True)
    # mfrec.eval_mf_by_author_gender()

