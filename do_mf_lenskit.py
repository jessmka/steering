# from lenskit.als import BiasedMFScorer
# from lenskit.sklearn.svd import BiasedSVDScorer
# from lenskit.funksvd import FunkSVDScorer
# from lenskit.batch import recommend
# from lenskit.data import ItemListCollection, UserIDKey, load_movielens
# from lenskit.knn import ItemKNNScorer
# from lenskit.metrics import NDCG, RBP, RecipRank, Precision, Recall, RMSE, RunAnalysis
# from lenskit.pipeline import topn_pipeline
# from lenskit.splitting import SampleFrac, crossfold_users
# import lenskit
import implicit
import scipy.sparse as sp

from get_recs_ranker import *
from data_classes import BookData, MovieData, BookMappingDicts

# data_path = '~/Documents/repos/Glocal_K/1'
class LoadBook:
    def __init__(self, data_path):
        book_rec_data = BookData(data_path)
        self.df = book_rec_data.get_df()
        # self.df2 = df[['user_id','book_id','rating']]
        # self.df2.columns = ['user_id','item_id','rating']
        

    def get_df(self):
        return self.df

# class LoadMovie:
#     def __init__(self, data_path):
#         movie_rec_data = MovieData(data_path)
#         rec_data = movie_rec_data.get_all_rating_df()[['user_id','item_id','rating']]
    
#     def get_df(self):
#         return self.data0



class RunMF:
    def __init__(self, data, item_type, use_lk, save):
        self.item_type = item_type
        # if item_type == 'book':
        #     map_folder = '/Users/jessicakahn/Documents/repos/probing_classifiers/goodreads_data/'
        # elif item_type == 'movie':
        #     map_folder = '~/Documents/repos/Glocal_K/1/'
        
        
        if item_type == 'book':
            maps = BookMappingDicts()
            df = data.merged(agg=False)
        if self.item_type == 'movie':
            self.title_map = data.get_title_dict()
            df = data.get_all_rating_df()

        # Replace below lenskit with implicit package instead
        users = df.user_id.unique()
        items = df.item_id.unique()
        u2i = {u:i for i,u in enumerate(users)}
        i2i = {v:i for i,v in enumerate(items)}
        i2u = {i:u for u,i in u2i.items()}
        i2v = {i:v for v,i in i2i.items()}
        rows = df.user_id.map(u2i)
        cols = df.item_id.map(i2i)
        ratings_mat = sp.csr_matrix((df.rating, (rows, cols)))
        # Train ALS model
        model = implicit.als.AlternatingLeastSquares(
            factors=50, 
            regularization=0.1, 
            iterations=20
        )

        # Since implicit expects "confidence" instead of ratings, you can just pass ratings as confidence
        model.fit(ratings_mat)

        # Save model + mappings
        if save:
            with open(f'output_data/{item_type}/probes/als_explicit.pkl','wb') as f:
                pickle.dump((model, u2i, i2i, i2u, i2v), f)
        if use_lk:
            int_data = lenskit.data.from_interactions_df(df, user_col='user_id',item_col='item_id', rating_col='rating')

            model_als = BiasedMFScorer(features=50)
            self.pipe_als = topn_pipeline(model_als,name='rating-predictor')
            self.pipe_als.train(int_data)
            self.all_users = int_data.users.ids()
            if save:
                with open(os.path.join(f'output_data/{item_type}/probes', f'rec_mf.pkl'),'wb') as f:
                    pickle.dump(self.pipe_als,f)

    def recommend(self, user_idx, n=5):
        """This function needs to be a wrapper so that I can get Top N"""
        top_items = self.model.recommend(user_idx, sp.csr_matrix([[0]]), N=n, filter_already_liked_items=False)
        return top_items

    def get_pipeline(self):
        return self.pipe_als

    def get_candidates(self, num_cands = 10, save=True):
        """Creates a list of candidates for training"""
        full_recommendations = self.recommend(self.pipe_als, self.all_users, num_cands) # n is already set in pipeline, but can be overridden
        # Create candidates
        candidate_dict = {}
        for user_key, item_list in full_recommendations.items():
            if self.item_type == 'movie':
                candidate_dict[int(user_key.user_id)] = [self.title_map[(i)] for i in item_list.ids()]
            elif self.item_type == 'book':
                candidate_dict[int(user_key.user_id)] = [self.title_map[str(i)]['title'] for i in item_list.ids()]
        if save:
            print('Saving candidates to : ',f'"output_data/{self.item_type}/candidate_dict_{self.item_type}_n{num_cands}.json')
            with open(f"output_data/{self.item_type}/candidate_dict_{self.item_type}_n{num_cands}.json", "w") as json_file:
                json.dump(candidate_dict, json_file, indent=4)
        else:
            return candidate_dict



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--item_type",type=str, help='Type of item recommendation')
    args = parser.parse_args()
    if args.item_type == 'book':
        data = BookData()
        # df = data.get_df()
        # print(df.head())
    elif args.item_type == 'movie':
        data = MovieData()
        # df = data.get_all_rating_df()
    # print(df.head())
    mf = RunMF(data, args.item_type, save=True)
    # mf.get_candidates(save=False)
    


    