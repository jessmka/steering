from lenskit.als import BiasedMFScorer
# from lenskit.sklearn.svd import BiasedSVDScorer
# from lenskit.funksvd import FunkSVDScorer
from lenskit.batch import recommend
# from lenskit.data import ItemListCollection, UserIDKey, load_movielens
# from lenskit.knn import ItemKNNScorer
# from lenskit.metrics import NDCG, RBP, RecipRank, Precision, Recall, RMSE, RunAnalysis
from lenskit.pipeline import topn_pipeline
# from lenskit.splitting import SampleFrac, crossfold_users
import lenskit
from get_recs_ranker import *
from output_calcs import MappingDicts, BookRecData


# data_path = '~/Documents/repos/Glocal_K/1'
class LoadBook:
    def __init__(self, data_path):
        book_rec_data = BookRecData(data_path)
        self.df = book_rec_data.get_df()
        # self.df2 = df[['user_id','book_id','rating']]
        # self.df2.columns = ['user_id','item_id','rating']
        

    def get_df(self):
        return self.df

class LoadMovie:
    def __init__(self, data_path):
        header = ['user_id', 'item_id', 'rating', 'timestamp']
        rec_data = pd.read_csv(os.path.join(data_path,'ml-100k/u.data'), sep='\t', names=header)
        data0 = rec_data[['user_id','item_id','rating']]
    
    def get_df(self):
        return self.data0



class RunMF:
    def __init__(self, df, item_type, save):
        self.item_type = item_type
        if item_type == 'book':
            map_folder = '/Users/jessicakahn/Documents/repos/probing_classifiers/goodreads_data/'
        elif item_type == 'movie':
            map_folder = '~/Documents/repos/Glocal_K/1/'
        
        maps = MappingDicts(map_folder, item_type)
        if item_type == 'book':
            merged_df = pd.merge(
            df, 
            maps.book_id_map, 
            left_on ='book_id',
            right_on='book_id_csv', 
            how='left'
            )
            merged_df = merged_df.rename({'book_id_y':'item_id'})
        self.title_map = maps.get_title_map()
        data = lenskit.data.from_interactions_df(df, user_col='user_id',item_col='item_id', rating_col='rating')

        model_als = BiasedMFScorer(features=50)
        self.pipe_als = topn_pipeline(model_als)
        self.pipe_als.train(data)
        self.all_users = data.users.ids()
        if save:
            with open(os.path.join(f'output_data/{item_type}/probes', f'rec_mf.pkl'),'wb') as f:
                pickle.dump(self.pipe_als,f)

    def get_candidates(self, num_cands = 10, save=True):
        """Creates a list of candidates for training"""
        full_recommendations = recommend(self.pipe_als, self.all_users, num_cands) # n is already set in pipeline, but can be overridden
        # Create candidates
        candidate_dict = {}
        for user_key, item_list in full_recommendations.items():
            if self.item_type == 'movie':
                candidate_dict[int(user_key.user_id)] = [self.title_map[str(i)] for i in item_list.ids()]
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
        book_data = LoadBook('/Users/jessicakahn/Documents/repos/probing_classifiers/goodreads_data/')
        df = book_data.get_df()
        # print(df.head())
    elif args.item_type == 'movie':
        movie_data = LoadMovie('~/Documents/repos/Glocal_K/1/')
        df = movie_data.get_df()
    print(df.head())
    mf = RunMF(df, args.item_type, save=True)
    mf.get_candidates()
    


    