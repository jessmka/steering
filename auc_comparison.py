from get_recs_ranker import *
from get_recs_ranker import MovieData as DataProcessor
from data_classes import MovieData



if __name__ == '__main__':
    get_recs = GetRecs(args.item_type, folder, model_name)
    data_dict = get_recs.get_data_dict()
    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--size_of_sample", type=int, default=10, help="Number of users to include")
    parser.add_argument("-i","--item_type",type=str, help='Type of item recommendation')
    parser.add_argument('-m','--model_name', type=str, help ='Model name')
    args = parser.parse_args()

    np.random.seed(42)

    base_path = '/Users/jessicakahn/Documents/repos/'
    folder = os.path.join(base_path,"Glocal_K/1/")
    movie_data = DataProcessor(folder)

    get_recs = movie_data.get_rating_df(rating_type='all',agg=True)
    data_dict = get_recs.get_data_dict()
    items = list(data_dict.items())
    np.random.seed(42)

    # Randomly sample the desired number of users
    if size_of_sample > len(data_dict):
        size_of_sample = len(data_dict)
    new_dict = dict(random.sample(items, args.size_of_sample))


    gender_dict = get_recs.get_gender_dict(new_dict)
    
    # Might need to do some processing here to take only users with 10+ ratings or something?
    items = list(data_dict.items())
    # Randomly sample the desired number of users
    if args.size_of_sample > len(data_dict):
        size_of_sample = len(data_dict)
    new_dict = dict(random.sample(items, args.size_of_sample))

    embedding_data_dict = get_recs.get_prompts_hidden(new_dict, gender_dict)
    regress_list, results = get_recs.get_regress_list(embedding_data_dict)