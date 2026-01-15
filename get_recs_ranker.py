import pandas as pd
import numpy as np
import argparse
import re
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
# torch.set_printoptions(profile="full")
import re
import os
import collections
# import gc
from datetime import datetime

# pd.set_option('display.max_columns', None)

POS_RATING = 4
NUM_RATINGS = 5


class MovieData:
    def __init__(self, data_path):
        # Load data
        header = ['user_id', 'item_id', 'rating', 'timestamp']
        self.rating_df = pd.read_csv(os.path.join(data_path,'ml-100k/u.data'), sep='\t', names=header)
        user = pd.read_csv(os.path.join(data_path,'ml-100k/u.user'), sep="|", encoding='latin-1', header=None)
        user.columns = ['user_id', 'age', 'gender', 'occupation', 'zip code']
        self.gender_dict = user.set_index('user_id')['gender'].to_dict()
        self.movie_title_dict = self.item_data_dict(data_path)

    @staticmethod
    def process_rating_df(rating_df, rating_type='pos', agg=True, num_pos_ratings=NUM_RATINGS):
    # Need to filter on users with more than NUM_RATINGS pos ratings
        rating_df_neg = rating_df[rating_df.rating <3]
        neg_totals = rating_df.groupby('user_id')['rating'].agg(lambda x: (x < 3).sum()).reset_index()
        neg_totals.columns = ['user_id','rating_count']
        # Only need to calculate the following for pos or all types
        if rating_type in ('pos','all'):
            rating_df_pos = rating_df[rating_df.rating >= POS_RATING]
            rating_df_pos['rating_count'] = rating_df_pos.groupby('user_id')['item_id'].transform('count')
            rating_df_pos = rating_df_pos[rating_df_pos.rating_count >= num_pos_ratings][['user_id','item_id']]
            # Filter again on users that have at least 2 negative ratings
            rating_df_pos = pd.merge(rating_df_pos, 
                                    neg_totals[neg_totals.rating_count>=2], 
                                    on ='user_id',
                                    how='inner')

            # Filter total DF on only users with at least NUM_RATINGS positive ratings
            rating_df = rating_df[rating_df.user_id.isin(rating_df_pos.user_id)]

        if rating_type=='pos':
            df = rating_df_pos
        elif rating_type == 'neg':
            df = rating_df_neg
        elif rating_type == 'all':
            df = rating_df
        if agg:
            # Converts long format to column of list of item IDs
            return df.groupby('user_id').agg({'item_id': lambda x: x.tolist()}).reset_index()
        else:
            return df
        

    def item_data_dict(self, data_path):
        item = pd.read_csv(os.path.join(data_path,'ml-100k/u.item'), sep="|", encoding='latin-1', header=None)
        item.columns = ['movie_id', 'movie_title' ,'release_date','video_release_date', 'IMDb_URL', 'unknown', 'Action', 
                'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 
                'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        item = item.set_index('movie_id')
        return item['movie_title'].to_dict() 


    def get_neg_items(self):
        """Return dict of negatively rated items"""
        rating_df_neg = MovieData.process_rating_df(self.rating_df, rating_type='neg',agg=False)
        rating_df_neg['item'] = rating_df_neg['item_id']#.map(self.movie_title_dict)
        neg_list_df = rating_df_neg.groupby('user_id')['item'].agg(list).reset_index()
        return neg_list_df.set_index('user_id')['item'].to_dict()

    def get_movie_gender_dict(self, new_dict):
        return self.gender_dict
    
    # TODO: Make  a staticmethod so it can be used by other data classes for manipulating
    # the item DF
    def get_rating_df(self, rating_type='pos', agg=True):
        return self.process_rating_df(self.rating_df, rating_type, agg)
        
    
    def get_title_dict(self):
        return self.movie_title_dict


class BookData:
    """ 
    Inputs are:
    goodreads_samples2.csv, user_id_map.csv, book_id_map.csv
    book_data.json, author_data.json
    """
    def __init__(self, folder):
        df = pd.read_csv(os.path.join(folder, "goodreads_samples2.csv"))
        user_id_map = pd.read_csv(os.path.join(folder, "user_id_map.csv"))
        book_id_map = pd.read_csv(os.path.join(folder, "book_id_map.csv"))
        with open(os.path.join(folder,"book_data.json"), 'r') as f:
            self.book_dict = json.load(f)
        with open(os.path.join(folder,"author_data.json"), 'r') as f:
            self.author_dict = json.load(f)
        merged_df = pd.merge(df, book_id_map, left_on ='book_id',right_on='book_id_csv', how='left')
        merged_df['lang'] = merged_df['book_id_y'].apply(lambda x: self.book_dict[str(x)]['language_code'])

        
        self.merged_df1 = merged_df[merged_df['lang'].isin(['eng','en-US','en-GB','en-CA','en'])]
        # Count negative ratings by user
        neg_totals = self.merged_df1.groupby('user_id')['rating'].agg(lambda x: (x < 3).sum()).reset_index()
        neg_totals.columns = ['user_id','rating_count']
        # Count positive ratings by user
        totals = self.merged_df1.groupby('user_id')['rating'].agg(lambda x: (x >= POS_RATING).sum()).reset_index()

        totals.columns = ['user_id','rating_count']
        merged_df2 = pd.merge(self.merged_df1, totals[totals.rating_count >= 10], on ='user_id', how='inner')
        merged_df3 = pd.merge(merged_df2, neg_totals[neg_totals.rating_count>=2], on ='user_id', how='inner')
        self.rating_df = merged_df3[['user_id','book_id_y','rating']].rename(columns={'book_id_y':'item_id'})
        # self.rating_df = merged_df3.groupby('user_id')['book_id_y'].agg(list).reset_index()

    def get_neg_items(self):
        """ Return dict of negative items per user if they exist"""
        # Make a neg df then create lists from the rows:
        neg_df = MovieData.process_rating_df(self.rating_df, rating_type='neg', agg=False)
        neg_df2 = neg_df.groupby('user_id')['item_id'].agg(list).reset_index()
        neg_df_list = neg_df2.groupby('user_id')['item_id'].agg(list).reset_index()
        neg_dict = neg_df_list.set_index('user_id')['item_id'].to_dict()
        return neg_dict
        
    def get_book_dict(self):
        return self.book_dict
    
    def get_rating_df(self, rating_type='pos', agg=True):
        return  MovieData.process_rating_df(self.rating_df,rating_type, agg)
    
    def get_author_dict(self):
        return self.author_dict

            
## End of data classes

class GetRecs:
    def __init__(self, item_type, data_path, model_name='meta-llama/Llama-3.2-3B-Instruct'):
        self.item_type = item_type
        steering_data_path = os.path.join(data_path,'steering/')
        if item_type == 'movie':
            self.data = MovieData(data_path)
            self.movie_title_dict = self.data.get_title_dict()
            
            
        elif item_type == 'book':
            # Need to change the input to be the location where the folder is
            self.data = BookData(os.path.join(data_path, 'probing_classifiers/goodreads_data/'))
            self.book_dict = self.data.get_book_dict()
            self.author_dict = self.data.get_author_dict()
        
        self.neg_dict = self.data.get_neg_items()
        # TODO: Need to add args here. Currently is only pulling in positive
        self.rating_df = self.data.get_rating_df()
        # self.gender_dict = data.get_gender_dict(data_dict)
        load_model = LoadModel(model_name)
        self.model = load_model.get_model()
        self.max_layer = load_model.get_layers()
        self.tokenizer = load_model.get_tokenizer()
        self.data_path = steering_data_path

    def get_title_dict(self):
        if self.item_type == 'movie':
            return self.movie_title_dict
        elif self.item_type == 'book':
            return self.book_dict


    def get_device(self):
        return self.model.device

    def get_book_gender_dict(self, new_dict):
        gender_dict = {}
        max_new_tokens = 10
        size_of_sample = len(new_dict)
        for k,v in new_dict.items():
            strings = [[{
                "role": "user", 
                "content": v['prompt'] + "Based on these, can you please guess what my gender is? Please respond with only a single word as your answer."
            }]]  # single conversation
            inputs = self.tokenizer.apply_chat_template(
                strings[0],
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True
            ).to(self.model.device)
        
            inputs_dict = {"input_ids": inputs}
            with torch.inference_mode(), torch.autocast("cuda"):
                    output = self.model.generate(
                    **inputs_dict,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                    # temperature=0.7,
                )
            
            gender_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            # print(gender_text)
            # response_only = gender_text[len(v):].strip()
            split_text = re.sub(r'[^a-zA-Z]', '', gender_text.split("assistant")[1])
            gender_dict[k] = split_text

        for k,v in gender_dict.items():
            if v not in ('Male', 'Female'):
                gender_dict[k] = 'Unknown'
        print('Gender dict: ', collections.Counter(gender_dict.values()))
        with open(os.path.join(self.data_path,f"output_data/book/gender_dict_{size_of_sample}.json"), "w") as json_file:
            json.dump(gender_dict, json_file, indent=4)
        return gender_dict


    def get_gender_dict(self, new_dict):
        if self.item_type == 'book':
            return self.get_book_gender_dict(new_dict)
        elif self.item_type == 'movie':
            return self.data.get_movie_gender_dict(new_dict)


    
    def get_prompt(self):
        if self.item_type == 'movie':
            prompt = f"""
            Hi, I've watched and enjoyed the following movies:
            """
        elif self.item_type == 'book':
            prompt = f"""
            Hi, I've read and enjoyed the following books: 
            """
        return prompt

    
    def get_data_dict(self, num_sam=5):
        """Returns a dict with prompts and items"""
        np.random.seed(123)
        prompt = self.get_prompt()
        data_dict = {}
        
        # print('Rating DF: ', self.rating_df.head())
        if self.item_type == 'movie':
            for i, row in self.rating_df.iterrows():
                # print('Row item length: ',len(row.item_id))
                # TODO: Does rating_df only include pos items here? Yes. But need list
                #  of unsampled items too
                data_dict[row.user_id] = dict(cans_items =[], val_items=[], neg_items =[], prompt='')
                ids = random.sample(row.item_id, num_sam)
                data_dict[row.user_id]['cans_items'] = ids
                data_dict[row.user_id]['val_items'] = list(set(row.item_id) - set(ids))
                title_list = [self.movie_title_dict[i] for i in ids]
                data_dict[row.user_id]['prompt'] = prompt + ",".join(title_list)
        elif self.item_type == 'book':
            for i, row in self.rating_df.iterrows():
            # sample ids from user list
                data_dict[row.user_id] = dict(cans_items =[], val_items=[], neg_items=[], prompt='')
                ids = random.sample(row.item_id, num_sam)
                data_dict[row.user_id]['cans_items'] = ids
                data_dict[row.user_id]['val_items'] = list(set(row.item_id) - set(ids))
                # other_seen_rated_titles = [self.book_dict[str(i)]['title'] for i in row.book_id_y if i not in ids]
                # other_seen_liked[row.user_id] = other_seen_rated_titles
                book_list = []
                for v in ids:
                    title = self.book_dict[str(v)]['title']
                    if len(self.book_dict[str(v)]['authors']) > 0:
                        author_id = self.book_dict[str(v)]['authors'][0]['author_id']
                        author_name = self.author_dict[author_id]
                        book_list.append(f"{title} by {author_name}")
                    else:
                        book_list.append(title)
                    data_dict[row.user_id]['prompt'] = prompt + ",".join(book_list)

        for k, v in data_dict.items():
            data_dict[k]['neg_items'] = self.neg_dict.get(k)

        return data_dict

    def get_prompts_hidden(self, data_dict, gender_dict):
        embedding_data_dict = {}
        # Create keys with empty lists for each layer in hidden_states
        for j in range(self.max_layer+1):
            embedding_data_dict[j] = []
        # Loop through and add demo + hidden_state to data_dict
        for k,v in data_dict.items():
            inputs = self.tokenizer(v['prompt'], return_tensors="pt", padding=True).to(self.model.device)
            # Run forward pass and request hidden states
            with torch.inference_mode():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    output_attentions=False,
                    return_dict=True,
                )
            
            # Extract hidden states
            hidden_states = outputs.hidden_states
        
            # Lookup gender
            gender = gender_dict[k]
            for idx, repr in enumerate(hidden_states):
                if idx >= (self.max_layer-5):
                    hidden_repr = repr.mean(dim=1).squeeze(0).squeeze(0).detach().cpu()
                else:
                    hidden_repr = None
                embedding_data_dict[idx].append(dict(demo=gender, hidden=hidden_repr))
        del hidden_states
        del outputs
        torch.cuda.empty_cache()
        # gc.collect()

        return embedding_data_dict

    def get_regress_list(self, embedding_data_dict):
        regress_list = []
        results = []
        # i = 0
        for key_layer, value in embedding_data_dict.items():
            clf = LogisticRegression(max_iter=1000, solver='lbfgs')

            if key_layer >=  (self.max_layer-5):
                # for j in value:
                X = [j['hidden'].detach().cpu() for j in value if j['demo']!='Unknown']
                # print('X:  ',X)
                X_tensor = torch.stack([
                    x.to(torch.float32)          # convert each element
                    .detach()
                    .cpu()
                    for x in X
                ])

                X_np = X_tensor.numpy()

                y = [j['demo'] for j in value if j['demo']!='Unknown']
                # clf = LogisticRegression(multi_class='multinomial',solver='newton-cg')
                
                clf = clf.fit(X_np, y)
            
                scores = cross_val_score(clf, X_np, y, cv=5, scoring='roc_auc')
                results.append(np.array(scores).mean())
                regress_list.append(clf)
            else:
                regress_list.append(clf)

        return regress_list, results



    def steer_prompt_compare(
            self,
            prompt: str,
            alpha: float,
            layer_to_steer: int,
            max_new_tokens: int,
            probe_list: list,
            item_type: str,
            candidates: str, 
            W_probe_T: any,
            pinv_W_T: any
    ):
        """
        Runs baseline and steered generations for a single prompt.
        Captures pre- and post-steering hidden activations from a given layer.
        """
        candidate_str = ",".join(candidates)
        # print(candidate_str)
        # === Prepare model input ===
        if item_type == 'book':
            verb = 'read'
        elif item_type == 'movie':
            verb = 'watch'
        request_str = f"Please rank the following {item_type}s in order from most to least likely to recommend to them to {verb} next. Only give the {item_type} ranking with no other content or explanation."
        # print('TEXT: ', prompt, request_str, candidates)
        chat = [[{"role": "user", "content": prompt + request_str + candidate_str}]]
        inputs = self.tokenizer.apply_chat_template(
            chat[0],
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)
        inputs_dict = {"input_ids": inputs}

        # === Run BASELINE (no steering) ===
        with torch.inference_mode(), torch.autocast("cuda"):
            baseline_out = self.model.generate(
                **inputs_dict,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=False
                # temperature=0.7,
            )
        baseline_text = self.tokenizer.decode(baseline_out[0], skip_special_tokens=True)
        # print("=== BASELINE OUTPUT ===\n", baseline_text, "\n")
        del baseline_out
        # torch.cuda.empty_cache()
        # gc.collect()
        # === Set up capture and steering for steered run ===
        # capture = {"pre": None, "post": None}
        first_pass_done = False

        # Determine expected sequence length (avoid capturing inputs inside closures)
        seq_len_expected = inputs.shape[1]
        
        # first_pass_done = False

        # --------- Safe pre-capture hook (observes only, stores CPU copy) ----------
        # def get_hidden_state_hook(module, input, output):
        #     # keep this minimal and only store a small CPU copy summary
        #     try:
        #         # Only capture on the initial full-sequence forward
        #         if output.shape[1] == seq_len_expected and not first_pass_done["value"]:
        #             # summarize across sequence dimension to produce a compact vector
        #             # detach -> move to CPU immediately so we don't keep GPU memory
        #             cpu_summary = output.mean(dim=1).squeeze(0).detach().cpu()
        #             capture["pre"] = cpu_summary
        #             # delete local refs to free GPU memory ASAP
        #             del cpu_summary
        #             torch.cuda.empty_cache()
        #     except Exception:
        #         # don't let hook exceptions break generation
        #         pass
        #     # forward hooks should not return unless modifying output; here we only observe
        #     return None

        # --------- Safe steering hook (does computation but returns steered output) ----------
        def steering_hook(module, input, output):
            # This hook will run many times during generation; keep it minimal:
            try:
                if output.shape[1] == seq_len_expected and not first_pass_done["value"]:
                    # operate on GPU tensor 'output' (no copies kept)
                    hidden = output  # alias for clarity

                    # Project hidden into probe subspace: x_proj = hidden @ W.T @ pinv(W).T
                    # We create the projection using a small temporary on GPU, then discard.
                    # W_probe is already on the model device.
                    # W = W_probe  # small matrix, precomputed

                    # Compute projection (keep ops on GPU)
                    x_proj0 = hidden @ W_probe_T
                    # pinv on GPU (temporary); try to keep dtype float32 for numerical stability
                    # pinv_W_T = torch.linalg.pinv(W).T
                    x_proj = x_proj0 @ pinv_W_T

                    # steering vector v (on GPU)
                    v = -x_proj

                    # seq_len = hidden.shape[1]
                    steered = hidden + alpha * v

                    # Store only a CPU copy (detach) to avoid retaining GPU memory
                    # capture["post"] = steered.detach().cpu()

                    # mark done and free temporaries
                    first_pass_done = True
                    # Explicitly delete large GPU temporaries
                    del hidden
                    # , W, x_proj0, pinv_W_T, x_proj, v, weights
                    # torch.cuda.empty_cache()

                    # Return steered GPU tensor so generation continues with modified activations
                    return steered
            except Exception:
                # If anything goes wrong, don't break generation - return unmodified output
                try:
                    return output
                except Exception:
                    return None
            # For non-initial or later passes, return output unchanged
            return output

        # === Register hooks safely on the minimal set of layers needed ===
        # (adjust layer indices to your model architecture if necessary)
        # h0 = self.model.model.layers[layer_to_steer].register_forward_hook(get_hidden_state_hook)
        # h1a = self.model.model.layers[layer_to_steer-4].register_forward_hook(steering_hook)
        # h1  = self.model.model.layers[layer_to_steer-3].register_forward_hook(steering_hook)
        # h2  = self.model.model.layers[layer_to_steer-2].register_forward_hook(steering_hook)
        # h3  = self.model.model.layers[layer_to_steer-1].register_forward_hook(steering_hook)
        h1  = self.model.model.layers[layer_to_steer].register_forward_hook(steering_hook)
        # h2  = self.model.model.layers[layer_to_steer].register_forward_hook(get_hidden_state_hook)

        # === Run STEERED generation (use_cache=False to avoid kv-cache blowup) ===
        with torch.inference_mode():
            steered_out = self.model.generate(
                **inputs_dict,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )

        # === Clean up hooks and GPU temporaries ===
        # for h in [h1]: #h0, h1a,, h5
        try:
            h1.remove()
        except Exception:
            pass

        # free the precomputed W and any other temporaries
        # try:
        #     del W_probe
        # except Exception:
        #     pass
        # torch.cuda.empty_cache()

        steered_text = self.tokenizer.decode(steered_out[0], skip_special_tokens=True)

        del steered_out
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        #     torch.cuda.synchronize()
        # === Decode steered text ===
        # print("=== STEERED OUTPUT ===\n", steered_text, "\n")

        # === Return results ===
        return {
            "baseline_text": baseline_text,
            "steered_text": steered_text,
            # "pre_hidden": capture["pre"],
            # "post_hidden": capture["post"],
        }
    
class LoadModel:
    def __init__(self, model_str="llama3b"):
        # Load the model
        model_dict = {
            'llama3b':'meta-llama/Llama-3.2-3B-Instruct', 
            'tinyllama':'TinyLlama/TinyLlama-1.1B-Chat-v1.0'}
        model_name = model_dict[model_str]
        print('Loading model: ', model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            dtype=torch.bfloat16, 
            device_map="auto"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"

        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
        if model_name == 'meta-llama/Llama-3.2-3B-Instruct':
            self.max_layer = 28
        elif model_name == 'TinyLlama/TinyLlama-1.1B-Chat-v1.0':
            self.max_layer = 22

    def get_layers(self):
        return self.max_layer
    
    def get_model(self):
        return self.model
     
    def get_tokenizer(self):
        return self.tokenizer
    
class Candidates():
    def __init__(self, data_path, can_type='from_file', random_rank=True):
        # ~/Documents/repos/probing_classifiers/cf.ipynb creates the parquet
        self.random_rank = random_rank
        if can_type == 'from_file':
            with open(os.path.join(data_path, 'candidate_dict_n10.json'), 'r') as file:
                self.candidate_dict = json.load(file)

    def get_candidates(self):
        return self.candidate_dict
    
    # def restrict_to_set(self, cans_dict, mydict):
    #     """Restrict candidate dict to only the sample in mydict"""
    #     return {key: cans_dict[str(key)]['id'] for key in mydict.keys()}
    
    def get_negs_and_vals(self, subset_dict, cans_dict):
        new_cans_dict = {}
        for k, v in cans_dict.items():
            new_cans_dict[k] = dict(cans_titles=[], cans_ids=[])
            new_cans = []
            if subset_dict.get(int(k)) is not None: #Make sure user id exists in our new subset of users
                negs = subset_dict.get(int(k)).get('neg_items')
                # If the user has more than 2 negatively rated items just sample 2 of them
                if len(negs) > 2:
                    negs = random.sample(negs, 2)
                vals = subset_dict.get(int(k)).get('val_items')
                # Similarly if the user has more than 5 positively rated items just sample 5 of them
                if len(vals) > 5:
                    vals = random.sample(vals, 5)
                new_cans.extend(negs)
                new_cans.extend(vals)
            # If validation items and negative items is less than 10, fill to 10 with top 10 candidates
            # from the CF model (or other kind of model)
            if len(new_cans) < 10:
                cans_sample = random.sample(v, 10 - len(new_cans))
                new_cans.extend(cans_sample)

            # Shuffle the candidates to avoid position bias
            if self.random_rank:
                random.shuffle(new_cans)
            new_cans_dict[k]['cans_ids'] = new_cans
        # Currently this dict only has IDs - need to map to titles (maybe do it outside of this class)
        # print('NEW CANDIDATES: ', new_cans_dict)
        return new_cans_dict


if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--size_of_sample", type=int, default=10, help="Number of users to include")
    parser.add_argument("-i","--item_type",type=str, help='Type of item recommendation')
    parser.add_argument('-m','--model_name', type=str, help ='Model name')
    # parser.add_argument('')
    args = parser.parse_args()
    size_of_sample = args.size_of_sample

    # Define paths
    if torch.cuda.is_available():
        base_path = '/home/ec2-user/studies/swb-s4149184-personal-study/'
        if args.model_name is None:
            model_name = 'llama3b'
        else: model_name = args.model_name
        if args.item_type == 'book':
            folder = base_path
        if args.item_type == 'movie':
            folder = os.path.join(base_path,"1/")
    else:
        base_path = '/Users/jessicakahn/Documents/repos/'
        if args.model_name is None:
            model_name = 'tinyllama'
        else: model_name = args.model_name
        if args.item_type == 'book':
            folder = base_path
        if args.item_type == 'movie':
            folder = os.path.join(base_path,"Glocal_K/1/")
    
    # print('BASE and MODEL: ', base_path, model_name)
    output_path = os.path.join(base_path,f'steering/output_data/{args.item_type}/')
    print('output path', output_path)

    

    # Check if directories exist
    if os.path.isdir(output_path):
        print(f"The folder '{output_path}' exists.")
        # pass
    else:
        print(f"The folder '{output_path}' does not exist.")
    
    if os.path.isdir(base_path):
        print(f"The folder '{base_path}' exists.")
        # pass
    else:
        print(f"The folder '{base_path}' does not exist.")

    if os.path.isdir(os.path.join(output_path, 'probes/')):
        # pass
        print(f"The folder '{os.path.join(output_path, 'probes/')} exists.")
    else:
        print(f"The folder '{os.path.join(output_path, 'probes/')} does not exist.")

    if os.path.isdir(folder):
        # pass
        print(f"The folder '{folder} exists.")
    else:
        print(f"The folder '{folder} does not exist.")
    

    # folder = "/Users/jessicakahn/Documents/repos/probing_classifiers/goodreads_data"
    get_recs = GetRecs(args.item_type, folder, model_name)
    # Only implemented for books so far
    data_dict = get_recs.get_data_dict()

    # Get a list of (key, value) pairs
    items = list(data_dict.items())
    
    np.random.seed(42)

    # Randomly sample the desired number of users
    if size_of_sample > len(data_dict):
        size_of_sample = len(data_dict)
    new_dict = dict(random.sample(items, size_of_sample))
    
    # Create a new dictionary from the sampled items
    print('CREATING DATA DICT')

    if torch.cuda.is_available():
        print(torch.cuda.memory_summary(device=None, abbreviated=False))
    gender_dict = get_recs.get_gender_dict(new_dict)
    # torch.cuda.empty_cache()
    # gc.collect()

    embedding_data_dict = get_recs.get_prompts_hidden(new_dict, gender_dict)
    regress_list, results = get_recs.get_regress_list(embedding_data_dict)
    del embedding_data_dict
    i = 0
    for mod in regress_list:
        print('trying to save to path:', os.path.join(output_path, f'probes/model{i}.pkl'))
        with open(os.path.join(output_path, f'probes/model{i}.pkl'),'wb') as f:
            pickle.dump(mod,f)
        i+=1
    print('Results: ', results)

    steer_compare_results = []
    counter = 0

    if model_name == 'tinyllama':
        last_layer = 21
    else:
        last_layer = 27

    # Check on memory
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # Get candidates
    print('CREATING CANDIDATES')
    print(datetime.now())
    cans = Candidates(output_path)
    cans_dict = cans.get_candidates()
    with open(os.path.join(output_path,'can_dict_old.json'), 'w') as json_file:
        json.dump(cans_dict, json_file, indent=4)

    # Restrict candidate set to only users in the subset dict
    subset_cans_dict = {key: cans_dict[str(key)]['ids'] for key in new_dict.keys()}

    all_cands_dict = cans.get_negs_and_vals(new_dict, subset_cans_dict)
    # Map ids to titles
    title_dict = get_recs.get_title_dict()
    # print('title test', title_dict.get(273), title_dict.get('273'))
    for k, v in all_cands_dict.items():
        all_cands_dict[k]['cans_titles'] = [title_dict.get(can_id) for can_id in v['cans_ids']]
    # print('CANS: ', all_cands_dict)


    # TODO []: This dict isn't working properly, lists aren't right
    with open(os.path.join(output_path,'can_dict_new.json'), 'w') as json_file:
        json.dump(all_cands_dict, json_file, indent=4)

    print('Candidate length: ', len(cans_dict), len(subset_cans_dict))
    # del subset_cans_dict # Free up some space
    # if args.save:
    print(datetime.now())

    # Loop over users
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    print(' **** Steering Beginning ***')
    # Define W here and pass it to the function

    device = get_recs.get_device()

    # Precompute the probe weight matrix (as float32) once, on the same device as the model.
    # Note: this is small relative to model parameters; move to CPU if you want
    print('Computing W matrix',  (datetime.now()))
    W_probe = torch.tensor(regress_list[last_layer].coef_, dtype=torch.float32, device=device)
    W_probe_T = W_probe.T
    
    pinv_W_T = torch.linalg.pinv(W_probe).T
    # weights = torch.linspace(0.5, 1.0, seq_len_expected, device=device).view(1, -1, 1)
    print('Start steer_prompt_compare', datetime.now())
    inner_dict = {}
    for k, v in new_dict.items():
        result = get_recs.steer_prompt_compare(
            prompt=v['prompt'],
            alpha=1.0,
            layer_to_steer=last_layer,
            max_new_tokens=150,
            probe_list=regress_list,
            item_type=args.item_type,
            candidates = all_cands_dict[k]['cans_titles'],
            W_probe_T = W_probe,
            pinv_W_T = pinv_W_T
        )
        inner_dict[k] = result
        # torch.cuda.empty_cache()
        # gc.collect()
        counter += 1
        if counter % 10 == 0:
            print(counter, datetime.now())

    torch.save(
        inner_dict, 
        os.path.join(
            output_path,
            f'output_dict_{args.item_type}_{size_of_sample}_withnegs.pt'
                     )
    )
