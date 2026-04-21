import argparse
import re
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, make_scorer
import os
import collections
from datetime import datetime
from dataclasses import dataclass, field
from helpers import dcg_at_k, ndcg_at_k
from functools import partial





class LoadData:
    def __init__(self, item_type, model_name, sample=False):
        self.item_type = item_type

        with open(f'output_data/{self.item_type}/reverse_title_dict.json', 'r') as file_object:
            self.reverse_title_dict = json.load(file_object)
        with open(f'output_data/{self.item_type}/processed_data_dict.json', 'r') as file:
            data_dict = json.load(file)

        if sample:
            sample_size=20
            keys_list = list(data_dict.keys())
            random_keys = random.sample(keys_list, sample_size)
            random_samples = {}
            for key in random_keys:
                random_samples[key] = data_dict[key]
            self.data_dict = random_samples
        else:
            self.data_dict = data_dict
        
        n = len(self.data_dict)
        print('DATA DICT LEN: ',n)

        with open(f'output_data/{self.item_type}/gender_dict.json','r') as file:
            self.gender_dict = json.load(file)

        file_path = f'output_data/{self.item_type}/embeddings/embed_{n}_{model_name}.pkl'
        with open(file_path, 'rb') as file:
            self.embedding_data_dict = pickle.load(file)

        
        if torch.cuda.is_available():
            self.base_path = '/home/ec2-user/studies/swb-s4149184-personal-study/'
            movie_data_path = os.path.join(self.base_path,'1/ml-100k/')
        else:
            self.base_path = '/Users/jessicakahn/Documents/repos/'
            movie_data_path = os.path.join(self.base_path,'Glocal_K/1/ml-100k/')
            

        if self.item_type == 'movie':
            header = ['user_id', 'movie_id', 'rating', 'timestamp']
            orig_rating_df = pd.read_csv(os.path.join(movie_data_path,'u.data'), sep='\t', names=header)
            self.orig_rating_dict = (
                orig_rating_df.groupby("user_id")
                .apply(lambda g: dict(zip(g["movie_id"], g["rating"])))
                .to_dict()
            )
        elif self.item_type == 'music':
            orig_rating_df = pd.read_csv('output_data/music/orig_rating_df.csv')
            self.orig_rating_dict = (
                orig_rating_df.groupby('user_id')
                .apply(lambda g: dict(zip(g['item_id'], g['rating'])))
                .to_dict()
            )
        elif self.item_type == 'book':
            orig_rating_df = pd.read_csv('output_data/book/orig_rating_df.csv')
            self.orig_rating_dict = (
                orig_rating_df.groupby('user_id')
                .apply(lambda g: dict(zip(g['item_id'], g['rating'])))
                .to_dict()
            )

class LoadModel:
    def __init__(self, model_str):
        model_dict = {
            'llama3b':'meta-llama/Llama-3.2-3B-Instruct', 
            'tinyllama':'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            'mistral':"mistralai/Mistral-7B-Instruct-v0.2",
            'qwen':'Qwen/Qwen2.5-7B-Instruct',
            'phi': 'microsoft/Phi-3-mini-128k-instruct',
            'qwen3b':'Qwen/Qwen2.5-3B'}
        model_name = model_dict[model_str]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                        device_map="auto",
                                                        torch_dtype=torch.float16)
        model_layer_dict = {
            'meta-llama/Llama-3.2-3B-Instruct':28,
            'TinyLlama/TinyLlama-1.1B-Chat-v1.0':22,
            'mistralai/Mistral-7B-Instruct-v0.2':32,
            'Qwen/Qwen2.5-7B-Instruct':28,
            'microsoft/Phi-3-mini-128k-instruct':32,
            'Qwen/Qwen2.5-3B':36
        }
        self.max_layer = model_layer_dict[model_name]
    
    def get_max_layer(self):
        return self.max_layer

class Regression:
    def __init__(self, 
                 model_instance, 
                 app_data, 
                 output_path='output_data/movie',
                 log_reg_type='reg',
                 load_from_saved=True):
        self.app_data = app_data

        if load_from_saved:
            self.regress_dict = {}
            for i in range(model_instance.get_max_layer()):
                with open(os.path.join(output_path, f'probes/model{i}_{log_reg_type}.pkl'),'rb') as f:
                    self.regress_dict[i] = pickle.load(f)
        else:
            self.regress_dict, results = self.get_regress_list('reg','gender')

    def get_regress_dict(self):
        return self.regress_dict

    ### Run Regression ###
    def get_regress_list(self,log_reg_type='reg', demo_var='gender'):
        """ Run the classification regressions for the given demo var"""
        regress_dict = {}
        results = []
        if demo_var in( 'gender', 'age_binary'):
            scorer = 'roc_auc'
        else:
            scorer = make_scorer(
                roc_auc_score,
                multi_class='ovr', # or 'ovo'
                response_method="predict_proba",
                average='macro' # or 'weighted'
            )

        for key_layer, value in self.app_data.embedding_data_dict.items():
            if log_reg_type == 'reg':
                clf = LogisticRegression(max_iter=1000, solver='lbfgs')
            elif log_reg_type=='elastic':
                clf = LogisticRegression(
                    penalty='elasticnet',
                    solver='saga',
                    l1_ratio=0.5,  # 50% L1, 50% L2
                    C=0.1,
                    random_state=0
                )

            # Run regression for the 2 layers before layer_to_steer
            if (key_layer >= 17):
                X = [j['hidden'].detach().cpu() for j in value if demo_var in j['demo'] and j['demo'][demo_var]!='Unknown']
                X_tensor = torch.stack([
                    x.to(torch.float32)          # convert each element
                    .detach()
                    .cpu()
                    for x in X
                ])
                X_np = X_tensor.numpy() # Convert to numpy array
                y = [j['demo'][demo_var] for j in value if demo_var in j['demo'] and j['demo'][demo_var]!='Unknown']
                skf = 5   
                clf = clf.fit(X_np, y)
                    
                scores = cross_val_score(clf, X_np, y, cv=skf, scoring=scorer)
                results.append(np.array(scores).mean())
                
                regress_dict[key_layer] = clf
            # else:
            #     regress_list.append(clf)
        print('REGRESSION RESULTS: ', results)
        return regress_dict, results
    
    def steering_weights(self):
        """Returns the matrics for each layer required for steering"""
        layers = list(self.regress_dict.keys())
        W_dict = {i:None for i in layers}

        for layer_to_steer in layers:
            W_probe_l = torch.tensor(self.regress_dict[layer_to_steer].coef_, dtype=torch.float32, device=device)
            pinv_W_Tl = torch.linalg.pinv(W_probe_l).T
            W_dict[layer_to_steer] = (W_probe_l.T,pinv_W_Tl)

        return W_dict

class Steering:
    def __init__(self, model_instance, app_data):
        # self.mode = mode
        self.app_data = app_data
        self.model_instance = model_instance
        # self.W_dict = W_dict

    def ranker_baseline(self, history, candidates):
        """
        Ranks movie candidates based on the model's likelihood scores.
        """
        self.model_instance.model.eval()
        results = []
        
        base_prompt = f"User history: {history}. Recommended movie:"
        # base_prompt = f"User history: {history}. Please don't take my gender into account and recommend a movie to watch next:"
        # base_prompt = f"User history of liked movies: {history}. Please rank the following movies: {candidates} from most to least likely to watch next: "
        
        for movie in candidates:
            full_text = f"{base_prompt} {movie}"
            inputs = self.model_instance.tokenizer(full_text, return_tensors="pt").to(self.model_instance.model.device)
            input_ids = inputs["input_ids"]
            
            with torch.inference_mode():
                outputs = self.model_instance.model(**inputs)
                # Logits shape: [1, seq_len, vocab_size]
                logits = outputs.logits
                # print('LOGITS', logits)
                # print('LOGITS2', outputs.logits[0, :-1])
                # 1. Convert raw logits to log-probabilities
                log_probs = torch.log_softmax(logits, dim=-1)
                
                
                # Only need the log-probs of the "movie title" tokens
                # These are the tokens from the end of the prompt
                shift_log_probs = log_probs[..., :-1, :].contiguous()
                shift_input_ids = input_ids[..., 1:].contiguous()
                
                # 3. Gather the log-probs of the actual tokens that make up the movie title
                # We look at the last N tokens, where N is the length of the movie title
                movie_token_ids = self.model_instance.tokenizer(f" {movie}", return_tensors="pt", add_special_tokens=False)["input_ids"]
                num_movie_tokens = movie_token_ids.shape[1]
                
                # Extract only the log-probs for the candidate movie title part
                target_log_probs = shift_log_probs[0, -num_movie_tokens:]
                target_ids = shift_input_ids[0, -num_movie_tokens:]
                
                # Gather the specific log-probs for the tokens in the title
                token_scores = target_log_probs.gather(1, target_ids.unsqueeze(1)).squeeze()
                
                # Calculate the mean Log-Probability for this movie
                mean_score = token_scores.mean().item()
                results.append((movie, mean_score))

        # Sort results: Higher (less negative) is better
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def ranker_steered(self, history, candidates, W_dict,layer_to_steer,alpha=1):
        scores = []
        # layer_to_steer = 16

        # Clean up any previous hooks
        for layer in self.model_instance.model.model.layers:
            layer._forward_hooks.clear()

        history_ids = self.model_instance.tokenizer(history, return_tensors="pt").to(device)
        history_len = history_ids["input_ids"].shape[1]

        def make_steering_hook(W_probe, alpha=1):
            # Preprocess: transpose & normalize once
            w = W_probe[0]
            if w.shape[0] == 1:
                w = w.T   # make [hidden_dim, 1]
            w = w / torch.norm(w)

            def steering_hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output

                # Move probe to same device/dtype as hidden
                w_curr = w.to(device=hidden.device, dtype=hidden.dtype)

                proj = (hidden @ w_curr) @ w_curr.T
                steered = hidden - alpha * proj

                if isinstance(output, tuple):
                    return (steered,) + output[1:]
                return steered

            return steering_hook

        hook_prev = make_steering_hook(W_dict[layer_to_steer-1], alpha)
        hook_curr = make_steering_hook(W_dict[layer_to_steer], alpha)

        h_prev = self.model_instance.model.model.layers[layer_to_steer-1].register_forward_hook(hook_prev)
        h_curr = self.model_instance.model.model.layers[layer_to_steer].register_forward_hook(hook_curr)

        for movie in candidates:
            text = f"{history} Recommended: {movie}"
            inputs = self.model_instance.tokenizer(text, return_tensors="pt").to(device)

            with torch.inference_mode():
                outputs = self.model_instance.model(**inputs)

            logits = outputs.logits
            log_probs = torch.log_softmax(logits, dim=-1)
            input_ids = inputs["input_ids"]

            candidate_log_probs = [
                log_probs[0, i, input_ids[0, i+1]] 
                for i in range(history_len, input_ids.shape[1] - 1)
            ]

            score = torch.sum(torch.stack(candidate_log_probs))
            scores.append((movie, score))

        # Remove hooks after loop
        h_prev.remove()
        h_curr.remove()

        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    def get_rankings_baseline(self):
        counter = 0
        results_dict = {}
        for k,v in self.app_data.data_dict.items():
            history = ",".join(v['prompt_titles'])
            candidates = v['pos_titles'] + v['neut_titles'] + v['neg_titles']
            random.shuffle(candidates)
            results = self.ranker_baseline(history, candidates)
            counter += 1
            results_dict[k] = results
            if counter % 25 ==0:
                print(counter, datetime.now())
        return results_dict
    
    def get_rankings_steered(self, W_dict, layer_to_steer,alpha):
        """ Runs the rankings function depending on the class mode on the dataset"""
        counter = 0
        results_dict = {}
        for k,v in self.app_data.data_dict.items():
            # sweep_results[k] = {}
            history = ",".join(v['prompt_titles'])
            candidates = v['pos_titles'] + v['neut_titles'] + v['neg_titles']
            random.shuffle(candidates)
            results = self.ranker_steered(history, candidates, W_dict, layer_to_steer, alpha)
            counter += 1
            results_dict[k] = results
            if counter % 25 ==0:
                print(counter, datetime.now())
        return results_dict


    

class NDCGCalc:
    def __init__(self, app_data):
        self.app_data = app_data

    def calc_ndcg_results(self, results_dict):
        """ Calculate NDCG from rankings result"""
        ndcg_results_dict = {}
        for k, v in results_dict.items():
            ndcg_results_dict[k] = dict(user_gender=None, ndcg_results={})
            ranked_ids = [self.app_data.reverse_title_dict.get(i[0].split(" by ")[0],[]) for i in v]
            flat_list = [item for sublist in ranked_ids for item in sublist]
            # print(flat_list)
            actual = self.app_data.orig_rating_dict[int(k)]
            s = [actual.get(i) for i in flat_list if actual.get(i) is not None]
            user_gender = self.app_data.gender_dict[k]
            ndcg_dict = {}

            for i in range(1,11):
                ndcg_dict[i] = ndcg_at_k(s, i)

            ndcg_results_dict[k]['user_gender'] = user_gender
            ndcg_results_dict[k]['ndcg_results'] = ndcg_dict

        return ndcg_results_dict     
    

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--item_type",type=str, default='movie', help='Type of item recommendation')
    parser.add_argument('-m','--model_name', type=str, default='tinyllama', help ='Model name')
    parser.add_argument('-l', '--layer_to_steer', type=int, help="Which layer to steer on will do this and one before")
    parser.add_argument('-a', '--alpha', type=float, help="Strength of steering vector")
    parser.add_argument('-d', '--demo', type=str, default='gender', help='Demographic for regression')
    parser.add_argument('-bs','--bsorst', type=str, default='baseline',help='baseline or steered')
    parser.add_argument('-s', '--sample', action='store_true', help='Bool: sample of size 20 or full dataset')
    parser.add_argument('-lf','--load_from_file',action='store_true',help='Bool:load from file')
    args = parser.parse_args()

    base_path = '/Users/jessicakahn/Documents/repos/'

    device = "cuda" if torch.cuda.is_available() else "mps"
    log_reg_type = 'reg'
    mode = args.bsorst

    if not args.load_from_file:
        # Do steering
        print(f'Running ranker for mode = {mode}, layer={args.layer_to_steer}, alpha={args.alpha}')
        app_data = LoadData(args.item_type, args.model_name, sample=args.sample) # Only use sample for testing
        model_load = LoadModel(args.model_name)
        reg = Regression(model_load, app_data, load_from_saved=False)
        steering = Steering(model_load, app_data)
        ndcg_calc = NDCGCalc(app_data)

        if mode == 'baseline':
            W_dict = {}
            rank_results = steering.get_rankings_baseline()
        elif mode == 'steered':
            W_dict = reg.steering_weights()
            rank_results = steering.get_rankings_steered(W_dict, args.layer_to_steer, args.alpha)

        filename = f'output_data/{args.item_type}/logit_outputs/output_{mode}_{args.model_name}_{args.layer_to_steer}_{args.alpha}.json'

        try:
            with open(filename, 'w') as json_file:
                json.dump(rank_results, json_file, indent=4)
        except:
            try:
                torch.save(rank_results, f'output_data/{args.item_type}/logit_outputs/output_{mode}_{args.model_name}_{args.layer_to_steer}_{args.alpha}.pt')
            except:
                print('Unable to save rank_results')
    else:
        app_data = LoadData(args.item_type, args.model_name, sample=args.sample) # Only use sample for testing
        ndcg_calc = NDCGCalc(app_data)
        with open(f'output_data/{args.item_type}/logit_outputs/output_{mode}_{args.model_name}_{args.layer_to_steer}_{args.alpha}_newp.json', 'r', encoding='utf-8') as file:
        # Use json.load() to deserialize the file's content
            rank_results = json.load(file)

    
    print('Calculating NDCG')
    ndcg_results = ndcg_calc.calc_ndcg_results(rank_results)

    # Save data
    filename = f'output_data/{args.item_type}/logit_outputs/{mode}_ndcg_{args.model_name}_{args.layer_to_steer}_{args.alpha}_newp2.json'
    with open(filename, 'w') as json_file:
        json.dump(ndcg_results, json_file, indent=4)
    



    










