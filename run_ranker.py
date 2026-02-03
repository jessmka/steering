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
import re
import os
import collections
from datetime import datetime

class FolderCheck:
    def __init__(self):
        # Define paths
        if torch.cuda.is_available():
            self.base_path = '/home/ec2-user/studies/swb-s4149184-personal-study/'
            if args.model_name is None:
                model_name = 'llama3b'
            else: model_name = args.model_name
            if args.item_type == 'book':
                self.folder = self.base_path
            elif args.item_type == 'movie':
                self.folder = os.path.join(self.base_path,"1/")
            elif args.item_type == 'music':
                self.folder = 'output_data/music'
        else:
            self.base_path = '/Users/jessicakahn/Documents/repos/'
            if args.model_name is None:
                model_name = 'tinyllama'
            else: model_name = args.model_name
            if args.item_type == 'book':
                self.folder = self.base_path
            elif args.item_type == 'movie':
                self.folder = os.path.join(self.base_path,"Glocal_K/1/")
            elif args.item_type == 'music':
                self.folder = 'output_data/music'
        
        self.output_path = os.path.join(self.base_path,f'steering/output_data/{args.item_type}/')
        for dir_path in (self.base_path, self.output_path, os.path.join(self.output_path, 'probes/'), self.folder):
            try:
                # Attempt an operation, for example, listing contents or accessing a file within it
                files = os.listdir(self.output_path)

            except FileNotFoundError:
                print(f"Error: Directory not found at '{self.output_path}'")
                raise
        print('All folders available')

    def get_paths(self):
        return self.base_path, self.output_path, self.folder

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
    
    def get_device(self):
        return self.model.device
            
class GetRecs:
    def __init__(self, item_type, data_path, model_name='meta-llama/Llama-3.2-3B-Instruct'):
        self.item_type = item_type
        # These files are created by data_prep.py
        steering_data_path = os.path.join(data_path,'steering/')
        with open(f'output_data/{item_type}/processed_data_dict.json', 'r') as file:
            self.data_dict = json.load(file)
        with open(f'output_data/{item_type}/title_dict.json', 'r') as file:
            self.title_dict = json.load(file)
        with open(f'output_data/{item_type}/gender_dict.json','r') as file:
            self.gender_dict = json.load(file)

        load_model = LoadModel(model_name)
        self.model = load_model.get_model()
        self.max_layer = load_model.get_layers()
        self.tokenizer = load_model.get_tokenizer()
        self.data_path = steering_data_path
    
    def get_data_dict(self):
        return self.data_dict

    def max_layer(self):
        return self.max_layer
    
    def get_device(self):
        return self.model.device

    def get_prompts_hidden(self, data_dict):
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
            gender = self.gender_dict[k]
            for idx, repr in enumerate(hidden_states):
                if idx >= (self.max_layer-5):
                    hidden_repr = repr.mean(dim=1).squeeze(0).squeeze(0).detach().cpu()
                else:
                    hidden_repr = None
                embedding_data_dict[idx].append(dict(demo=gender, hidden=hidden_repr))
        del hidden_states
        del outputs
        torch.cuda.empty_cache()

        return embedding_data_dict
    
    def get_regress_list(self, embedding_data_dict, log_reg_type='reg'):
        regress_list = []
        results = []
        # i = 0
        for key_layer, value in embedding_data_dict.items():
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
            W_probe_l: any,
            pinv_W_Tl: any,
            W_probe_l_1: any,
            pinv_W_Tl_1: any
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
            noun = 'book'
        elif item_type == 'movie':
            verb = 'watch'
            noun = 'movie'
        elif item_type == 'music':
            verb = 'listen to'
            noun = 'song'
        request_str = f"Please rank the following {noun}s in order from most to least likely to recommend to them to {verb} next. Only give the {noun} ranking with no other content or explanation."
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
        first_pass_done = {"value":False}

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
        def make_steering_hook(W_probe_T_local, pinv_W_T_local):
            def steering_hook(module, input, output):
                try:
                    if output.shape[1] == seq_len_expected and not first_pass_done['value']:
                        hidden = output

                        x_proj0 = hidden @ W_probe_T_local
                        x_proj = x_proj0 @ pinv_W_T_local
                        v = -x_proj

                        steered = hidden + alpha * v

                        # mark done
                        first_pass_done['value'] = True

                        return steered

                except Exception:
                    return output

                return output

            return steering_hook

        hook_l_1 = make_steering_hook(W_probe_l_1, pinv_W_Tl_1)
        hook_l = make_steering_hook(W_probe_l, pinv_W_Tl)
        # === Register hooks safely on the minimal set of layers needed ===
        # (adjust layer indices to your model architecture if necessary)
        # h0 = self.model.model.layers[layer_to_steer].register_forward_hook(get_hidden_state_hook)
        # h1a = self.model.model.layers[layer_to_steer-4].register_forward_hook(steering_hook)
        # h1  = self.model.model.layers[layer_to_steer-3].register_forward_hook(steering_hook)
        # h2  = self.model.model.layers[layer_to_steer-2].register_forward_hook(steering_hook)
        hl_1  = self.model.model.layers[layer_to_steer-1].register_forward_hook(hook_l_1)
        hl  = self.model.model.layers[layer_to_steer].register_forward_hook(hook_l)
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
            hl.remove()
            hl_1.remove()
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
    
    
        

        
if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--size_of_sample", type=int, default=10, help="Number of users to include")
    parser.add_argument("-i","--item_type",type=str, default='movie', help='Type of item recommendation')
    parser.add_argument('-m','--model_name', type=str, default='tinyllama', help ='Model name')
    parser.add_argument('-sr', '--saved_regression', action='store_true', help='Use saved probes from regression')
    # parser.add_argument('')
    args = parser.parse_args()
    size_of_sample = args.size_of_sample
    folder_check = FolderCheck()
    base_path, output_path, folder = folder_check.get_paths()
    get_recs = GetRecs(args.item_type, base_path, args.model_name)
    data_dict = get_recs.get_data_dict()
    # Get a list of (key, value) pairs
    items = list(data_dict.items())
    
    np.random.seed(42)

    # Randomly sample the desired number of users
    if args.size_of_sample > len(data_dict):
        size_of_sample = len(data_dict)

    new_dict = dict(random.sample(items, size_of_sample))

    embedding_data_dict = get_recs.get_prompts_hidden(new_dict)
    log_reg_type = 'reg' # ('elastic', 'reg')

    if args.saved_regression:
        regress_list = []
        for i in range(get_recs.max_layer()):
            with open(os.path.join(output_path, f'probes/model{i}_{log_reg_type}.pkl'),'rb') as f:
                regress_list.append(pickle.load(f))
        
    if not args.saved_regression:
        regress_list, results = get_recs.get_regress_list(embedding_data_dict, log_reg_type)
        print('Regression results: ', results)
        i = 0
        for mod in regress_list:
            with open(os.path.join(output_path, f'probes/model{i}_{log_reg_type}.pkl'),'wb') as f:
                pickle.dump(mod,f)
            i+=1
    steer_compare_results = []
    counter = 0

    if args.model_name == 'tinyllama':
        last_layer = 21
    else:
        last_layer = 27

    print(datetime.now())

    print(' **** Steering Beginning ***')
    # Define W here and pass it to the function
    device = get_recs.get_device()

    # Precompute the probe weight matrix (as float32) once, on the same device as the model.
    # Note: this is small relative to model parameters; move to CPU if you want
    print('Computing W matrix',  (datetime.now()))
    W_probe_l = torch.tensor(regress_list[last_layer].coef_, dtype=torch.float32, device=device)
    # W_probe_T1 = W_probe1.T

    W_probe_l_1 = torch.tensor(regress_list[last_layer-1].coef_, dtype=torch.float32, device=device)
    # W_probe_T2 = W_probe2.T

    # W_probe_T = (W_probe_T1,W_probe_T2 )

    # Calculate pseudo-inverse
    pinv_W_Tl = torch.linalg.pinv(W_probe_l).T
    pinv_W_Tl_1 = torch.linalg.pinv(W_probe_l_1).T
    # weights = torch.linspace(0.5, 1.0, seq_len_expected, device=device).view(1, -1, 1)
    print('Start steer_prompt_compare', datetime.now())
    inner_dict = {}
    for k, v in new_dict.items():
        result = get_recs.steer_prompt_compare(
            prompt=v['prompt'],
            alpha=1.0,
            layer_to_steer=last_layer,
            max_new_tokens=200,
            probe_list=regress_list,
            item_type=args.item_type,
            candidates = v['pos_titles']+v['neut_titles']+v['neg_titles'],
            W_probe_l = W_probe_l,
            pinv_W_Tl = pinv_W_Tl,
            W_probe_l_1 = W_probe_l_1,
            pinv_W_Tl_1 = pinv_W_Tl_1
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
            f'output_dict_{args.item_type}_{size_of_sample}_withnegs_{log_reg_type}.pt'
                     )
    )
    


