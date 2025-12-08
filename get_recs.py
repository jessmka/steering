import pandas as pd
import numpy as np
import argparse
import re
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import random
# from probes import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
torch.set_printoptions(profile="full")
import re
import pandas as pd
import os
import collections
pd.set_option('display.max_columns', None)

# Make prompts for gender classification
def make_data_dict(result_df):
    data_dict = {}
    for i, row in result_df.iterrows():
        # sample 3 ids from list
        ids = random.sample(row.book_id_y, 5)
        # group recommendations for each user
        # title_list = []
        # author_list = []
        book_list = []
        for v in ids:
            title = book_dict[str(v)]['title']
            if len(book_dict[str(v)]['authors']) > 0:
                author_id = book_dict[str(v)]['authors'][0]['author_id']
                author_name = author_dict[author_id]
                book_list.append(f"{title} by {author_name}")
            else:
                book_list.append(title)
        
        prompt = f"""
        Hi, I've read and enjoyed the following books: {book_list[0]},  
        {book_list[1]},{book_list[2]},{book_list[3]}  and {book_list[4]}. """
        
        data_dict[row.user_id] = prompt
    return data_dict

def get_gender_dict(new_dict):
    gender_dict = {}
    max_new_tokens = 10
    for k,v in new_dict.items():
        strings = [[{
            "role": "user", 
            "content": v + "Based on these, can you please guess what my gender is? Please respond with only a single word as your answer."
        }]]  # single conversation
        inputs = tokenizer.apply_chat_template(
            strings[0],
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True
        ).to(model.device)
    
        inputs_dict = {"input_ids": inputs}
        with torch.inference_mode(), torch.autocast("cuda"):
                output = model.generate(
                **inputs_dict,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.7,
            )
        
        gender_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # response_only = gender_text[len(v):].strip()
        split_text = re.sub(r'[^a-zA-Z]', '', gender_text.split(".assistant\n\n")[1])
        gender_dict[k] = split_text

    for k,v in gender_dict.items():
        if v not in ('Male', 'Female'):
            gender_dict[k] = 'Unknown'

    return gender_dict

def get_prompts_hidden(new_dict, gender_dict):
    embedding_data_dict = {}
    # Create keys with empty lists for each layer in hidden_states
    for j in range(29):
        embedding_data_dict[j] = []
    # Loop through and add demo + hidden_state to data_dict
    for k,v in new_dict.items():
        inputs = tokenizer(v, return_tensors="pt", padding=True).to(model.device)
        # Run forward pass and request hidden states
        with torch.inference_mode():
            outputs = model(
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
            if idx >=25:
                hidden_repr = repr.mean(dim=1).squeeze(0)
            else:
                hidden_repr = None
            embedding_data_dict[idx].append(dict(demo=gender, hidden=hidden_repr))

    return embedding_data_dict

def get_regress_list(embedding_data_dict):
    regress_list = []
    results = []
    # i = 0
    for key_layer, value in embedding_data_dict.items():
        clf = LogisticRegression(max_iter=1000, solver='lbfgs')

        if key_layer >= 25:
            # for j in value:
            X = [j['hidden'].detach().cpu() for j in value if j['demo']!='Unknown']
            X_tensor = torch.stack(X).cpu().numpy()
            # print(type(X), type(X[0]), X_tensor.shape)
            # i += 1
            # if i >= 1:
                # break
            y = [j['demo'] for j in value if j['demo']!='Unknown']
            # clf = LogisticRegression(multi_class='multinomial',solver='newton-cg')
            
            clf = clf.fit(X_tensor, y)
        
            scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
            results.append(np.array(scores).mean())
            regress_list.append(clf)
        else:
            regress_list.append(clf)

    return regress_list, results



def steer_prompt_compare(
    prompt: str,
    alpha: float,
    layer_to_steer: int,
    max_new_tokens: int,
    probe_list: list
):
    """
    Runs baseline and steered generations for a single prompt.
    Captures pre- and post-steering hidden activations from a given layer.
    """

    # === Prepare model input ===
    chat = [[{"role": "user", "content": prompt + 
              " Please recommend new books based on the user's reading preferences and only return the 5 books you recommend in JSON format like {\"Books\": {'title':..., 'author':...}}, and nothing else."}]]
    inputs = tokenizer.apply_chat_template(
        chat[0],
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True
    ).to(model.device)
    inputs_dict = {"input_ids": inputs}

    # === Run BASELINE (no steering) ===
    with torch.inference_mode(), torch.autocast("cuda"):
        baseline_out = model.generate(
            **inputs_dict,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            # temperature=0.7,
        )
    baseline_text = tokenizer.decode(baseline_out[0], skip_special_tokens=True)
    # print("=== BASELINE OUTPUT ===\n", baseline_text, "\n")

    # === Set up capture and steering for steered run ===
    capture = {"pre": None, "post": None}
    first_pass_done = {"value": False}

    def get_hidden_state_hook(module, input, output):
    # capture only the full-sequence hidden state
        if output.shape[1] == inputs.shape[1] and not first_pass_done["value"]:
            if capture["pre"] is None:
                capture["pre"] = output.to("cpu")


    def steering_hook(module, input, output):
    # steer only if it's the *initial full-sequence* forward
        if output.shape[1] == inputs.shape[1] and not first_pass_done["value"]:
            print('Steering beginning! ')
            hidden = output
            W = torch.tensor(probe_list[layer_to_steer].coef_,
                             dtype=torch.float32, device=hidden.device)
            x_proj0 = hidden @ W.T
            x_proj = x_proj0 @ torch.linalg.pinv(W).T
            v = -x_proj
    
            seq_len = hidden.shape[1]
            weights = torch.linspace(0.5, 1.0, seq_len, device=hidden.device).view(1, seq_len, 1)
            steered = hidden + alpha * weights * v
    
            capture["post"] = steered.to("cpu")
            first_pass_done["value"] = True   # block all later decoding passes
            return steered
        else:
            return output


    # === Register hooks ===
    h0 = model.model.layers[layer_to_steer-4].register_forward_hook(get_hidden_state_hook)
    h1a = model.model.layers[layer_to_steer-4].register_forward_hook(steering_hook)
    h1 = model.model.layers[layer_to_steer-3].register_forward_hook(steering_hook)
    h2 = model.model.layers[layer_to_steer-2].register_forward_hook(steering_hook)
    h3 = model.model.layers[layer_to_steer-1].register_forward_hook(steering_hook)
    h4 = model.model.layers[layer_to_steer].register_forward_hook(steering_hook)
    h5 = model.model.layers[layer_to_steer].register_forward_hook(get_hidden_state_hook)
    # h3 = model.model.layers[layer_to_steer].register_forward_hook(get_hidden_state_hook)

    # === Run STEERED generation ===
    with torch.inference_mode(), torch.autocast("cuda"):
        steered_out = model.generate(
            **inputs_dict,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
        )

    # Clean up hooks
    h0.remove();h1a.remove();h1.remove(); h2.remove(); h3.remove(); h4.remove();h5.remove()

    # === Decode steered text ===
    steered_text = tokenizer.decode(steered_out[0], skip_special_tokens=True)
    # print("=== STEERED OUTPUT ===\n", steered_text, "\n")

    # === Print shapes for sanity check ===
    # print(f"Pre-steered shape:  {capture['pre'].shape if capture['pre'] is not None else None}")
    # print(f"Post-steered shape: {capture['post'].shape if capture['post'] is not None else None}")

    # === Return results ===
    return {
        "baseline_text": baseline_text,
        "steered_text": steered_text,
        "pre_hidden": capture["pre"],
        "post_hidden": capture["post"],
    }

    
if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--size_of_sample", type=int, default=10, help="Number of users to include")
    args = parser.parse_args()
    size_of_sample = args.size_of_sample
    folder = "/home/ec2-user/studies/swb-s4149184-personal-study/goodreads/"
    # folder = "/Users/jessicakahn/Documents/repos/probing_classifiers/goodreads_data"
    df = pd.read_csv(os.path.join(folder, "goodreads_samples2.csv"))
    user_id_map = pd.read_csv(os.path.join(folder, "user_id_map.csv"))
    book_id_map = pd.read_csv(os.path.join(folder, "book_id_map.csv"))
    with open(os.path.join(folder,"book_data.json"), 'r') as f:
        book_dict = json.load(f)
    with open(os.path.join(folder,"author_data.json"), 'r') as f:
        author_dict = json.load(f)
    merged_df = pd.merge(df, book_id_map, left_on ='book_id',right_on='book_id_csv', how='left')
    merged_df['lang'] = merged_df['book_id_y'].apply(lambda x: book_dict[str(x)]['language_code'])

    # Count ratings by user
    merged_df1 = merged_df[merged_df['lang'].isin(['eng','en-US','en-GB','en-CA','en'])]
    totals = merged_df1.groupby('user_id')['rating'].agg(lambda x: (x >= 4).sum()).reset_index()

    totals.columns = ['user_id','rating_count']
    merged_df2 = pd.merge(merged_df1, totals[totals.rating_count >= 10], on ='user_id', how='inner')
    result_df = merged_df2.groupby('user_id')['book_id_y'].agg(list).reset_index()
    print('DF: ', result_df.shape)
    data_dict = make_data_dict(result_df)
    # Load the model

    model_name = "meta-llama/Llama-3.2-3B-Instruct"  # Replace with your desired model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    # Make a small sample of data_dict for testing
    # small_data = {6:data_dict[6], 1077:data_dict[1077]}
    # Get a list of (key, value) pairs
    items = list(data_dict.items())
    
    np.random.seed(42)
    # Randomly sample the desired number of items
    random_items = random.sample(items, size_of_sample)
    
    # Create a new dictionary from the sampled items
    new_dict = dict(random_items)
    gender_dict = get_gender_dict(new_dict)
    print('Gender dict: ', collections.Counter(gender_dict.values()))
    with open(os.path.join(folder,f"gender_dict_{size_of_sample}.json"), "w") as json_file:
        json.dump(gender_dict, json_file, indent=4)

    embedding_data_dict = get_prompts_hidden(new_dict, gender_dict)
    regress_list, results = get_regress_list(embedding_data_dict)
    i = 0
    for mod in regress_list:
        # save
        with open(os.path.join(folder, f'probes/model{i}.pkl'),'wb') as f:
            pickle.dump(mod,f)
        i+=1
    print('Results: ', results)

    steer_compare_results = []
    inner_dict = {}
    al = 1
    counter = 0
    for k, v in new_dict.items():
        result = steer_prompt_compare(
            v,
            alpha=al,
            layer_to_steer=27,
            max_new_tokens=100,
            probe_list=regress_list,
        )
        inner_dict[k] = result
        counter += 1
        if counter % 10 == 0:
            print(counter)

    torch.save(inner_dict, f'output_dict_{size_of_sample}.pt')
    
