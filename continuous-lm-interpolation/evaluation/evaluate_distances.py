import numpy as np 
from sklearn.manifold import MDS
from sklearn.cross_decomposition import CCA
import argparse 
import json
import torch
import gc
from peft import load_peft_weights
import matplotlib.pyplot as plt
import pickle
import os
import seaborn as sns
import torch.nn.functional as F
import math

def compute_pairwise_distances(model_list, alphas=[0.0, 1.0]):
    print("computing distances")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = len(model_list)
    na = len(alphas)
    d_mat = np.zeros(((n-1)//2*na + 1, (n-1)*na//2 + 1))
    r = 0
    for i in range(0, len(model_list), 2):
        path1 = model_list[i]
        if path1 == "base_model":
            model1_weights = load_peft_weights(model_list[0], device=device)
            model1_weights = {k:torch.tensor([0.0]) for k in model1_weights}
            model1_weights_anti = None
            alphas1 = [1.0]
        else:
            model1_weights = load_peft_weights(path1, device=device)
            model1_weights_anti = load_peft_weights(model_list[i+1], device=device)
            alphas1 = alphas
        
        for alpha1 in alphas1:
            c = 0
            for j in range(0, len(model_list), 2): #, path2 in enumerate(model_list):
                path2 = model_list[j]
                if path2 == "base_model":
                    model2_weights = {k:torch.tensor([0.0]) for k in model1_weights}
                    model2_weights_anti = None
                    alphas2 = [1.0]
                else:
                    model2_weights = load_peft_weights(path2, device=device)
                    model2_weights_anti = load_peft_weights(model_list[j+1], device=device)
                    alphas2 = alphas
                
                for alpha2 in alphas2:
                    tot_dist = 0.0
                    for key in model1_weights:
                        if "lora_A" in key:
                            A_key = key
                            B_key = key.replace("lora_A", "lora_B")

                            curr_weight1 = alpha1 * torch.matmul(model1_weights[B_key], model1_weights[A_key])
                            if model1_weights_anti:
                                curr_weight1 += (1 - alpha1) * torch.matmul(model1_weights_anti[B_key], model1_weights_anti[A_key])

                            curr_weight2 = alpha2 * torch.matmul(model2_weights[B_key], model2_weights[A_key])
                            if model2_weights_anti:
                                curr_weight2 += (1 - alpha2) * torch.matmul(model2_weights_anti[B_key], model2_weights_anti[A_key])

                            tot_dist += torch.norm(curr_weight1 - curr_weight2).item()
                        elif "lora_B" not in key:
                            print(key)
                            
                    d_mat[r, c] = tot_dist

                    c += 1

                del model2_weights
                del model2_weights_anti
                torch.cuda.empty_cache()
                gc.collect()

            r += 1
        
        del model1_weights
        del model1_weights_anti
        torch.cuda.empty_cache()
        gc.collect()    

    return d_mat

def compute_correlations(model_list, alphas=[0.0, 1.0], dist_type="CCA", n_comp = 2, out_r=None, out_c=None):
    print("computing distances")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = len(model_list)
    na = len(alphas)
    d_mat = np.zeros((n//2*na, n//2*na))
    r = 0
    for i in range(0, len(model_list), 2):
        if (out_r is None) or (out_r >= r and out_r < r + na):
            path1 = model_list[i]
            if path1 == "base_model":
                model1_weights = load_peft_weights(model_list[0], device=device)
                model1_weights = {k:torch.tensor([0.0]) for k in model1_weights}
                model1_weights_anti = None
                alphas1 = [1.0]
            else:
                model1_weights = load_peft_weights(path1, device=device)
                model1_weights_anti = load_peft_weights(model_list[i+1], device=device)
                alphas1 = alphas
            
            for alpha1 in alphas1:
                c = 0
                if (out_r is None) or r == out_r:
                    for j in range(0, len(model_list), 2):
                        if (out_c is None) or (out_c >= c and out_c < c + na):
                            path2 = model_list[j]
                            if path2 == "base_model":
                                model2_weights = {k:torch.tensor([0.0]) for k in model1_weights}
                                model2_weights_anti = None
                                alphas2 = [1.0]
                            else:
                                model2_weights = load_peft_weights(path2, device=device)
                                model2_weights_anti = load_peft_weights(model_list[j+1], device=device)
                                alphas2 = alphas
                            
                            for alpha2 in alphas2:
                                if (out_c is None) or c == out_c:
                                    tot_dist = []
                                    for key in model1_weights:
                                        if "lora_A" in key:
                                            A_key = key
                                            B_key = key.replace("lora_A", "lora_B")

                                            curr_weight1 = alpha1 * torch.matmul(model1_weights[B_key], model1_weights[A_key])
                                            if model1_weights_anti:
                                                curr_weight1 += (1 - alpha1) * torch.matmul(model1_weights_anti[B_key], model1_weights_anti[A_key])

                                            curr_weight2 = alpha2 * torch.matmul(model2_weights[B_key], model2_weights[A_key])
                                            if model2_weights_anti:
                                                curr_weight2 += (1 - alpha2) * torch.matmul(model2_weights_anti[B_key], model2_weights_anti[A_key])

                                            if dist_type == "CCA":
                                                cca = CCA(n_components = n_comp).fit(curr_weight1.cpu(), curr_weight2.cpu())
                                                tot_dist.append(np.mean((np.diag(np.corrcoef(cca._x_scores, cca._y_scores, rowvar=False)[:n_comp, n_comp:]))))
                                            elif dist_type == "L2":
                                                tot_dist.append(torch.linalg.matrix_norm(curr_weight1 - curr_weight2).item()**2)
                                            else:
                                                tot_dist.append(torch.mean(F.cosine_similarity(curr_weight1, curr_weight2)).item())
                                        elif "lora_B" not in key:
                                            print(key)
                                        
                                    d_mat[r, c] = sum(tot_dist)/len(tot_dist)
                                    print(f"Wrote to row {r} column {c}")
                                    print(d_mat[r, c])
                                    del tot_dist
                                    gc.collect()

                                c += 1

                            del model2_weights
                            del model2_weights_anti
                            torch.cuda.empty_cache()
                            gc.collect()
                        else:
                            c += na

                r += 1
            
            del model1_weights
            del model1_weights_anti
            torch.cuda.empty_cache()
            gc.collect()  
        else:
            r += na  

    return d_mat

def compute_mds(distances):
    # Create an MDS object with 
    # 2 dimensions and random start 
    mds = MDS(n_components=2, random_state=0, dissimilarity='precomputed') 
    
    # Fit the data to the MDS 
    # object and transform the data 
    return mds.fit_transform(distances)

def plot_mds(dist_mat, labels, alphas, outfile):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)
    dist_mat = dist_mat - dist_mat[-1, :]

    for i in range(len(labels)//2):
        ax.plot(dist_mat[i*len(alphas):(i+1)*len(alphas), 0], dist_mat[i*len(alphas):(i+1)*len(alphas), 1], alpha=0.25)
        ax.scatter(dist_mat[i*len(alphas):(i+1)*len(alphas), 0], dist_mat[i*len(alphas):(i+1)*len(alphas), 1])
    
    ax.scatter(dist_mat[-1, 0], dist_mat[-1, 1], c="black")
    ax.grid(True, which='both')

    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()

    al = len(alphas)
    a0 = alphas.index(0.0)
    a1 = alphas.index(1.0)
    for r in range(0, len(labels)-1, 2):
        ax.text(dist_mat[r//2*al + a0,0] + 1, dist_mat[r//2*al + a0,1] + 1, labels[r])
        ax.text(dist_mat[r//2*al + a1,0] + 1, dist_mat[r//2*al + a1,1] + 1, labels[r+1])
    ax.text(dist_mat[-1, 0] + 1, dist_mat[-1, 1] + 1, labels[-1])
    plt.savefig(outfile)
    return

def plot_correlation(corr_mat, model_list, outfile, title_str = "Cosine Similarity"):
    params = {'xtick.labelsize': 14,
              'ytick.labelsize': 14,
          'axes.titlesize': 16}
    plt.rcParams.update(params) 
    #plotting the heatmap for correlation
    plt.figure(figsize=(8, 6))
    mlist = []
    for m in model_list:
        if "Sentiment" not in m:
            mlist.append(m)
        else:
            mlist.append(f"{m[:3]}. Sent.")
    ax = sns.heatmap(corr_mat.round(3), annot=True, xticklabels=mlist, yticklabels=mlist)
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
    plt.title(f"Average {title_str} Between LoRA Layers")
    ax.figure.tight_layout()
    plt.savefig(outfile)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--distance_type", help="type of analysis to do and plot", type=str)
    parser.add_argument("-m", "--models_dict", help='a json file with the names of models', type=str)
    parser.add_argument("-o", "--output_file", help='a file path for outputting the final png', type=str)
    parser.add_argument("-l", "--load_dir", help='a directory path for loading the matrices', type=str, default=None)
    parser.add_argument("-a", "--alphas", help="list of alphas to plot", nargs='+')
    parser.add_argument("-i", "--index", help="list of indices", nargs='+')
    args = parser.parse_args()
    distance_type = args.distance_type
    models_dict_path = args.models_dict
    outfile = args.output_file
    load_dir = args.load_dir
    if args.alphas is not None:
        alphas = [float(a) for a in args.alphas]
    else:
        alphas = [0.0, 1.0]
    if args.index is not None:
        indices = [float(i) for i in args.index]
    else:
        indices = None

    with open(models_dict_path) as json_file:
        model_dict = json.load(json_file)

        print("Getting Model Paths")
        model_names = []
        model_paths = []
        i = 0
        for dimension in model_dict:
            for expert in "expert", "antiexpert":
                new_key = f'{dimension}_{expert}'
                model_names.append(new_key)
                if dimension != "politeness":
                    model_paths.append(model_dict[dimension][expert])
                else:
                    model_paths.append(model_dict[dimension][expert][:-1])
                i += 1
        if distance_type == "MDS":
            model_names.append("base_model")
            model_paths.append("base_model")

    dct = {
        "sentiment":{"expert": "Positive Sentiment", "antiexpert": "Negative Sentiment"},
        "politeness":{"expert": "Polite", "antiexpert": "Impolite"},
        "humor":{"expert": "Humorous", "antiexpert": "Nonhumorous"},
        "formality":{"expert": "Formal", "antiexpert": "Informal"},
        "simplicity":{"expert": "Simple", "antiexpert": "Complex"}
    }
    mnames = []
    loop_arr = model_names if distance_type != "MDS" else model_names[:-1]
    for m in loop_arr:
        [dim, expert] = m.split("_")
        mnames.append(dct[dim][expert])
    if distance_type == "MDS":
        mnames.append("Llama-2-7b Base Model")

    print("Starting...")
    if distance_type == "MDS":
        if not load_dir or not os.path.exists(f"{load_dir}/d_mat_transformed"):
            print("Computing Pairwise Distances")
            d_mat = compute_pairwise_distances(model_paths, alphas=alphas)
            print(d_mat)

            print("Computing MDS")
            twodim_distances = compute_mds(d_mat)
            print(twodim_distances)

            print("Saving")
            if load_dir:
                if not os.path.exists(load_dir):
                    os.makedirs(load_dir)

                pickle.dump(d_mat, open(f"{load_dir}/d_mat", "wb"))
                pickle.dump(twodim_distances, open(f"{load_dir}/d_mat_transformed", "wb"))
        else:
            print("Loading")
            twodim_distances = pickle.load(open(f"{load_dir}/d_mat_transformed", "rb"))

        print("Plotting")
        plot_mds(twodim_distances, mnames, alphas, outfile)
    elif distance_type == "CCA":
        if indices is not None and not load_dir or not os.path.exists(f"{load_dir}/cca_mat"):
            for i in indices:
                r = int(math.floor(-0.5 + math.sqrt(0.25 + 2*i)))
                c = int(i - (r * (r + 1) / 2))
                curr_cca = compute_correlations(model_paths, alphas=alphas, dist_type="CCA", out_r=r, out_c=c)
                if load_dir:
                    if not os.path.exists(load_dir):
                        os.makedirs(load_dir)
                        
                    pickle.dump(curr_cca, open(f"{load_dir}/cca_row_{r}_column_{c}", "wb"))
                    pickle.dump(curr_cca, open(f"{load_dir}/cca_row_{c}_column_{r}", "wb"))

        elif not load_dir or not os.path.exists(f"{load_dir}/cca_mat"):
            print("Computing Pairwise Correlations")
            cca_mat = compute_correlations(model_paths, alphas=alphas, dist_type="CCA")

            print("Saving")
            if load_dir:
                if not os.path.exists(load_dir):
                    os.makedirs(load_dir)

                pickle.dump(cca_mat, open(f"{load_dir}/cca_mat", "wb"))
        else:
            print("Loading")
            cca_mat = pickle.load(open(f"{load_dir}/cca_mat", "rb"))

        if indices is None:
            print("Plotting")
            plot_correlation(cca_mat, mnames, outfile, title_str="CCA Correlation")
    elif distance_type == "L2":
        if not load_dir or not os.path.exists(f"{load_dir}/l2_mat"):
            print("Computing Pairwise Squared L2 Norm")
            l2_mat = compute_correlations(model_paths, alphas=alphas, dist_type="L2")

            print("Saving")
            if load_dir:
                if not os.path.exists(load_dir):
                    os.makedirs(load_dir)

                pickle.dump(l2_mat, open(f"{load_dir}/l2_mat", "wb"))
        else:
            print("Loading")
            l2_mat = pickle.load(open(f"{load_dir}/l2_mat", "rb"))

        print("Plotting")
        plot_correlation(l2_mat, mnames, outfile, title_str="Squared L2 Norm")
    else:
        if not load_dir or not os.path.exists(f"{load_dir}/cossim_mat"):
            print("Computing Pairwise Cosine Similarity")
            cossim_mat = compute_correlations(model_paths, alphas=alphas, dist_type="Cosine Similarity")

            print("Saving")
            if load_dir:
                if not os.path.exists(load_dir):
                    os.makedirs(load_dir)

                pickle.dump(cossim_mat, open(f"{load_dir}/cossim_mat", "wb"))
        else:
            print("Loading")
            cossim_mat = pickle.load(open(f"{load_dir}/cossim_mat", "rb"))

        print("Plotting")
        plot_correlation(cossim_mat, mnames, outfile, title_str="Cosine Similarity")

if __name__ == '__main__':
    main()