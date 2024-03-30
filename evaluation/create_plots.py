import matplotlib.pyplot as plt
import numpy as np
import argparse
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
import time
import pandas as pd

def get_score(fname, dim, n_gens = 3):
    with open(fname, 'r') as fo:
        scores = []
        for i, line in enumerate(fo):
            if dim in line:
                score = float(line.rstrip().replace(f'{dim} score {i % n_gens} = ', ''))
                scores.append(score)
    if len(scores) < n_gens:
        print(fname)
        print(scores)
    scores = np.array(scores)
    return np.mean(scores), np.std(scores)

def get_score_csv(fname, dim):
    df = pd.read_csv(fname)[dim]
    if df.shape[0] != 1000:
        raise RuntimeError("Wrong shape")
    if isinstance(dim, str):
        return df.mean(), df.std()
    else:
        return list(df.mean()), list(df.std())


def plot_alpha_scores(base_dir, dimensions, alphas, outfile, csv=True):
    for dim in dimensions:
        mean_scores = []
        std_scores = []
        curr_alphas = []
        for alpha in alphas:
            if not csv:
                fname=f"{base_dir}/{dim}/a-{alpha}/eval_results.txt"
                curr_alphas.append(alpha)
                mean_score, std_score = get_score(fname, dim)
            else:
                fname=f"{base_dir}/{dim}2/a-{alpha}/eval_results.csv"
                curr_alphas.append(alpha)
                mean_score, std_score = get_score_csv(fname, dim)

            mean_scores.append(mean_score)
            std_scores.append(std_score)
        
        mean_scores = np.array(mean_scores)
        std_scores = np.array(std_scores)

        plt.plot(curr_alphas, mean_scores, '-', label=dim.title())

        plt.fill_between(curr_alphas, mean_scores - std_scores, mean_scores + std_scores, alpha=0.1)
    
    if min(alphas) < 0.0:
        ylim = plt.gca().get_ylim()

        plt.axvline(0.0, linestyle=':', linewidth=1.0, c="black")
        plt.axvline(-1.0, linestyle='--', linewidth=1.0, c="black")

        plt.axvline(1.0, linestyle=':', linewidth=1.0, c="black")
        plt.axvline(2.0, linestyle='--', linewidth=1.0, c="black")

        plt.text(0.15,ylim[0]+0.02,'Interp.')
        plt.text(1.125,ylim[0]+0.02,'Stable\nextrap.')
        plt.text(-0.875,ylim[0]+0.02,'Stable\nextrap.')
        plt.text(2.2,ylim[0]+0.02,'Unstable extrap.')
        plt.text(-3.0,ylim[0]+0.02,'Unstable extrap.')
    
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Attribute Score')
    plt.title('Attribute Score on WritingPrompts')
    plt.legend()
    plt.savefig(outfile)
    return

def plot_lambda_scores(base_dir, dimensions, lambdas, alphas, outfile, base_alpha = 0.0, lambda_cutoff = 0.2, csv=True):
    for i, dim in enumerate(dimensions):
        mean_scores = []
        std_scores = []
        curr_lambdas = []
        for l in lambdas:
            if l == 0.0:
                fname = f"{base_dir}/5dim/5dim-{dim}-a-0.0-{base_alpha}-l-0.25/eval_results.csv"
                vals_lst = list(pd.read_csv(fname)[dim])
            elif l < lambda_cutoff:
                vals_lst = []
                for d in dimensions[:i] + dimensions[i+1:]:
                    for alpha in alphas:
                        fname = f"{base_dir}/5dim/5dim-{d}-a-{alpha}-{base_alpha}-l-{l}/eval_results.csv"
                        vals_lst.extend(list(pd.read_csv(fname)[dim]))
                vals_lst = np.array(vals_lst)
            elif l < 1.0:
                n_digits = 1 if (10-10*l)%4 == 0 else 2
                base_l = round((1.0 - l)/4, n_digits)
                fname = f"{base_dir}/5dim/5dim-{dim}-a-{base_alpha}-{base_alpha}-l-{base_l}/eval_results.csv"
                vals_lst = list(pd.read_csv(fname)[dim])
            else:
                fname = f"{base_dir}/{dim}2/a-{base_alpha}/eval_results.csv"
                vals_lst = list(pd.read_csv(fname)[dim])

            mean_scores.append(np.mean(vals_lst))
            std_scores.append(np.std(vals_lst))
            curr_lambdas.append(l)
        
        mean_scores = np.array(mean_scores)
        std_scores = np.array(std_scores)

        plt.plot(curr_lambdas, mean_scores, '-', label=dim.title())

        plt.fill_between(curr_lambdas, mean_scores - std_scores, mean_scores + std_scores, alpha=0.1)

    plt.xlabel(r'$\lambda$')
    plt.ylabel('Attribute Score')
    plt.title('Attribute Score on WritingPrompts')
    plt.legend()
    plt.savefig(outfile)
    return

def plot_perplexity_scores(base_dir, dimensions, alphas, outfile, log_scale = True, cutoff=None):
    fig, ax = plt.subplots(1, 1)
    min_val = -1
    for dim in dimensions:
        mean_scores = []
        curr_alphas = []
        for alpha in alphas:
            fname=f"{base_dir}/{dim}2/a-{alpha}/perplexity_results.txt"
            try:
                curr_alphas.append(alpha)
                with open(fname, 'r') as fo:
                    for i, line in enumerate(fo):
                        if "perplexity" in line:
                            score = float(line.rstrip().replace(f'perplexity = ', ''))
                            mean_scores.append(score)
            except FileNotFoundError:
                curr_alphas = curr_alphas[:-1]
                continue
        
        mean_scores = np.array(mean_scores)
        curr_min = np.min(mean_scores)
        min_val = min_val if min_val <= curr_min else curr_min

        ax.plot(curr_alphas, mean_scores, '-', label=dim.title())
    
    if min(alphas) < 0.0:
        ax.axvline(0.0, linestyle=':', linewidth=1.0, c="black")
        ax.axvline(-1.0, linestyle='--', linewidth=1.0, c="black")

    if max(alphas) > 1.0:
        ax.axvline(1.0, linestyle=':', linewidth=1.0, c="black")
        ax.axvline(2.0, linestyle='--', linewidth=1.0, c="black")

    
    if log_scale is True:
        ax.set_yscale('log')
    else:
        ax.set_yscale('linear')
        if cutoff is not None:
            ax.set_ylim((min_val - 0.25,cutoff))
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(f'Perplexity Score')
    ax.set_title('WikiText Perplexity')
    ax.legend()
    plt.savefig(outfile)
    return

def plot_3d_simplex(base_dir, lambdas, alphas, dimensions, eval_idx, outfile, csv=True):
    eval_dim = dimensions[eval_idx]
    scores = []
    idxs = []
    clip_val = None
    for lambda1 in lambdas:
        for lambda2 in lambdas:
            for lambda3 in lambdas:
                if round(lambda1 + lambda2 + lambda3, 2) == 1.0:
                    idxs.append([lambda1, lambda2, lambda3])
                    if not csv:
                        curr_fname = f"{base_dir}/{dimensions[0]}-{dimensions[1]}-{dimensions[2]}-a-{alphas[0]}-{alphas[1]}-{alphas[2]}-l-{lambda1}-{lambda2}-{lambda3}-test/eval_results.txt"
                        mean_score, _ = get_score(curr_fname, eval_dim)
                    else:
                        curr_fname = f"{base_dir}/{dimensions[0]}-{dimensions[1]}-{dimensions[2]}-a-{alphas[0]}-{alphas[1]}-{alphas[2]}-l-{lambda1}-{lambda2}-{lambda3}-test/eval_results.csv"
                        mean_score, _ = get_score_csv(curr_fname, eval_dim)
                    scores.append(mean_score)

    idxs_arr = np.array(idxs).T
    
    fig = ff.create_ternary_contour(idxs_arr, np.array(scores),
                                pole_labels=[r'$\large{{\text{{{}}} \lambda}}$'.format(d.title() + " ") for d in dimensions],
                                interp_mode='cartesian',
                                ncontours=20,
                                colorscale='Viridis',
                                showscale=True,)
    
    fig.update_layout(
        title=dict(text=r'$\Large{{\text{{{}}}}}$'.format(f'{eval_dim.title()} Score'), font=dict(size=60)),
        font=dict(
            family='Open Sans',
            size=18, 
            color="Black"
        ),
    )
    fig.update_yaxes(automargin=True)
    fig.update_layout(
        margin=dict(l=100),
    )
    fig.write_image(outfile)
    return

def plot_spider_grid(base_dir, alphas, dimensions, eval_dims, outfile, base_lambdas = [0.2], base_alpha = 1.0, csv=True):
    cols = plotly.colors.DEFAULT_PLOTLY_COLORS
    n = len(eval_dims)
    print("Plotting spider grid")

    row_titles =[r'$\lambda_i=\text{{{}}}$'.format(f"{round(1.0 - 4*l, 1)}") for l in base_lambdas]
    col_titles = [t.title() for t in eval_dims]
    fig = make_subplots(rows=len(base_lambdas), cols=n, specs=[[{"type": "scatterpolar"} for a in range(n)] for b in range(len(base_lambdas))],
                        row_titles=row_titles,
                        column_titles=col_titles,
                        horizontal_spacing=0.06,
                        vertical_spacing = 0.05
    )
    lab_dim = [d.title() for d in dimensions]
    lab_dim[lab_dim.index("Sentiment")] = "Sent."
    lab_dim = lab_dim + [lab_dim[0]]

    for row, base_lambda in enumerate(base_lambdas):
        for i, d in enumerate(eval_dims):
            all_scores = []
            for j, alpha in enumerate(alphas):
                if not csv:
                    fname = f"{base_dir}/5dim-{d}-a-{alpha}-{base_alpha}-l-{base_lambda}/eval_results.txt"
                    scores = []
                    for dim in dimensions:
                        curr_score, _ = get_score(fname, dim)
                        scores.append(curr_score)
                else:
                    fname = f"{base_dir}/5dim-{d}-a-{alpha}-{base_alpha}-l-{base_lambda}/eval_results.csv"
                    scores, _ = get_score_csv(fname, dimensions)

                all_scores.append(scores)

            all_scores = np.array(all_scores)

            for j, scores in enumerate(all_scores):
                showlegend = True if row == 0 and i == 0 else False
                fig.add_trace(go.Scatterpolar(
                    r=list(scores) + [scores[0]],
                    theta=lab_dim,
                    name=r'$\alpha_i=\text{{{}}}$'.format(f"{alphas[j]}"),
                    legendgroup=f"group{j}",
                    marker=dict(color=cols[j]),
                    showlegend=showlegend,
                    ),
                row = row+1, col = i+1)

        fig.update_polars(
        radialaxis=dict(
            visible=True,
            tickmode = 'linear',
            tick0 = 0.0,
            dtick = 0.2,
            range=[0.0, 1.0]
        )
        )

    fig.update_layout(
    height=1400//(n) * len(base_lambdas), width=1700, legend={
            'x': 1.05,
        },
    font=dict(
        size=16,  # Set the font size here
    ),
    title=dict(text="Attribute Scores on WritingPrompts", font=dict(size=25)),
    title_y=0.99 
    )

    fig.for_each_annotation(lambda a:  a.update(xshift=80) if a.text in row_titles else())
    fig.for_each_annotation(lambda a:  a.update(yshift=20, font=dict(size=18)) if a.text in col_titles else())
    fig.write_image(outfile)
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", help='which task to run', type=str)
    parser.add_argument("-b", "--base_directory", help='directory with generations', type=str)
    parser.add_argument("-o", "--output_file", help='a file path for outputting the final png', type=str)
    parser.add_argument("-d", "--dimensions", help="list of dimensions to plot", nargs='+')
    parser.add_argument("-e", "--eval_dimensions", help="list of evaluation dimensions", nargs='+')
    parser.add_argument("-a", "--alphas", help="list of alphas to plot", nargs='+')
    parser.add_argument("-l", "--lambdas", help="list of lambdas to plot", nargs='+')
    parser.add_argument("--base_alpha", help="base alpha", type=float)

    args = parser.parse_args()
    task = args.task
    base_dir = args.base_directory
    outfile = args.output_file
    dimensions = args.dimensions
    eval_dims = args.eval_dimensions
    if args.alphas is not None:
        alphas = [float(a) for a in args.alphas]
    if args.lambdas is not None:
        lambdas = [float(l) for l in args.lambdas]
    base_alpha = args.base_alpha


    figure="some_figure.pdf"
    fig=px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    fig.write_image(figure, format="pdf")
    time.sleep(2)

    if task == "plot_alpha_scores":
        plot_alpha_scores(base_dir, dimensions, alphas, outfile)
    if task == "plot_perplexity_scores":
        plot_perplexity_scores(base_dir, dimensions, alphas, outfile, log_scale=True)
        plt.close()
        plt.clf()
        for c in [7, 10, 50, 100]:
            plot_perplexity_scores(base_dir, dimensions, alphas, f"{outfile[:-4]}_cutoff_{c}.pdf", log_scale=False, cutoff=c)
    if task == "create_simplex":
        for idx in range(len(dimensions)):
            plot_3d_simplex(base_dir, lambdas, alphas, dimensions, idx, f"{outfile}_{dimensions[idx]}.pdf")
    if task == "plot_spider_grid":
        plot_spider_grid(base_dir, alphas, dimensions, eval_dims, outfile, base_lambdas=lambdas, base_alpha=base_alpha if base_alpha is not None else 1.0)
    if task == "plot_lambda_scores":
        plot_lambda_scores(base_dir, dimensions, lambdas, alphas, outfile, base_alpha = base_alpha)

if __name__ == '__main__':
    main()



