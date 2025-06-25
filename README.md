# Piece-wise Polynomial functions approximated using Optimal Regression Trees

This is a repository containing the implementation and replication information for an article

Piecewise Polynomial Regression of Tame Functions via Integer Programming \
_Gilles Bareilles, Johannes Aspman, **Jiří Němeček**, Jakub Mareček_ \
[link to the preprint](https://arxiv.org/abs/2311.13544)

## Source code

All source files are in the `src` directory. Most notable is the `PWPolyTree_MIP.py` file, containing the implementation of the axis-aligned and affine-hyperplane MIP formulations.
`noise_sample_data.py` is a generator for the denoising scenarios. It is a direct reimplementation of the original work introducing the scenarios.

The code used to stratify the Neural Network outputs is not attached since it is well described in the appendix and is not as relevant to the topic of the work.

## Experiments

The main experimental results (those present in the main body of the article) can be replicated by running the

- `cones.py` to generate results for the cone function,
- `nn_fits.py` to generate the results for the Neural Network approximation, including the training of the NN,
- `denoising.py` to generate the results for the four denoising scenarios,
- and finally, `optimal_norm.py` to store the entire optimization process until optimality on the $\|\cdot\|_{\infty}$ norm.

The scripts save the results into `.pickle` files, which can then be handled separately. The precomputed results in `.pickle` files are in the `results` directory.

Lastly, the `src` directory also contains two Jupyter notebooks.

- `visualizations.ipynb` contains code used to generate the figures and tables in the main body of the article from the results in the `.pickle` files (this requires that you move them to the same folder or change their paths in the notebook),
- and `appendix.ipynb`, which contains the code required to generate and visualize the results described in the appendix of the article.

### Requirements

We also provide the `requirements.txt` file containing the list of packages and versions used to run the code.

```bash
# optionally create new conda env
conda create -n tame_pwp python==3.11
conda activate tame_pwp

# install the required packages
pip install -r requirements.txt

# run any experiment
python cones.py
python nn_fits.py
python denoising.py
python optimal_norm.py

# or generate the plots
jupyter lab .
```

## Figures

In the `figs` folder are the generated figures, which are the exact figures created as output of the two Jupyter notebooks.
