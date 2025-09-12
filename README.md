# Fairer NMF

Code for [Towards a Fairer Non-negative Matrix Factorization](https://arxiv.org/abs/2411.09847) by Lara Kassab, Erin George, Deanna Needell, Haowen Geng, Nika Jafar Nia, and Aoxi Li.

# Using this repository.

The code to run our implementation of Fairer-NMF is in `fairnmf.py`.  Use the `FairNMF_MU` and `FairNMF_AM` for the multiplicative update and alternating minimization versions, respectively.

To reproduce the experiments in the paper, make sure to install the appropriate packages (numpy, pandas, sklearn, matplotlib, nltk, and tqdm) and obtain the datasets.  The 20news dataset is included in scikit-learn, and the heart disease dataset can be downloaded from the following link:

https://archive.ics.uci.edu/dataset/45/heart+disease

Use the `processed.cleveland.data` file and make sure it is in the same folder as `experiments.ipynb`.

Then, run the notebook `experiments.ipynb` to generate the plots after (optionally) running the experiments.  By default, this file loads the experimental data used to create the paper.

When `experiments.ipynb` is ran, the running time is saved as well.  The file `time_plots.ipynb` loads this data and creates the time plots as shown in the paper.