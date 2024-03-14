import sys
import scanpy as sc
import pandas as pd
import numpy as np
from os.path import dirname
from path import Path
from sklearn.linear_model import LinearRegression as LR

pth = Path(dirname(dirname(__file__))) / "toomanycells"
sys.path.insert(0, pth)
from toomanycells import TooManyCells as tmc

source = "/home/javier/Documents/ailles/data/merged"
fname = Path(source) / "matrix.h5ad"

A = sc.read_h5ad(fname)
samples = A.obs["batch"].unique().tolist()
n_iter_list = []
t_elap_list = []
n_cell_list = []
for k in range(0,len(samples)):
    if k+1 != len(samples):
        continue
    L = samples[:k+1]
    print(L)
    print(f"{len(L)=}")
    mask = A.obs["batch"].isin(L)
    B = A[mask].copy()
    tmc_obj = tmc(B)
    tmc_obj.run_spectral_clustering()
    t_elap = tmc_obj.delta_clustering
    n_iter = tmc_obj.final_n_iter
    n_cell = B.shape[0]
    print(f"{t_elap=}")
    print(f"{n_iter=}")
    print(f"{n_cell=}")
    n_iter_list.append(n_iter)
    t_elap_list.append(t_elap)
    n_cell_list.append(n_cell)
    print("==================")

exit()

D = {"N": n_cell_list, "T":t_elap_list, "I":n_iter_list}
df = pd.DataFrame(D)
local_path = dirname(__file__)

fname = Path(local_path) / "benchmark_table.csv"
df.to_csv(fname, index=False)

df = df.apply(np.log)
fname = Path(local_path) / "log_benchmark_table.csv"
df.to_csv(fname, index=False)

X = df["N"].values.reshape(-1,1)
t_model = LR().fit(X, df["T"])
t_model_score = t_model.score(X, df["T"])
print(f"{t_model_score=}")
print(f"{t_model.coef_=}")
print(f"{t_model.intercept_=}")

i_model = LR().fit(X, df["I"])
i_model_score = i_model.score(X, df["I"])
print(f"{i_model_score=}")
print(f"{i_model.coef_=}")
print(f"{i_model.intercept_=}")
