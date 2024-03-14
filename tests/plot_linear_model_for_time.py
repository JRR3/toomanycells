import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 600
mpl.use("agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from path import Path
import numpy as np
from os.path import dirname
from sklearn.linear_model import LinearRegression as LR

pth = Path(dirname(__file__))
fname = pth / "benchmark_table_mod.csv"
df = pd.read_csv(fname)
print(df)
df["T"] /= 60
fig, ax = plt.subplots()
sb.scatterplot(df,
               x="N",
               y="T",
               color="blue",
               ax=ax)
fname = pth / "log_linear_time.png"
plt.yscale("log")
plt.xscale("log")
plt.grid(True, which="both")

X = np.log10(df["N"].values.reshape(-1,1))
y = np.log10(df["T"])
t_model = LR().fit(X, y)
t_model_score = t_model.score(X, y)
print(f"{t_model_score=}")
print(f"{t_model.coef_=}")
print(f"{t_model.intercept_=}")

t_vec = np.array([5_000, 500_000])
k = np.power(10, t_model.intercept_)
xp= t_model.coef_[0]
y_vec = k * np.power(t_vec, t_model.coef_)
ax.plot(t_vec, y_vec, "r-", label="slope=1.08")
ax.set_xlabel("Number of cells")
ax.set_ylabel("Time (minutes)")
ax.legend(loc="upper left")

fig.savefig(fname, bbox_inches="tight")