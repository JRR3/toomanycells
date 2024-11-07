#########################################################
#Princess Margaret Cancer Research Tower
#Schwartz Lab
#Javier Ruiz Ramirez
#October 2024
#########################################################
#Questions? Email me at: javier.ruizramirez@uhn.ca
#########################################################
import os
import scprep
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Union
from typing import List
from numpy.typing import ArrayLike

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["figure.dpi"] = 600
mpl.use("agg")

class MultiPlotter:

    #=====================================
    def __init__(
            self,
            h5ad_object: sc.AnnData,
            output: str
    ):
        """
        """
        self.A = h5ad_object
        self.output = output

    #=====================================
    def load_embedding(self, path: str):
        df = pd.read_csv(path, index_col=0)
        mat = df.to_numpy()
        return mat

    #=====================================
    def plot_embedding(self,
                       matrix: ArrayLike,
                       file_name: str,
                       colors: ArrayLike,
                       color_map: Union[dict, str],
    ):

        fig, ax = plt.subplots()

        if os.path.exists(color_map):
            df = pd.read_csv(color_map,
                             index_col=None,
                             header=0)
            color_map = {}
            for index, row in df.iterrows():
                cell_type = row["Cell"]
                color = row["Color"]
                color_map[cell_type] = color

        scprep.plot.scatter2d(
            matrix,
            ax=ax,
            c = colors,
            cmap = color_map,
            legend_loc = (1, 0.5),
            ticks=False,
        )
        #ax.set_aspect("equal")
        fname = file_name
        fname = os.path.join(self.output, fname)
        fig.savefig(fname, bbox_inches="tight")
    #=====================================
    def process_embedding(self,
                          path: str,
                          label: str,
                          column: str,
                          color_map: Union[dict, str],
    ):
        """
        """

        colors = self.A.obs[column]

        mtx = self.load_embedding(path)
        # print(mtx)

        self.plot_embedding(mtx,
                            label,
                            colors,
                            color_map,
        )