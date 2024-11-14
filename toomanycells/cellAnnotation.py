#########################################################
#Princess Margaret Cancer Research Tower
#Schwartz Lab
#Javier Ruiz Ramirez
#October 2024
#########################################################
#This is a Python script to produce TMC trees using
#the original too-many-cells tool.
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7439807/
#########################################################
#Questions? Email me at: javier.ruizramirez@uhn.ca
#########################################################
import os
import numpy as np
import pandas as pd
from typing import List
from typing import Tuple
import matplotlib as mpl
from typing import Optional
from numpy.typing import ArrayLike
import networkx as nx
from anndata import AnnData

class CellAnnotation:

    #=====================================
    def __init__(
            self,
            graph: nx.DiGraph,
            adata: AnnData,
            output: str,
    ):
        """
        """
        self.G = graph
        self.output = output
        self.A = adata

    #=====================================
    def homogenize_leaf_nodes(
            self,
            cell_ann_col: str = "cell_annotations",
            upper_threshold: float = 0.80,
            change_below_this: float = 0.50,
            minimum_level: int = 0,
            change_all: bool = True,
            labels_to_change: List[str] = [],
    ):
        """
        How should we homogenize a leaf node?
        Either we use the majority present in the
        parent node, or we use the majority already
        present in the leaf node.
        """

        CA = cell_ann_col
        nodes_to_relabel = set()

        for node in self.G.nodes:

            if 0 < self.G.out_degree(node):
                #Not a leaf node.
                continue

            majority = self.find_majority_from_node(
                node,
                CA,
                upper_threshold,
                minimum_level,
                labels_to_change,
            )

            if majority is None:
                print(f"Error at {node=}")
                raise ValueError("No majority candidate.")


            #Child
            mask = self.A.obs["sp_cluster"].isin([node])
            S = self.A.obs[CA].loc[mask]
            vc = S.value_counts(normalize=True)

            for cell_type, ratio in vc.items():
                # If change all and ratio below lower
                # threshold.
                condition = change_all
                condition &= ratio < change_below_this
                # Or cell type is in the list of targets.
                condition |= cell_type in labels_to_change
                if condition:
                    mask = S.isin([cell_type])
                    indices = S.loc[mask].index
                    self.A.obs.loc[indices, CA] = majority

        fname = "homogenized_cell_annotations.csv"
        fname = os.path.join(self.output, fname)
        S = self.A.obs[CA]
        # To be consistent with TooManyCells interactive
        # labeling conventions for the cell annotations.
        S.index.name = "item"
        S.name       = "label"
        S.to_csv(fname, index=True)

    #=====================================
    def find_majority_from_node(
            self,
            starting_node: int,
            cell_ann_col: str = "cell_annotations",
            threshold: float = 0.80,
            minimum_level: int = 0,
            cell_types_to_avoid: List[str] = [],
    ) -> Optional[str]:
        """
        Find an ancestor node whose majority is
        above the threshold.
        """

        CA = cell_ann_col

        keep_looking = True
        level = -1
        best_candidate_label = None
        best_candidate_ratio = 0
        best_candidate_node  = -1

        node = starting_node

        while keep_looking:

            if 0 < self.G.out_degree(node):
                #This is not a leaf node.
                descendants = nx.descendants(self.G, node)
            else:
                descendants = [node]

            level += 1
            mask = self.A.obs["sp_cluster"].isin(descendants)
            S = self.A.obs[CA].loc[mask]
            vc = S.value_counts(normalize=True)
            child_majority = vc.index[0]
            child_ratio = vc.iloc[0]

            if threshold <= child_ratio:
                if child_majority in cell_types_to_avoid:
                    pass
                else:
                    if minimum_level <= level:
                        return child_majority

            #Find a best candidate in case we 
            #are not able to satisfy all the
            #conditions.
            for cell_type, ratio in vc.items():
                # print(f"-----------")
                # print(f"{node=}")
                # print(f"{cell_type=}")
                # print(f"{ratio=}")
                # print(f"-----------")
                if cell_type in cell_types_to_avoid:
                    continue
                if best_candidate_ratio < ratio:
                    best_candidate_ratio = ratio
                    best_candidate_label = cell_type
                    best_candidate_node  = node
                    break

            if 0 == self.G.in_degree(node):
                #This is a root node.
                break

            #Find the parent node.
            node = next(self.G.predecessors(node))

        print("Cell type not found.")
        print(f"{best_candidate_label=}")
        print(f"{best_candidate_ratio=}")
        print(f"{best_candidate_node=}")
        return best_candidate_label

    #=====================================
    def check_leaf_homogeneity(
            self,
            cell_ann_col: str = "cell_annotations",
    ):
        """
        Determine if all the leaf nodes are homogeneous.
        As soon as one heterogeneous node is found, the
        function returns False.
        """

        CA = cell_ann_col

        for node in self.G.nodes:
            if 0 < self.G.out_degree(node):
                #This is not a leaf node.
                continue

            #Child
            mask = self.A.obs["sp_cluster"].isin([node])
            S = self.A.obs[CA].loc[mask].unique()

            if len(S) == 1:
                #The node is already homogeneous
                continue
            else:
                #We found one leaf node that is not
                #homogeneous.
                self.leaf_nodes_are_homogeneous = False
                return False

        self.leaf_nodes_are_homogeneous = True

        return True
