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
from typing import Optional
from numpy.typing import ArrayLike

from anndata import AnnData
from networkx import DiGraph


class CellAnnotation:

    #=====================================
    def __init__(
            self,
            graph: DiGraph,
            adata: AnnData,
            output: str,
    ):
        """
        """
        self.G = graph
        self.output = output
        self.A = adata

        self.cells_to_be_eliminated = None

    #=====================================
    def homogenize_leaf_nodes(
            self,
            cell_ann_col: str = "cell_annotations",
            upper_threshold: float = 0.80,
            change_below_this: float = 0.50,
            max_n_up_steps: int = 1,
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

        #Iterate over all nodes.
        for node in self.G.nodes:

            if 0 < self.G.out_degree(node):
                #Not a leaf node.
                continue

            majority = self.find_majority_from_node(
                node,
                CA,
                upper_threshold,
                max_n_up_steps,
                labels_to_change,
            )

            if majority is None:
                print(f"No majority found at {node=}.")
                continue
                # raise ValueError("No majority candidate.")


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

        return self.A

    #=====================================
    def find_majority_from_node(
            self,
            starting_node: int,
            cell_ann_col: str = "cell_annotations",
            threshold: float = 0.80,
            max_n_up_steps: float = np.inf,
            cell_types_to_avoid: List[str] = [],
        ) -> Optional[str]:
        """
        Find an ancestor node whose majority is
        above the threshold.
        """

        from networkx import descendants as nx_descendants

        CA = cell_ann_col

        level = 0
        best_candidate_label = None
        best_candidate_ratio = 0
        best_candidate_node  = -1

        node = starting_node

        while level <= max_n_up_steps:

            if 0 < self.G.out_degree(node):
                #This is not a leaf node.
                descendants = nx_descendants(self.G, node)
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
                    return child_majority

            #Find a best candidate in case we 
            #are not able to satisfy all the
            #conditions. Notice that we iterate
            #over the cell types because some 
            #of them could be in the list of 
            #cell types to avoid.
            for cell_type, ratio in vc.items():

                # print(f"-----------")
                # print(f"{node=}")
                # print(f"{cell_type=}")
                # print(f"{ratio=}")
                # print(f"-----------")

                if cell_type in cell_types_to_avoid:
                    continue

                #If we are here that means that
                #we have a valid cell type. Since
                #the ratios are decreasing, either
                #this cell type is above the ratio
                #or none are.
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
        # return best_candidate_label
        return None

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

    #=====================================
    def annotate_using_tree(
            self,
            cell_group_path: str,
            cell_marker_path: str,
            cell_ann_col: str = "cell_annotations",
            clean_threshold: float = 0.8,
            favor_minorities: bool = False,
            conversion_threshold: float = 0.9,
            confirmation_threshold: float = 0.9,
            elimination_ratio: float = -1.,
            homogeneous_leafs: bool = False,
            follow_parent: bool = False,
            follow_majority: bool = False,
            no_mixtures: bool = False,
    ):
        """
        Use the tree structure with the current labels
        to improve the cell annotation.
        TODO: Test this function
        """
        from collections import defaultdict as ddict
        from networkx import descendants as nx_descendants

        f_name = "compute_marker_median_value_for_cell_type"
        cmmv_for_cell_type = getattr(self, f_name)

        if not os.path.exists(cell_group_path):
            print(cell_group_path)
            raise ValueError("File does not exists.")

        if not os.path.exists(cell_marker_path):
            print(cell_marker_path)
            raise ValueError("File does not exists.")

        if homogeneous_leafs:
            if follow_majority == follow_parent:
                print("Homogeneous leafs strategy:")
                raise ValueError("Strategy is not unique.")
        
        CA = cell_ann_col

        # Make the connection between the cell groups and the 
        # cell types.
        self.load_group_and_cell_type_data(cell_group_path)

        # Make the connection between the markers and the 
        # cell types.
        self.load_marker_and_cell_type_data(cell_marker_path)

        self.marker_to_median_value_for_cell_type = ddict(
            dict)


        #Define an iterator.
        it = self.marker_to_cell_types.items()
        for marker, cell_types in it:
            for cell_type in cell_types:
                #In case some cell types have been erased.
                if cell_type not in self.cell_type_to_group:
                    continue

                #Note that we ignore the zeros for the
                #median.  This is to require higher standards
                #for a cell to classified as a member of a
                #given cell type.
                x = cmmv_for_cell_type(
                    marker,
                    cell_type,
                    ignore_zero=True,
                )
                self.marker_to_median_value_for_cell_type[
                    marker][cell_type] = x

        #Eliminate cells that belong to the erase category.
        if 0 < len(self.cell_types_to_erase):
            mask = self.A.obs[CA].isin(
                self.cell_types_to_erase)
            n_cells = mask.sum()
            vc = self.A.obs[CA].loc[mask].value_counts()
            #Take the complement of the cells we 
            #want to erase.
            self.A = self.A[~mask].copy()
            print("===============================")
            print(f"{n_cells} cells have been deleted.")
            print(vc)

        #Create a series where the original cell 
        #annotations have been mapped to their 
        #corresponding group.

        #To allow modifications to the series.
        #Categories cannot be directly modified.
        S = self.A.obs[CA].astype(str)

        for cell, group in self.cell_type_to_group.items():

            if cell == group:
                continue

            mask = S == cell
            S.loc[mask] = group

        S = S.astype("category")
        OCA = "original_cell_annotations"
        self.A.obs[OCA] = self.A.obs[CA].copy()
        self.A.obs[CA] = S
        vc = self.A.obs[CA].value_counts()
        print("===============================")
        print("Relabeled cell counts")
        print(vc)

        node = 0
        parent_majority = None
        parent_ratio = None
        # We use a deque to do a breadth-first traversal.
        from collections import deque
        DQ = deque()

        T = (node, parent_majority, parent_ratio)
        DQ.append(T)

        iteration = 0

        # Elimination container
        elim_set = set()

        self.labels_have_changed = False

        MNL = "majority_node_label"
        self.A.obs[MNL] = ""


        while 0 < len(DQ):
            print("===============================")
            T = DQ.popleft()
            node, parent_majority, parent_ratio = T
            children = self.G.successors(node)
            nodes = nx_descendants(self.G, node)
            is_leaf_node = False
            if len(nodes) == 0:
                is_leaf_node = True
                nodes = [node]
            else:
                x = self.set_of_leaf_nodes.intersection(
                    nodes)
                nodes = list(x)

            mask = self.A.obs["sp_cluster"].isin(nodes)
            S = self.A.obs[CA].loc[mask]
            node_size = mask.sum()
            print(f"Working with {node=}")
            print(f"Size of {node=}: {node_size}")
            vc = S.value_counts(normalize=True)
            print("===============================")
            print(vc)

            majority_group = vc.index[0]
            majority_ratio = vc.iloc[0]

            if majority_ratio == 1:
                #The cluster is homogeneous.
                #Nothing to do here.
                continue


            if majority_ratio < clean_threshold:
                #We are below clean_threshold, so we add 
                #these nodes to the deque for 
                #further processing.
                print("===============================")
                for child in children:
                    print(f"Adding node {child} to DQ.")
                    T = (child,
                         majority_group,
                         majority_ratio)
                    DQ.append(T)
            else:
                #We are above the cleaning threshold. 
                #Hence, we can star cleaning this node.
                print("===============================")
                print(f"Cleaning {node=}.")
                print(f"{majority_group=}.")
                print(f"{majority_ratio=}.")

                self.A.obs.loc[mask, MNL] = majority_group

                if no_mixtures:
                    #We do not allow minorities.
                    mask = S != majority_group
                    Q = S.loc[mask]
                    elim_set.update(Q.index)
                    continue

                #We are going to iterate over all the 
                #groups below the majority group.
                #We call these the minority_groups.

                #We have two options. Start checking if
                #the minority actually belongs to the 
                #majority or first check if the minority
                #is indeed a true minority.
                iter = vc.iloc[1:].items()
                for minority_group, minority_ratio in iter:


                    #These are the cells that belong to one
                    #of the minorities. We label them as
                    #Q because their current status 
                    #is under question.
                    mask = S == minority_group
                    minority_size = mask.sum()
                    if minority_size == 0:
                        #Nothing to be done with this and 
                        #subsequent minorities because the
                        #cell ratios are sorted in 
                        #decreasing order. If one is zero,
                        #the rest are zero too.
                        break
                    Q = S.loc[mask]

                    if minority_ratio < elimination_ratio:
                        #If the ratio is below the 
                        #given threshold, then we 
                        #remove these cells.
                        elim_set.update(Q.index)
                        continue

                    #Check membership
                    if favor_minorities:
                        #We first check if the minority is
                        #indeed a true minority.
                        x=self.check_if_cells_belong_to_group(
                            Q, 
                            minority_group, 
                            conversion_threshold,
                            cell_ann_col,
                        )
                        belongs_to_minority = x
                        if belongs_to_minority:
                            #Move to the next minority.
                            continue
                        #Otherwise, check if belongs to 
                        #the majority group.
                        x=self.check_if_cells_belong_to_group(
                            Q, 
                            majority_group, 
                            conversion_threshold,
                            cell_ann_col,
                        )
                        identity_was_determined = x
                        belongs_to_majority = x

                        if belongs_to_majority:
                            self.labels_have_changed = True

                    else:
                        #We first check if the minority is
                        #actually part of the majority.
                        x=self.check_if_cells_belong_to_group(
                            Q, 
                            majority_group, 
                            conversion_threshold,
                            cell_ann_col,
                        )
                        belongs_to_majority = x
                        if belongs_to_majority:
                            #Move to the next minority.
                            self.labels_have_changed = True
                            continue
                        #Otherwise, check if belongs to 
                        #the minority group.
                        x=self.check_if_cells_belong_to_group(
                            Q, 
                            minority_group, 
                            conversion_threshold,
                            cell_ann_col,
                        )
                        identity_was_determined = x

                    if identity_was_determined:
                        #Nothing to be done.
                        #Move to the next minority.
                        continue
                    else:
                        #Cells could not be classified
                        #and therefore will be eliminated.
                        elim_set.update(Q.index)


            if iteration == 1:
                pass
                #break
            else:
                iteration += 1
            

        #Elimination phase 1
        print("Elimination set size before homogenization:",
              len(elim_set))

        #Homogenization
        if homogeneous_leafs:

            if follow_parent:
                print("Using parent node majority.")

            if follow_majority:
                print("Using leaf node majority.")
            
            S = self.homogenize_leaf_nodes(
                CA,
                follow_parent,
                follow_majority)

            if 0 < len(S):
                print("Cells lost through homogenization:",
                    len(S))
                elim_set.update(S)

        #If there are cells to be eliminated, then
        #we label them with an X.
        if 0 < len(elim_set):
            print("Total cells lost:", len(elim_set))
            remaining_cells = self.A.X.shape[0]
            remaining_cells -= len(elim_set)
            print("Remaining cells:", remaining_cells)

            #Create a new category.
            x = self.A.obs[CA].cat.add_categories("X")
            self.A.obs[CA] = x
            #Label the cells to be eliminated with "X".
            mask = self.A.obs_names.isin(elim_set)
            self.A.obs[CA].loc[mask] = "X"

            self.labels_have_changed = True

        if self.labels_have_changed:
            self.generate_cell_annotation_file(
                cell_ann_col=CA, tag = "updated_cell_labels")
        else:
            print("Nothing has changed.")

        #This set constains the cells to be eliminated
        #and can be used for subsequent processing 
        #in other functions.

        self.cells_to_be_eliminated = elim_set

    #=====================================
    def find_stable_tree(
            self,
            cell_group_path: str,
            cell_marker_path: str,
            cell_ann_col: str = "cell_annotations",
            clean_threshold: float = 0.8,
            favor_minorities: bool = False,
            conversion_threshold: float = 0.9,
            confirmation_threshold: float = 0.9,
            elimination_ratio: float = -1,
            homogeneous_leafs: bool = False,
            follow_parent: bool = False,
            follow_majority: bool = False,
            no_mixtures: bool = False,
            storage_path: str = "stable_tree",
            max_n_iter: int = 100,
    ):
        """
        This function will identify outliers in the
        cell annotation labels based on the main branches
        and subsequently will remove those outliers and
        recompute the tree until no more outliers are found.
        """
        CA = cell_ann_col

        # TODO: Import TMC.
        tmc_obj = TooManyCells(self, storage_path)

        something_has_changed = False
        iteration = 0

        while iteration < max_n_iter:

            tmc_obj.annotate_using_tree(
            cell_group_path,
            cell_marker_path,
            cell_ann_col,
            clean_threshold,
            favor_minorities,
            conversion_threshold,
            confirmation_threshold,
            elimination_ratio,
            homogeneous_leafs,
            follow_parent,
            follow_majority,
            no_mixtures,
            )

            iteration += 1

            if not tmc_obj.labels_have_changed:
                #No cells have changed their label
                #and no cell has been tagged for 
                #elimination.
                print("Nothing has changed.")
                break

            something_has_changed = True

            #We know the labels have changed.
            #We will only recompute the tree if 
            #cells have been eliminated.

            S = tmc_obj.cells_to_be_eliminated

            if 0 == len(S):
                print("No cells have been eliminated.")
                break

            #Cells have been eliminated.
            #A new tree will be generated with the
            #remaining cells.
            mask = tmc_obj.A.obs_names.isin(S)
            A = tmc_obj.A[~mask].copy()
            tmc_obj = TooManyCells(A, storage_path)
            tmc_obj.run_spectral_clustering()
            tmc_obj.store_outputs()

        if something_has_changed:
            print(f"{iteration=}")
    #=====================================END OF CLASS
            
