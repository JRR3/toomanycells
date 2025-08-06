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
import re
import sys
import json
import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx
from typing import Set
from typing import List
from typing import Union
from os.path import dirname
from typing import Optional
from collections import deque

sys.path.insert(0, dirname(__file__))
from common import MultiIndexList
from common import JEncoder

class TMCGraph:
    #=====================================
    def __init__(self,
                 graph: nx.DiGraph,
                 adata: sc.AnnData,
                 output: str,
        ):

        self.G = graph
        self.A = adata
        self.output = output


        # In case the graph is not empty, 
        # populate the set_of_leaf_nodes().
        self.find_leaf_nodes()

        self.J = MultiIndexList()

        #Map a node to the path in the
        #binary tree that connects the
        #root node to the given node.
        self.node_to_path = {}

        #Map a node to a list of indices
        #that provide access to the JSON
        #structure.
        self.node_to_j_index = {}

        self.node_counter = 0



    #=====================================
    def find_leaf_nodes(self):
        """
        Find all leaf nodes in the graph.

        This function clears and then populates the
        attribute set_of_leaf_nodes.
        """

        self.set_of_leaf_nodes = set()

        for node in self.G.nodes():

            # not_leaf_node = 0 < self.G.out_degree(node)
            is_leaf_node =  (0 == self.G.out_degree(node))

            if is_leaf_node:
                self.set_of_leaf_nodes.add(node)

    #=====================================
    def eliminate_cell_type_outliers(
            self,
            cell_ann_col: Optional[str] = "cell_annotations",
            clean_threshold: Optional[float] = 0.8,
            no_mixtures: Optional[bool] = True,
            batch_ann_col: Optional[str] = "",
    ):
        """
        Eliminate all cells that do not belong to the
        majority.
        """

        self.find_leaf_nodes()

        CA =cell_ann_col
        node = 0
        parent_majority = None
        parent_ratio = None
        # We use a deque to do a breadth-first traversal.
        DQ = deque()

        T = (node, parent_majority, parent_ratio)
        DQ.append(T)

        iteration = 0

        # Elimination container
        elim_set = set()
        self.set_of_red_clusters = set()

        while 0 < len(DQ):
            print("===============================")
            T = DQ.popleft()
            node, parent_majority, parent_ratio = T

            not_leaf_node = 0 < self.G.out_degree(node)
            is_leaf_node = not not_leaf_node

            children = self.G.successors(node)

            if is_leaf_node:
                nodes = [node]
            else:
                nodes = nx.descendants(self.G, node)
                x = self.set_of_leaf_nodes.intersection(
                    nodes)
                # nodes = list(x)

            mask = self.A.obs["sp_cluster"].isin(x)
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

                if no_mixtures:
                    #We do not allow minorities.
                    mask = S != majority_group
                    Q = S.loc[mask]
                    elim_set.update(Q.index)
                    continue

        print(f"Cells to eliminate: {len(elim_set)}")
        self.cells_to_be_eliminated = elim_set

        #List of cell ids to be eliminated.
        ES = list(elim_set)

        #Cell types of cells to be eliminated.
        cell_labels = self.A.obs[CA].loc[ES]

        #Batches containing cells to be eliminated.
        if 0 < len(batch_ann_col):
            batch_labels = self.A.obs[
                batch_ann_col
            ].loc[ES]

            #Batch origin quantification.
            batch_vc = batch_labels.value_counts()
            print(batch_vc)

        #Clusters containing cells to be eliminated.
        cluster_labels = self.A.obs["sp_cluster"].loc[ES]

        #Cell type quatification.
        cell_vc = cell_labels.value_counts()


        #Cluster quantification.
        cluster_vc = cluster_labels.value_counts()

        # We then compare against the original number
        # of cells in each cluster.
        cluster_ref = self.A.obs["sp_cluster"].value_counts()

        print(cell_vc)
        print(cluster_vc)

        #Compare side-by-side the cells to be eliminated
        #for each cluster with the total number of cells
        #for that cluster.
        df = pd.merge(cluster_ref, cluster_vc,
                      left_index=True, right_index=True,
                      how="inner")

        df["status"] = df.count_x == df.count_y

        #Clusters to be eliminated
        red_clusters = df.index[df["status"]]

        self.set_of_red_clusters = set(red_clusters)

        ids_to_erase = self.A.obs.index.isin(elim_set)

        #Create a new AnnData object after eliminating 
        #the cells.
        self.A = self.A[~ids_to_erase].copy()

    #=====================================
    def rebuild_graph_after_removing_cells(
            self,
    ):
        """
        TODO
        """

        DQ = deque()
        DQ.append(0)

        while 0 < len(DQ):

            # (gp_node_id, p_node_id,
            #  s_node_id, node_id) = S.pop()
            node_id = DQ.popleft()
            cluster_size = self.G.nodes[node_id]["size"]
            not_leaf_node = 0 < self.G.out_degree(node_id)
            is_leaf_node = not not_leaf_node

            flag_to_erase = False
            if is_leaf_node:
                if node_id in self.set_of_red_clusters:
                    flag_to_erase = True
            else:
                nodes = nx.descendants(self.G, node_id)
                mask = self.A.obs["sp_cluster"].isin(nodes)
                n_viable_cells = mask.sum()
                if n_viable_cells == 0:
                    flag_to_erase = True

            p_node_id = self.get_parent_node(node_id)
            gp_node_id = self.get_grandpa_node(node_id)
            s_node_id = self.get_sibling_node(node_id)
            
            if flag_to_erase:

                #Connect the grandpa node to the sibling node
                self.G.add_edge(gp_node_id, s_node_id)

                #Remove the edge between the parent node
                #and the sibling node.
                self.G.remove_edge(p_node_id, s_node_id)

                #Remove the parent node.
                self.G.remove_node(p_node_id)
                print(f"Removed {p_node_id=}")

                if not_leaf_node:
                    #Remove all descendants
                    self.G.remove_nodes_from(nodes)

                #Remove the current node.
                self.G.remove_node(node_id)
                print(f"Removed {node_id=}")
                continue

            #No elimination took place.
            children = self.G.successors(node_id)
            for child in children:
                DQ.append(child)

    #=====================================
    def get_parent_node(self, node: int) -> int:
        """
        """

        if node is None:
            return None

        it = self.G.predecessors(node)
        parents = list(it)

        if len(parents) == 0:
            return None

        return parents[0]

    #=====================================
    def get_grandpa_node(self, node: int) -> int:
        """
        """

        parent  = self.get_parent_node(node)
        grandpa =  self.get_parent_node(parent)

        return grandpa

    #=====================================
    def get_sibling_node(self, node: int) -> int:
        """
        """

        parent  = self.get_parent_node(node)

        if parent is None:
            return None

        children = self.G.successors(parent)

        for child in children:

            if child != node:
                return child

        return None

    #=====================================
    def rebuild_tree_from_graph(
            self,
    ):
        """
        TODO
        """
        S      = []
        self.J = MultiIndexList()
        node_id= 0

        self.node_to_j_index = {}
        self.node_to_j_index[node_id] = (1,)

        Q = self.G.nodes[node_id]["Q"]
        D = self.modularity_to_json(Q)

        self.J.append(D)
        self.J.append([])

        children = self.G.successors(node_id)

        # The largest index goes first so that 
        # when we pop an element, we get the smallest
        # of the two that were inserted.
        children = sorted(children, reverse=True)
        for child in children:
            T = (node_id, child)
            S.append(T)

        while 0 < len(S):

            p_node_id, node_id = S.pop()
            cluster_size = self.G.nodes[node_id]["size"]
            not_leaf_node = 0 < self.G.out_degree(node_id)
            is_leaf_node = not not_leaf_node

            nodes = nx.descendants(self.G, node_id)
            mask = self.A.obs["sp_cluster"].isin(nodes)
            n_viable_cells = mask.sum()

            j_index = self.node_to_j_index[p_node_id]
            n_stored_blocks = len(self.J[j_index])
            self.J[j_index].append([])
            #Update the j_index. For example, if
            #j_index = (1,) and no blocks have been
            #stored, then the new j_index is (1,0).
            #Otherwise, it is (1,1).
            j_index += (n_stored_blocks,)

            if not_leaf_node:
                #This is not a leaf node.
                Q = self.G.nodes[node_id]["Q"]
                D = self.modularity_to_json(Q)
                self.J[j_index].append(D)
                self.J[j_index].append([])
                j_index += (1,)
                self.node_to_j_index[node_id] = j_index
                children = self.G.successors(node_id)
                children = sorted(children, reverse=True)
                for child in children:

                    T = (node_id, child)
                    S.append(T)
            else:
                #Leaf node
                mask = self.A.obs["sp_cluster"] == node_id
                rows = np.nonzero(mask)[0]
                L = self.cells_to_json(rows)
                self.J[j_index].append(L)
                self.J[j_index].append([])

    #=====================================
    def rebuild_tree_without_rearrangements(
            self,
    ):
        """
        To show the stubs you need to modify the
        """

        S      = []
        self.J = MultiIndexList()
        node_id= 0

        self.node_to_j_index = {}
        self.node_to_j_index[node_id] = (1,)

        Q = self.G.nodes[node_id]["Q"]
        D = self.modularity_to_json(Q)

        self.J.append(D)
        self.J.append([])

        children = self.G.successors(node_id)

        # The largest index goes first so that 
        # when we pop an element, we get the smallest
        # of the two that were inserted.
        children = sorted(children, reverse=True)
        for child in children:
            T = (node_id, child)
            S.append(T)

        while 0 < len(S):

            p_node_id, node_id = S.pop()
            cluster_size = self.G.nodes[node_id]["size"]
            not_leaf_node = 0 < self.G.out_degree(node_id)
            is_leaf_node = not not_leaf_node

            nodes = nx.descendants(self.G, node_id)
            mask = self.A.obs["sp_cluster"].isin(nodes)
            n_viable_cells = mask.sum()

            # Non-leaf nodes with zero viable cells
            # are eliminated.
            if not_leaf_node and n_viable_cells == 0:
                # print(f"Cluster {node_id} has to "
                #       "be eliminated.")
                continue

            if node_id in self.set_of_red_clusters:
                continue

            j_index = self.node_to_j_index[p_node_id]
            n_stored_blocks = len(self.J[j_index])
            self.J[j_index].append([])
            #Update the j_index. For example, if
            #j_index = (1,) and no blocks have been
            #stored, then the new j_index is (1,0).
            #Otherwise, it is (1,1).
            j_index += (n_stored_blocks,)

            if not_leaf_node:
                #This is not a leaf node.
                Q = self.G.nodes[node_id]["Q"]
                D = self.modularity_to_json(Q)
                self.J[j_index].append(D)
                self.J[j_index].append([])
                j_index += (1,)
                self.node_to_j_index[node_id] = j_index
                children = self.G.successors(node_id)
                children = sorted(children, reverse=True)
                for child in children:

                    T = (node_id, child)
                    S.append(T)
            else:
                #Leaf node
                mask = self.A.obs["sp_cluster"] == node_id
                rows = np.nonzero(mask)[0]
                L = self.cells_to_json(rows)
                self.J[j_index].append(L)
                self.J[j_index].append([])

    #=====================================
    def modularity_to_json(self,Q):
        return {'_item': None,
                '_significance': None,
                '_distance': Q}

    #=====================================
    def cell_to_json(self, cell_name, cell_number):
        return {'_barcode': {'unCell': cell_name},
                '_cellRow': {'unRow': cell_number}}

    #=====================================
    def cells_to_json(self,rows):
        L = []
        for row in rows:
            cell_id = self.A.obs.index[row]
            D = self.cell_to_json(cell_id, row)
            L.append(D)
        return {'_item': L,
                '_significance': None,
                '_distance': None}

    #=====================================
    def convert_graph_to_tmc_json(self):
        """
        The graph structure stored in the attribute\
            self.J has to be formatted into a \
            JSON file. This function takes care\
            of that task. The output file is \
            named 'cluster_tree.json' and is\
            equivalent to the 'cluster_tree.json'\
            file produced by too-many-cells.
        """

        fname = "cluster_tree.json"
        fname = os.path.join(self.output, fname)

        with open(fname,"w",encoding="utf-8") as output_file:
            json.dump(
                self.J,
                output_file,
                cls=JEncoder,
                ensure_ascii=False,
                separators=(",", ":"),
                )

    #=====================================
    def write_cell_assignment_to_csv(self):
        """
        This function creates a CSV file that indicates \
            the assignment of each cell to a specific \
            cluster. The first column is the cell id, \
            the second column is the cluster id, and \
            the third column is the path from the root \
            node to the given node.
        """
        fname = 'clusters.csv'
        fname = os.path.join(self.output, fname)
        labels = ['sp_cluster','sp_path']
        df = self.A.obs[labels]
        df.index.names = ['cell']
        df = df.rename(columns={'sp_cluster':'cluster',
                                'sp_path':'path'})
        df.to_csv(fname, index=True)

    #=====================================
    def write_cluster_list_to_tmc_json(self):
        """
        This function creates a JSON file that indicates \
            the assignment of each cell to a specific \
            cluster. 
        """
        master_list = []
        relevant_cols = ["sp_cluster", "sp_path"]
        df = self.A.obs[relevant_cols]
        df = df.reset_index(names="cell")
        df = df.sort_values(["sp_cluster","cell"])
        for idx, row in df.iterrows():
            cluster = row["sp_cluster"]
            path_str= row["sp_path"]
            cell    = row["cell"]
            nodes = path_str.split("/")
            list_of_nodes = []
            sub_dict_1 = {"unCell":cell}
            sub_dict_2 = {"unRow":idx}
            main_dict = {"_barcode":sub_dict_1,
                         "_cellRow":sub_dict_2}
            for node in nodes:
                d = {"unCluster":int(node)}
                list_of_nodes.append(d)
            
            master_list.append([main_dict, list_of_nodes])

        fname = "cluster_list.json"
        fname = os.path.join(self.output, fname)
        with open(fname,"w",encoding="utf-8") as output_file:
            json.dump(
                master_list,
                output_file,
                cls=JEncoder,
                ensure_ascii=False,
                separators=(",", ":"),
                )

        # s = str(master_list)
        # replace_dict = {" ":"", "'":'"'}
        # pattern = "|".join(replace_dict.keys())
        # regexp  = re.compile(pattern)
        # fun = lambda x: replace_dict[x.group(0)] 
        # obj = regexp.sub(fun, s)
        # with open(fname, "w") as output_file:
        #     output_file.write(obj)

    #=====================================
    def convert_graph_to_json(self):
        """
        The graph is stored in the JSON format.
        """
        # Write graph "self.G" to JSON file.
        nld = nx.node_link_data(self.G)
        fname = "graph.json"
        fname = os.path.join(self.output, fname)
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(nld, f, ensure_ascii=False, indent=4)

    #=====================================
    def store_outputs(self,
                      store_in_uns_dict = False):
        """
        """
        self.write_cell_assignment_to_csv()
        self.convert_graph_to_tmc_json()
        self.convert_graph_to_json()
        self.write_cluster_list_to_tmc_json()

        if store_in_uns_dict:
            #The directed graph is stored in the dict.
            self.A.uns["tmc_graph"] = self.G
            x = self.set_of_leaf_nodes
            #The list of leaf nodes is stored in the dict.
            self.A.uns["tmc_leaf_nodes"] = x
            S = json.dumps(
                self.J,
                cls=JEncoder,
                ensure_ascii=False,
                separators=(",", ":"),
                )
            #The cluster_tree.json is stored in the
            #dictionary as a string.
            self.A.uns["tmc_json"] = S

    #=====================================
    def load_graph(
            self,
            json_fname: str = "graph.json",
            load_clusters_file: bool = True,
        ):
        """
        Load the JSON file. Note that when loading the data,
        the attributes of each node are assumed to be 
        strings. Hence, we have to convert them.
        We use int(x) for the number of cells, and float(x) 
        for the modularity.
        """

        json_fname = os.path.join(self.output, json_fname)

        if not os.path.exists(json_fname):
            raise ValueError("File does not exists.")

        # Avoid dependencies with GraphViz
        # dot_fname = "graph.dot"
        # dot_fname = os.path.join(self.output, dot_fname)
        # self.G = nx.nx_agraph.read_dot(dot_fname)

        print("Reading JSON file ...")

        with open(json_fname, encoding="utf-8") as f:
            json_graph = json.load(f)
        self.G = nx.node_link_graph(json_graph)
        
        print("Finished reading JSON file.")

        n_nodes = self.G.number_of_nodes()

        # Change string labels to integers.
        D = {}
        for k in range(n_nodes):
            D[str(k)] = k

        self.G = nx.relabel_nodes(self.G, D, copy=True)

        #We convert the number of cells of each node to
        #integer. We also convert the modularity to float.
        #Lastly, we populate the set of leaf nodes.
        for node in self.G.nodes():

            not_leaf_node = 0 < self.G.out_degree(node)
            is_leaf_node = not not_leaf_node

            if is_leaf_node:
                self.set_of_leaf_nodes.add(node)

            size = self.G.nodes[node]["size"]
            self.G.nodes[node]["size"] = int(size)
            if "Q" in self.G.nodes[node]:
                Q = self.G.nodes[node]["Q"]
                self.G.nodes[node]["Q"] = float(Q)

        if load_clusters_file:
            fname = "clusters.csv"
            fname = os.path.join(self.output, fname)
            df = pd.read_csv(fname,
                             header=0,
            )
            cell_ids = df["cell"].values
            clusters = df["cluster"].values
            #By default, the constructor of an 
            #AnnData object will produce indices of 
            #string type.
            if "int" in str(type(cell_ids[0])):
                cell_ids = map(str, cell_ids)
            # print(list(cell_ids))
            # print(self.A.obs.index)
            self.A.obs.loc[cell_ids,"sp_cluster"] = clusters

        print(self.G)

    #=====================================
    def isolate_cells_from_branches(
            self,
            path_to_csv_file: str = "",
            list_of_branches: List[int] = [],
            branch_column: str = "node",
        ):
        """
        This function produces a mask of booleans
        that indicate if a cell belongs or not
        to a leaf node contained in 
        one of the branches.
        """

        if 0 < len(path_to_csv_file):
            #This file contains all the branches
            df = pd.read_csv(path_to_csv_file, header=0)
            list_of_branches = df[branch_column].values

        elif 0 < len(list_of_branches):
            pass

        else:
            raise ValueError("No source has been specified.")

        set_of_leaf_nodes = set()
        sp_cluster = "sp_cluster"

        # print(f"{list_of_branches=}")

        for branch in list_of_branches:

            if 0 < self.G.out_degree(branch):
                #Not a leaf node.
                nodes = nx.descendants(self.G, branch)
                nodes = self.set_of_leaf_nodes.intersection(
                    nodes)
                set_of_leaf_nodes.update(nodes)
            else:
                #Is a leaf node
                set_of_leaf_nodes.add(branch)

        mask = self.A.obs[sp_cluster].isin(
            set_of_leaf_nodes)

        return mask

    #====END=OF=CLASS=====================