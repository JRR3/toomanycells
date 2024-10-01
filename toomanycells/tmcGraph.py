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
import pandas as pd
import scanpy as sc
import networkx as nx
from typing import Union
from typing import Optional
from collections import deque

class TMCGraph:
    #=====================================
    def __init__(self,
                 graph: nx.DiGraph,
                 adata: sc.AnnData,
        ) -> None:
        self.G = graph
        self.A = adata

    #=====================================
    def eliminate_cell_type_outliers(
            self,
            cell_ann_col: Optional[str] = "cell_annotations",
            clean_threshold: Optional[float] = 0.8,
            no_mixtures: Optional[bool] = True,
    ):
        """
        Eliminate all cells that do not belong to the
        majority.
        """
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
            children = self.G.successors(node)
            nodes = nx.descendants(self.G, node)
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
        batch_labels = self.A.obs["sample_id"].loc[ES]

        #Clusters containing cells to be eliminated.
        cluster_labels = self.A.obs["sp_cluster"].loc[ES]

        #Cell type quatification.
        cell_vc = cell_labels.value_counts()

        #Batch origin quantification.
        batch_vc = batch_labels.value_counts()

        #Cluster quantification.
        cluster_vc = cluster_labels.value_counts()

        # We then compare against the original number
        # of cells in each cluster.
        cluster_ref = self.A.obs["sp_cluster"].value_counts()

        print(cell_vc)
        print(batch_vc)
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
    def rebuild_graph(
            self,
    ):
        """
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

                #The parent of the sibling node
                #becomes the grandpa node
                self.G.nodes[s_node_id]["parent"] = gp_node_id

                #The parent of the grandpa node becomes 
                #the new grandpa of the sibling node.
                print(node_id, s_node_id, p_node_id, gp_node_id)
                # print(self.G.nodes[gp_node_id])
                new_gp = self.G.nodes[gp_node_id]["parent"]
                self.G.nodes[s_node_id]["grandpa"] = new_gp

                #The new sibling of the sibling node is the
                #sibling of the parent node.
                new_s = self.G.nodes[p_node_id]["sibling"]
                self.G.nodes[s_node_id]["sibling"] = new_s

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
            ch = list(children)

            self.G.nodes[ch[0]]["parent"] = node_id
            self.G.nodes[ch[0]]["grandpa"] = p_node_id
            self.G.nodes[ch[0]]["sibling"] = ch[1]
            DQ.append(ch[0])

            self.G.nodes[ch[1]]["parent"] = node_id
            self.G.nodes[ch[1]]["grandpa"] = p_node_id
            self.G.nodes[ch[1]]["sibling"] = ch[0]
            DQ.append(ch[1])

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