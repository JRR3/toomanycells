#########################################################
#Princess Margaret Cancer Research Tower
#Schwartz Lab
#Javier Ruiz Ramirez
#July 2024
#########################################################
#This is a Python implementation of the command line 
#tool too-many-cells.
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7439807/
#########################################################
#Questions? Email me at: javier.ruizramirez@uhn.ca
#########################################################
import os
import re
import sys
import gzip
import json
import subprocess
import numpy as np
import scanpy as sc
import pandas as pd
# import seaborn as sb
from tqdm import tqdm
import networkx as nx
import matplotlib as mpl
import celltypist as CT
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional
from os.path import dirname
from collections import deque
from scipy import sparse as sp
from scipy.stats import entropy
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from numpy.typing import ArrayLike
from time import perf_counter as clock
from collections import defaultdict as ddict
from scipy.stats import median_abs_deviation
from scipy.io import mmread as matrix_market_read
from scipy.io import mmwrite as matrix_market_write

from toomanycells import cellAnnotation

#Matplotlib parameters.
mpl.use("agg")
mpl.rcParams["figure.dpi"]=600
# mpl.rcParams["pdf.fonttype"]=42
mpl.rc("pdf", fonttype=42)

font = {'weight' : 'normal', 'size'   : 18}
mpl.rc("font", **font)

sys.path.insert(0, dirname(__file__))
from common import MultiIndexList
from tmcHaskell import TMCHaskell
from tmcGraph import TMCGraph
from similarityMatrix import SimilarityMatrix
from multiPlotter import MultiPlotter
from cellAnnotation import CellAnnotation

#=====================================================
class TooManyCells:
    """
    This class focuses on one aspect of the original
        Too-Many-Cells tool, the clustering.
        Features such as normalization,
        dimensionality reduction and many others can be
        applied using functions from libraries like
        Scanpy, or they can be implemented locally. This
        implementation also allows the possibility of
        new features with respect to the original
        Too-Many-Cells. For example, imagine you want to
        continue partitioning fibroblasts until you have
        at most a given number of cells, even if the
        modularity becomes negative, but for CD8+ T-cells
        you do not want to have partitions with less
        than 100 cells. This can be easily implemented
        with a few conditions using the cell annotations
        in the .obs data frame of the AnnData object.

    With regards to visualization, we recommend
        using the too-many-cells-interactive tool.
        You can find more information about it at:
        https://doi.org/10.1093/gigascience/giae056
        Once installed, you can use the function
        visualize_with_tmc_interactive() to
        generate the visualization. You will need
        path to the installation folder of
        too-many-cells-interactive.


    """
    #=================================================
    def __init__(self,
            input: Optional[Union[sc.AnnData, str]] = None,
            output: str = "",
            input_is_matrix_market: bool = False,
            use_full_matrix: bool = False,
            use_raw: bool = False,
            ):
        """
        The constructor takes the following inputs.

        :param input: Path to input directory or \
                AnnData object.
        :param output: Path to output directory.
        :param input_is_matrix_market: If true, \
                the directory should contain a \
                .mtx file, a barcodes.tsv file \
                and a genes.tsv file.

        :return: a TooManyCells object.
        :rtype: :obj:`TooManyCells`

        """

        #We use a directed graph to enforce the parent
        #to child relation.
        self.G = nx.DiGraph()
        self.set_of_leaf_nodes = set()

        if input is None:
            matrix = np.array([[]])
            self.A = sc.AnnData(matrix)

        elif isinstance(input, TooManyCells):

            #Clone the given TooManyCells object.
            self.A = input.A.copy()
            self.G = input.G.copy()
            self.output = input.output
            self.tmcGraph = TMCGraph(
                graph = self.G,
                adata = self.A,
                output= self.output,
            )

            S = input.set_of_leaf_nodes.copy()
            self.set_of_leaf_nodes = S

        elif isinstance(input, str):
            self.source = os.path.abspath(input)
            if self.source.endswith(".h5ad"):
                self.t0 = clock()
                self.A = sc.read_h5ad(self.source)
                self.tf = clock()
                delta = self.tf - self.t0
                txt = ("Elapsed time for loading: " +
                        f"{delta:.2f} seconds.")
                print(txt)
            else:
                if input_is_matrix_market:
                    try:
                        self.A = sc.read_10x_mtx(self.source)
                    except:
                        self.convert_mm_from_source_to_adata()
                else:
                    for f in os.listdir(self.source):
                        if f.endswith('.h5ad'):
                            fname = os.path.join(
                                self.source, f)
                            self.t0 = clock()
                            self.A = sc.read_h5ad(fname)
                            self.tf = clock()
                            delta = self.tf - self.t0
                            txt = ('Elapsed time for ' +
                                   'loading: ' +
                                    f'{delta:.2f} seconds.')
                            print(txt)
                            break

        elif isinstance(input, sc.AnnData):
            self.A = input
        else:
            raise ValueError('Unexpected input type.')

        #If no output directory is provided,
        #we use the current working directory.
        if output == "":
            output = os.getcwd()
            output = os.path.join(output, "tmc_outputs")
            print(f"Outputs will be saved in: {output}")

        os.makedirs(output, exist_ok=True)

        self.output = os.path.abspath(output)

        #This column of the obs data frame indicates
        #the correspondence between a cell and the 
        #leaf node of the spectral clustering tree.
        sp_cluster = "sp_cluster"
        sp_path = "sp_path"
        if sp_cluster not in self.A.obs.columns:
            self.A.obs[sp_cluster] = -1
        if sp_path not in self.A.obs.columns:
            self.A.obs[sp_path]    = ""

        t = self.A.obs.columns.get_loc(sp_cluster)
        self.cluster_column_index = t
        t = self.A.obs.columns.get_loc(sp_path)
        self.path_column_index = t

        self.delta_clustering = 0
        self.final_n_iter     = 0

        #self.FDT = np.float64
        self.FDT = np.float32
        txt = f"Data will be treated as: {self.FDT}."
        print(txt)

        #Create a copy to avoid direct modifications
        #of the original count matrix X.
        #Note that we are making sure that the 
        #sparse matrix has the CSR format. This
        #is relevant when we normalize.

        self.is_sparse = False
        self.X : Union[sp.spmatrix, sp.sparray, np.ndarray]

        if use_raw:
            if "raw" not in self.A.layers:
                txt = "Layer raw is not available."
                raise ValueError(txt)

            self.X = self.A.layers["raw"].copy()

        else:

            self.X = self.A.X.copy()

        if sp.issparse(self.X):
            #Compute the density of the matrix
            rho = self.X.nnz / np.prod(self.X.shape)
            #If more than 50% of the matrix is occupied,
            #we generate a dense version of the matrix.
            sparse_threshold = 0.50
            if use_full_matrix or sparse_threshold < rho:
                self.is_sparse = False
                self.X = self.X.toarray()
                txt = ("Using a dense representation" 
                       " of the count matrix.")
                print(txt)
                self.X = self.X.astype(self.FDT)
            else:
                self.is_sparse = True
                #Make sure we use a CSR array format.
                if not isinstance(self.X, sp.csr_array):
                    self.X = sp.csr_array(self.X,
                                          dtype=self.FDT,
                                          copy=True)
        else:
            #The matrix is dense.
            print("The matrix is dense.")
            self.is_sparse = False
            # self.X = self.X.copy()
            self.X = self.X.astype(self.FDT)

        self.n_cells, self.n_genes = self.A.shape

        self.too_few_observations = False
        if self.n_cells < 2:
            print("Warning: Too few observations (cells).")
            self.too_few_observations = True

        print(self.A)

        #Location of the matrix data for TMCI
        self.tmci_mtx_dir = ""

        self.spectral_clustering_has_been_called = False

        self.cells_to_be_eliminated = None

        # We use a deque to offer the possibility of breadth-
        # versus depth-first. Our current implementation
        # uses depth-first to be consistent with the 
        # numbering scheme of too-many-cells interactive.
        self.DQ = deque()

        #Map a node to the path in the
        #binary tree that connects the
        #root node to the given node.
        # self.tmcGraph.node_to_path = {}

        #Map a node to a list of indices
        #that provide access to the JSON
        #structure.
        # self.tmcGraph.node_to_j_index = {}

        # self.tmcGraph.node_counter = 0

        #the JSON structure representation
        #of the tree.
        # self.tmcGraph.J = MultiIndexList()

        self.tmcGraph = TMCGraph(graph=self.G,
                                 adata=self.A,
                                 output= self.output)


        #The threshold for modularity to 
        #accept a given partition of a set
        #of cells.
        self.eps = 1e-9

        # self.use_twopi_cmd   = True
        self.verbose_mode    = False


    #=====================================
    def estimate_n_of_iterations(self) -> int:
        """
        We assume a model of the form \
        number_of_iter = const * N^exponent \
        where N is the number of cells.

        Note that this estimate was derived
        from sparse scRNA-seq matrices.
        """

        #Average number of cells per leaf node
        k = np.power(10, -0.6681664297844971)
        exponent = 0.86121348
        #exponent = 0.9
        q1 = k * np.power(self.n_cells, exponent)
        q2 = 2
        iter_estimates = np.array([q1,q2], dtype=int)
        
        return iter_estimates.max()

    #=====================================
    def print_message_before_clustering(self):

        print("The first iterations are typically slow.")
        print("However, they tend to become faster as ")
        print("the size of the partition becomes smaller.")
        print("Note that the number of iterations is")
        print("only an estimate.")
    #=====================================
    def reverse_path(self, p: str)->str:
        """
        This function reverses the path from the root\
        node to the leaf node.
        """
        reversed_p = "/".join(p.split("/")[::-1])
        return reversed_p

    #=====================================
    def run_spectral_clustering(
            self,
            shift_similarity_matrix: float = 0,
            shift_until_nonnegative: bool = False,
            store_similarity_matrix: bool = False,
            normalize_rows: bool = False,
            similarity_function: str = "cosine_sparse",
            similarity_norm: float = 2,
            similarity_power: float = 1,
            similarity_gamma: Optional[float] = None,
            use_tf_idf: bool = False,
            tf_idf_norm: Optional[str] = None,
            tf_idf_smooth: bool = True,
            use_hermitian_method: bool = False,
            svd_algorithm: str = "arpack",
            plot_similarity_matrix: bool = False,
            modularity_threshold: float = 1e-9,
            use_exact_diameter: bool = False,
            use_adaptive_diameter: bool = True,
    ):
        """
        This function computes the partitions of the \
                initial cell population and continues \
                until the modularity of the newly \
                created partitions is below threshold.
        """

        #In case the user wants to call this function again.
        self.spectral_clustering_has_been_called = True

        if self.too_few_observations:
            raise ValueError("Too few observations (cells).")

        simMat = SimilarityMatrix(
            self.X,
            use_hermitian_method,
            svd_algorithm,
            self.output,
            self.verbose_mode,
        )

        simMat.compute_similarity_matrix(
            shift_similarity_matrix,
            shift_until_nonnegative,
            store_similarity_matrix,
            normalize_rows,
            similarity_function,
            similarity_norm,
            similarity_power,
            similarity_gamma,
            use_tf_idf,
            tf_idf_norm,
            tf_idf_smooth,
            plot_similarity_matrix,
            use_exact_diameter,
            use_adaptive_diameter,
        )

        #===========================================
        #=============Main=Loop=====================
        #===========================================
        self.t0 = clock()

        node_id = self.tmcGraph.node_counter

        #Initialize the array of cells to partition
        rows = np.array(range(self.X.shape[0]))

        #Initialize the deque
        # self.DQ.append((rows, None))
        # self.DQ.append(rows)

        #Initialize the graph
        self.G.add_node(node_id, size=len(rows))

        #Path to reach root node.
        self.tmcGraph.node_to_path[node_id] = str(node_id)

        #Indices to reach root node.
        self.tmcGraph.node_to_j_index[node_id] = (1,)

        #Update the node counter
        self.tmcGraph.node_counter += 1

        #============STEP=1================Cluster(0)

        p_node_id = node_id

        if similarity_function == "cosine_sparse":
            Q,S = simMat.compute_partition_for_cosp(rows)
        elif similarity_function == "norm_sparse":
            Q,S = simMat.compute_partition_for_normsp(rows)
        else:
            Q,S = simMat.compute_partition_for_full(rows)

        if modularity_threshold < Q:
            #Modularity is above threshold, and
            #thus each partition will be 
            #inserted into the deque.

            D = self.tmcGraph.modularity_to_json(Q)

            #Update json index
            self.tmcGraph.J.append(D)
            self.tmcGraph.J.append([])
            # self.tmcGraph.J.append([[],[]])
            # j_index = (1,)

            self.G.nodes[node_id]["Q"] = Q

            for indices in S:
                T = (indices, p_node_id)
                self.DQ.append(T)

        else:
            #Modularity is below threshold and 
            #therefore this partition will not
            #be considered.
            print("Step 1:")
            print("Modularity is below threshold.")
            txt = f"{Q=}"
            print(txt)
            txt = ("All cells belong" 
                    " to the same partition.")
            print(txt)
            #Update the relation between a set of
            #cells and the corresponding leaf node.
            #Also include the path to reach that 
            #node.
            c = self.cluster_column_index
            self.A.obs.iloc[rows, c] = node_id

            p = self.path_column_index
            self.A.obs.iloc[rows, p] = str(node_id)

            self.set_of_leaf_nodes.add(node_id)

            #Update the JSON structure for 
            #a leaf node.
            L = self.tmcGraph.cells_to_json(rows)
            self.tmcGraph.J.append(L)
            self.tmcGraph.J.append([])
            return -1

        max_n_iter = self.estimate_n_of_iterations()

        self.print_message_before_clustering()

        #============STEP=2================
        with tqdm(total=max_n_iter) as pbar:
            while 0 < len(self.DQ):

                #Get the rows corresponding to the
                #partition and the (parent) node
                #that produced such partition.
                rows, p_node_id = self.DQ.pop()

                #This id is for the new node.
                node_id += 1

                # For every cluster of cells that is popped
                # from the deque, we update the node_id. 
                # If the cluster is further partitioned we 
                # will store each partition but will not 
                # assign node numbers. Node numbers will 
                # only be assigned after being popped from 
                # the deque.

                # We need to know the modularity to 
                # determine if the node will be partitioned.
                if similarity_function == "cosine_sparse":
                    Q,S=simMat.compute_partition_for_cosp(
                            rows)
                elif similarity_function == "norm_sparse":
                    Q,S=simMat.compute_partition_for_normsp(
                            rows)
                    #Q = 0
                else:
                    Q,S=simMat.compute_partition_for_full(
                            rows)
                    #Q = 0

                # If the parent node is 0, then the path is
                # "0".
                current_path = self.tmcGraph.node_to_path[
                    p_node_id]

                #Update path for the new node
                new_path = current_path 
                new_path += "/" + str(node_id) 
                self.tmcGraph.node_to_path[node_id]=new_path

                # If the parent node is 0, then j_index is
                # (1,)
                j_index = self.tmcGraph.node_to_j_index[
                    p_node_id]

                n_stored_blocks = len(
                    self.tmcGraph.J[j_index])

                self.tmcGraph.J[j_index].append([])
                #Update the j_index. For example, if
                #j_index = (1,) and no blocks have been
                #stored, then the new j_index is (1,0).
                #Otherwise, it is (1,1).
                j_index += (n_stored_blocks,)

                #Include new node into the graph.
                self.G.add_node(node_id, size=len(rows))

                #Include new edge into the graph.
                self.G.add_edge(p_node_id, node_id)

                if modularity_threshold < Q:
                    #Modularity is above threshold, and
                    #thus each partition will be 
                    #inserted into the deque.

                    D = self.tmcGraph.modularity_to_json(Q)
                    self.tmcGraph.J[j_index].append(D)
                    self.tmcGraph.J[j_index].append([])
                    j_index += (1,)

                    # We only store the modularity of nodes
                    # whose modularity is above threshold.
                    self.G.nodes[node_id]["Q"] = Q

                    # Update the j_index for the newly 
                    # created node. (1,0,1)
                    self.tmcGraph.node_to_j_index[
                        node_id] = j_index

                    # Append each partition to the deque.
                    for indices in S:
                        T = (indices, node_id)
                        self.DQ.append(T)

                else:
                    #Modularity is below threshold and 
                    #therefore this partition will not
                    #be considered.

                    #Update the relation between a set of
                    #cells and the corresponding leaf node.
                    #Also include the path to reach that 
                    #node.
                    c = self.cluster_column_index
                    self.A.obs.iloc[rows, c] = node_id

                    reversed_path = self.reverse_path(
                        new_path)
                    p = self.path_column_index
                    self.A.obs.iloc[rows, p] = reversed_path

                    self.set_of_leaf_nodes.add(node_id)

                    #Update the JSON structure for 
                    #a leaf node.
                    L = self.tmcGraph.cells_to_json(rows)
                    self.tmcGraph.J[j_index].append(L)
                    self.tmcGraph.J[j_index].append([])

                pbar.update()

            #==============END OF WHILE==============
            pbar.total = pbar.n
            self.final_n_iter = pbar.n
            pbar.refresh()

        self.tmcGraph.set_of_leaf_nodes = self.set_of_leaf_nodes

        self.tf = clock()
        self.delta_clustering = self.tf - self.t0
        self.delta_clustering /= 60
        txt = ("Elapsed time for clustering: " +
                f"{self.delta_clustering:.2f} minutes.")
        print(txt)


    #=====================================
    def store_outputs(
            self,
            cell_ann_col: Optional[str] = None,
            store_in_uns_dict: bool = False,
            plot_tree: bool = False,
            draw_modularity: bool = False,
            draw_node_numbers: bool = False,
            ):
        """
        Store the outputs and plot the branching tree.

        File outputs:

        cluster_list.json: The json file containing a list 
        of clusters. 

        cluster_tree.json: The json file containing the 
        output tree in a recursive format.

        graph.json: A JSON file of the tree. It includes the 
        modularity and the size.

        node_info.csv: Size and modularity of each node.

        clusters.csv: The cluster membership for each cell.

        """

        self.t0 = clock()

        # nx.drawing.nx_pydot.write_dot(self.G, dot_fname)
        # nx.nx_agraph.write_dot(self.G, dot_fname)

        #Write graph data to file.
        self.tmcGraph.set_of_leaf_nodes
        self.tmcGraph.store_outputs(store_in_uns_dict)

        #Store the cell annotations in the output folder.
        if cell_ann_col:

            if cell_ann_col in self.A.obs.columns:
                pass
            else:
                txt = "Annotation column does not exists."
                raise ValueError(txt)

            self.generate_cell_annotation_file(cell_ann_col)

            if plot_tree:

                labels_path = os.path.join(
                    self.output,
                    "cell_annotation_labels.csv")

                haskell = TMCHaskell(
                    self.output,
                    tmc_tree_path = self.output,
                    path_to_cell_annotations=labels_path,
                    draw_modularity = draw_modularity,
                    draw_node_numbers = draw_node_numbers,
                )
        
                print("Generating Haskell plot ...")
                haskell.run()


        print(self.G)

        #Number of cells for each node
        size_list = []
        #Modularity for each node
        Q_list = []
        #Node label
        node_list = []

        for node, attr in self.G.nodes(data=True):
            node_list.append(node)
            size_list.append(attr["size"])
            if "Q" in attr:
                Q_list.append(attr["Q"])
            else:
                Q_list.append(np.nan)

        #Write node information to CSV
        D = {"node": node_list, "size":size_list, "Q":Q_list}
        df = pd.DataFrame(D)
        fname = "node_info.csv"
        fname = os.path.join(self.output, fname)
        df.to_csv(fname, index=False)

        # Eliminate dependencies with GraphViz
        # if self.use_twopi_cmd and store_tree_svg:
        #     self.plot_radial_tree_from_dot_file()

        self.tf = clock()
        delta = self.tf - self.t0
        txt = ("Elapsed time for storing outputs: " +
                f"{delta:.2f} seconds.")
        print(txt)


    #=====================================
    def convert_mm_from_source_to_adata(self):
        """
        This function reads the matrix.mtx file \
                located at the source directory.\
                Since we assume that the matrix \
                has the format genes x cells, we\
                transpose the matrix, then \
                convert it to the CSR format \
                and then into an AnnData object.
        """

        self.t0 = clock()

        print("Loading data from .mtx file.")
        print("Note that we assume the raw data "
              "has the following format:")
        print("genes=rows and cells=columns.")
        print("The AnnData object will have the format:")
        print("cells=rows and genes=columns.")

        fname = None
        for f in os.listdir(self.source):
            opt_1 = f.endswith(".mtx.gz")
            opt_2 = f.endswith(".mtx")
            if opt_1 or opt_2:
                fname = f
                break

        if fname is None:
            txt = ".mtx or .mtx.gz file not found."
            raise ValueError(txt)

        fname = os.path.join(self.source, fname)
        mat = matrix_market_read(fname)
        #Remember that the input matrix has
        #genes for rows and cells for columns.
        #Thus, just transpose.
        self.A = mat.T.tocsr()
        self.A = sc.AnnData(self.A)

        # ==================== BARCODES ================
        fname = None
        for f in os.listdir(self.source):
            opt_1 = f == "barcodes.tsv"
            opt_2 = f == "barcodes.tsv.gz"
            if opt_1 or opt_2:
                fname = f
                break

        if fname is not None:
            fname = os.path.join(self.source, fname)
            print(f"Loading {fname}")
            df_barcodes = pd.read_csv(
                    fname, delimiter="\t", header=None)
            barcodes = df_barcodes.loc[:,0].tolist()

            self.A.obs_names = barcodes

        # ==================== GENES ================
        fname = None
        for f in os.listdir(self.source):
            opt_1 = f == "genes.tsv"
            opt_2 = f == "genes.tsv.gz"
            opt_3 = f == "features.tsv"
            opt_4 = f == "features.tsv.gz"
            if opt_1 or opt_2 or opt_3 or opt_4:
                fname = f
                break

        if fname is not None:
            fname = os.path.join(self.source, fname)
            print(f"Loading {fname}")
            df_genes = pd.read_csv(
                    fname, delimiter="\t", header=None)
            genes = df_genes.loc[:,0].tolist()

            self.A.var_names = genes

        self.tf = clock()
        delta = self.tf - self.t0
        txt = ("Elapsed time for loading: " + 
                f"{delta:.2f} seconds.")
        print(txt)


    #=====================================
    def generate_cell_annotation_file(self,
            cell_ann_col: str = "cell_annotations",
            tag: str="cell_annotation_labels"
    ):
        """
        This function stores a CSV file with\
            the labels for each cell.

        :param column: Name of the\
            column in the .obs data frame of\
            the AnnData object that contains\
            the labels to be used for the tree\
            visualization. For example, cell \
            types.

        """
        if tag[-3:] == ".csv":
            pass
        else:
            fname = tag + ".csv"

        fname = os.path.join(self.output, fname)
        self.cell_annotations_path = fname

        ca = self.A.obs[cell_ann_col].copy()
        ca.index.names = ["item"]
        ca = ca.rename("label")
        ca.to_csv(fname, index=True)

    #=====================================
    def create_data_for_tmci(
            self,
            tmci_mtx_dir: str = "tmci_mtx_data",
            list_of_genes: List = [],
            path_to_genes: str = "",
            create_matrix: bool = True,
        ):
        """
        Produce the 10X files for a given set of\
            genes.  This function produces the\
            genes x cells matrix market format matrix,\
            the genes.tsv file and the barcodes.
        If a path is provided for the genes, then the\
            first column of the csv file must have the\
            gene names.
        """

        self.tmci_mtx_dir = os.path.join(
            self.output, tmci_mtx_dir)

        os.makedirs(self.tmci_mtx_dir, exist_ok=True)

        # Genes
        genes_f = "genes.tsv"
        genes_f = os.path.join(self.tmci_mtx_dir, genes_f)

        var_names   = []
        col_indices = []
        obs_cols    = []

        if 0 < len(path_to_genes):

            df = pd.read_csv(path_to_genes, header=0)
            #The first column should contain the genes.
            list_of_genes = df.iloc[:,0].to_list()

        if 0 < len(list_of_genes):

            # Note that if we provided a path then
            # the list_of_genes variable has a
            # positive length.
            gene_names_lower = self.A.var_names.str.lower()

            for gene in list_of_genes:

                gene_lower = gene.lower()

                if gene_lower in gene_names_lower:
                    col_index = gene_names_lower.get_loc(
                        gene_lower)
                    var_names.append(gene)
                    col_indices.append(col_index)

                elif gene in self.A.obs:

                    obs_cols.append(gene)

                else:
                    #Gene is not available
                    continue
    
            if 0 < len(var_names):

                #Use genes from the expression matrix.
                m1 = self.A.X[:,col_indices]

                if 0 < len(obs_cols):

                    var_names.extend(obs_cols)

                    #Use "genes" from the obs data frame.
                    m2 = self.A.obs[obs_cols].values

                    if sp.issparse(m1):

                        #Make m2 sparse too
                        #Concatenate two sparse objects.
                        m2 = sp.csr_matrix(m2)
                        G_mtx = sp.hstack((m1,m2))

                    else:

                        #Concatenate two numpy objects
                        G_mtx = np.hstack((m1,m2))

                else:
                    #No genes from the obs data frame.
                    #Just use the expression matrix
                    G_mtx = m1

            else:

                #Use "genes" only from the obs data frame.
                if 0 < len(obs_cols):
                    #Use "genes" from the obs data frame.
                    var_names = obs_cols.copy()
                    G_mtx = self.A.obs[obs_cols].values
                else:
                    print("No genes were found.")

        else:

            #If no list is provided, use all the genes.
            print("Warning: All genes will be used.")
            print("Warning: This is a lengthy process.")
            var_names = self.A.var_names
            G_mtx = self.A.X

        L = [var_names, var_names]
        pd.DataFrame(L).transpose().to_csv(
            genes_f,
            sep="\t",
            header=False,
            index=False)

        # Barcodes
        barcodes_f = "barcodes.tsv"
        barcodes_f = os.path.join(self.tmci_mtx_dir,
                                  barcodes_f)

        pd.Series(self.A.obs_names).to_csv(
            barcodes_f,
            sep="\t",
            header=False,
            index=False,
        )

        # Matrix
        if create_matrix:
            matrix_f = "matrix.mtx"
            matrix_f = os.path.join(self.tmci_mtx_dir,
                                    matrix_f)
            matrix_market_write(matrix_f,
                                sp.coo_matrix(G_mtx.T))

    #=====================================
    def visualize_with_tmc_interactive(self,
            path_to_tmc_interactive: str,
            use_column_for_labels: str = "",
            port: int = 9991,
            include_matrix_data: bool = False,
            tmci_mtx_dir: str = "",
            ) -> None:
        """
        This function produces a visualization\
                using too-many-cells-interactive.

        :param path_to_tmc_interactive: Path to \
                the too-many-cells-interactive \
                directory.
        :param use_column_for_labels: Name of the\
                column in the .obs data frame of\
                the AnnData object that contains\
                the labels to be used in the tree\
                visualization. For example, cell \
                types.
        :param port: Port to be used to open\
                the app in your browser using\
                the address localhost:port.

        """

        fname = "cluster_tree.json"
        fname = os.path.join(self.output, fname)
        tree_path = fname
        port_str = str(port)


        bash_exec = "./start-and-load.sh"


        if len(use_column_for_labels) == 0:
            label_path_str = ""
            label_path     = ""
        else:
            self.generate_cell_annotation_file(
                    use_column_for_labels)
            label_path_str = "--label-path"
            label_path     = self.cell_annotations_path
        
        if include_matrix_data:
            matrix_path_str = "--matrix-dir"
            if 0 < len(tmci_mtx_dir):
                matrix_dir = tmci_mtx_dir
            else:

                if len(self.tmci_mtx_dir) == 0:
                    print("No path for TMCI mtx.")
                    print("Creating TMCI mtx data.")
                    self.create_data_for_tmci()

                matrix_dir = self.tmci_mtx_dir
        else:
            matrix_path_str = ""
            matrix_dir = ""

        command = [
                bash_exec,
                matrix_path_str,
                matrix_dir,
                '--tree-path',
                tree_path,
                label_path_str,
                label_path,
                '--port',
                port_str
                ]

        command = list(filter(len,command))
        command = ' '.join(command)
        
        #Run the command as if we were inside the
        #too-many-cells-interactive folder.
        final_command = (f"(cd {path_to_tmc_interactive} "
                f"&& {command})")
        #print(final_command)
        url = 'localhost:' + port_str
        txt = ("Once the app is running, just type in "
                f"your browser \n        {url}")
        print(txt)
        txt="The app will start loading after pressing Enter."
        print(txt)
        pause = input('Press Enter to continue ...')
        p = subprocess.call(final_command, shell=True)

    #=====================================
    def update_cell_annotations(
            self,
            df: pd.DataFrame,
            column: str = "cell_annotations"):
        """
        Insert a column of cell annotations in the \
        AnnData.obs data frame. The column in the \
        data frame should be called "label". The \
        name of the column in the AnnData.obs \
        data frame is provided by the user through \
        the column argument.
        """

        if "label" not in df.columns:
            raise ValueError("Missing label column.")

        #Reindex the data frame.
        df = df.loc[self.A.obs.index]

        if df.shape[0] != self.A.obs.shape[0]:
            raise ValueError("Data frame size mismatch.")

        self.A.obs[column] =  df["label"]

    #=====================================
    def generate_matrix_from_signature_file(
            self,
            signature_path: str):
        """
        Generate a matrix from the signature provided \
            through a file. The entries with a positive
            weight are assumed to be upregulated and \
            those with a negative weight are assumed \
            to be downregulated. The algorithm will \
            standardize the matrix, i.e., centering \
            and scaling.

        If the signature has both positive and \
            negative weights, two versions will be \
            created. The unadjusted version simply \
            computes a weighted average using the \
            weights provided in the signature file.\
            In the adjusted version the weights \
            are adjusted to give equal weight to the \
            upregulated and downregulated genes.

        Assumptions

        We assume that the file has at least two \
            columns. One should be named "Gene" and \
            the other "Weight". \
            The count matrix has cells for rows and \
            genes for columns.
        """

        df_signature = pd.read_csv(signature_path, header=0)

        Z = sc.pp.scale(self.A, copy=True)
        Z_is_sparse = sp.issparse(Z)

        vec = np.zeros(Z.X.shape[0])

        up_reg = vec * 0
        down_reg = vec * 0

        up_count = 0
        up_weight = 0

        down_count = 0
        down_weight = 0

        G = df_signature["Gene"]
        W = df_signature["Weight"]

        for gene, weight in zip(G, W):
            if gene not in Z.var.index:
                continue
            col_index = Z.var.index.get_loc(gene)

            if Z_is_sparse:
                gene_col = Z.X.getcol(col_index)
                gene_col = np.squeeze(gene_col.toarray())
            else:
                gene_col = Z.X[:,col_index]

            if 0 < weight:
                up_reg += weight * gene_col
                up_weight += weight
                up_count += 1
            else:
                down_reg += weight * gene_col
                down_weight += np.abs(weight)
                down_count += 1
        
        total_counts = up_count + down_count
        total_weight = up_weight + down_weight

        list_of_names = []
        list_of_gvecs = []

        UnAdjSign = up_reg + down_reg
        UnAdjSign /= total_weight
        self.A.obs["UnAdjSign"] = UnAdjSign
        list_of_gvecs.append(UnAdjSign)
        list_of_names.append("UnAdjSign")

        up_factor = down_count / total_counts
        down_factor = up_count / total_counts

        modified_total_counts = 2 * up_count * down_count
        modified_total_counts /= total_counts
        
        check = up_factor*up_count + down_factor*down_count

        print(f"{up_count=}")
        print(f"{down_count=}")
        print(f"{total_counts=}")
        print(f"{modified_total_counts=}")
        print(f"{check=}")
        print(f"{up_factor=}")
        print(f"{down_factor=}")


        mixed_signs = True
        if 0 < up_count:
            UpReg   = up_reg / up_count
            self.A.obs["UpReg"] = UpReg
            list_of_gvecs.append(UpReg)
            list_of_names.append("UpReg")
            print("UpRegulated genes: stats")
            print(self.A.obs["UpReg"].describe())
    
        else:
            mixed_signs = False

        if 0 < down_count:
            DownReg   = down_reg / down_count
            self.A.obs["DownReg"] = DownReg
            list_of_gvecs.append(DownReg)
            list_of_names.append("DownReg")
            print("DownRegulated genes: stats")
            print(self.A.obs["DownReg"].describe())
            txt = ("Note: In our representation, " 
                   "the higher the value of a downregulated "
                   "gene, the more downregulated it is.")
            print(txt)
        else:
            mixed_signs = False

        if mixed_signs:
            AdjSign  = up_factor * up_reg
            AdjSign += down_factor * down_reg
            AdjSign /= modified_total_counts
            self.A.obs["AdjSign"] = AdjSign
            list_of_gvecs.append(AdjSign)
            list_of_names.append("AdjSign")

        m = np.vstack(list_of_gvecs)

        #This function will produce the 
        #barcodes.tsv and the genes.tsv file.
        self.create_data_for_tmci(
            list_of_genes = list_of_names,
            create_matrix=False)


        m = m.astype(self.FDT)

        mtx_path = os.path.join(
            self.tmci_mtx_dir, "matrix.mtx")

        matrix_market_write(mtx_path,
                            sp.coo_matrix(m))


    #=====================================
    def get_path_from_root_to_node(
            self,
            target: int,
            ):
        """
        For a given node, we find the path from the root 
        to that node.
        """

        node = target
        path_vec = [node]
        modularity_vec = [0]

        while node != 0:
            # Get an iterator for the predecessors.
            # There should only be one predecessor.
            predecessors = self.G.predecessors(node)
            node = next(predecessors)
            Q = self.G._node[node]["Q"]
            Q = float(Q)
            path_vec.append(node)
            modularity_vec.append(Q)
        
        # We assume that the distance between two children
        # nodes is equal to the modularity of the parent node.
        # Hence, the distance from a child to a parent is 
        # half the modularity.
        modularity_vec = 0.5 * np.array(
            modularity_vec, dtype=self.FDT)
        path_vec = np.array(path_vec, dtype=int)

        return (path_vec, modularity_vec)

    #=====================================
    def get_path_from_node_x_to_node_y(
            self,
            x: int,
            y: int,
            ):
        """
        For a given pair of nodes x and y, we find the
        path between those nodes.
        
        We use the path from the root node to the 
        corresponding nodes, and then we remove the
        intersection except at the branching point.
        """
        x_path, x_dist = self.get_path_from_root_to_node(x)
        y_path, y_dist = self.get_path_from_root_to_node(y)

        x_set = set(x_path)
        y_set = set(y_path)

        # print(x_dist)
        # print(y_dist)

        # print("===========")

        # print(x_path)
        # print(y_path)

        # print("===========")

        intersection = x_set.intersection(y_set)
        intersection = list(intersection)
        intersection = np.array(intersection)
        n_intersection = len(intersection)
        
        pivot_node = x_path[-n_intersection]
        pivot_dist = x_dist[-n_intersection]

        x_path = x_path[:-n_intersection]
        y_path = y_path[:-n_intersection]
        y_path = y_path[::-1]

        x_dist = x_dist[:-n_intersection]
        y_dist = y_dist[1:-n_intersection]
        y_dist = y_dist[::-1]

        full_path = np.hstack((x_path,pivot_node,y_path))
        full_dist = np.hstack(
            (x_dist, pivot_dist, pivot_dist, y_dist))
        full_dist = full_dist.cumsum()

        # print(full_path) 
        # print(full_dist)

        return (full_path, full_dist)

    #=====================================
    def compute_cluster_mean_expression(
            self, 
            node: int, 
            genes: Union[List[str], str],
            output_list: bool = False,
        ) -> Union[List[float], float]:
        """
        Compute the mean expression for a specific
        cluster.

        If you plan to use this function repeatedly,
        then consider calculating the mean expression
        for each node of the whole tree.

        The corresponding function is:
        populate_tree_with_mean_expression_for_all_markers()
        """

        #Get all the descendants for a given node.
        #This is a set.
        nodes = nx.descendants(self.G, node)

        if len(nodes) == 0:
            #This is a leaf node.
            nodes = [node]
        else:
            #Make sure these are leaf nodes.
            nodes = self.set_of_leaf_nodes.intersection(nodes)
            # nodes = list(nodes)

        is_string = False

        #Select the cells that belong to that node (branch).
        mask = self.A.obs["sp_cluster"].isin(nodes)

        if isinstance(genes, str):
            is_string = True
            list_of_genes = [genes]

        else:
            list_of_genes = genes

        exp_vec = []
        mean_exp = 0

        gene_names_lower = self.A.var_names.str.lower()

        for gene in list_of_genes:

            gene_lower = gene.lower()

            if gene_lower not in gene_names_lower:
                raise ValueError(f"{gene=} was not found.")

            col_index = gene_names_lower.get_loc(
                gene_lower)

            mean_exp = self.A.X[mask, col_index].mean()
            exp_vec.append(mean_exp)

        # print(f"{total_exp=}")
        # print(f"{n_cells=}")
        # print(f"{mean_exp=}")

        if is_string and not output_list:
            return mean_exp
        else:
            return exp_vec

    #=====================================
    def load_cluster_info(
            self,
            cluster_file_path: str ="",
            ):
        """
        Load the cluster file.
        """

        self.t0 = clock()

        if 0 < len(cluster_file_path):
            cluster_fname = cluster_file_path

        else:
            fname = "clusters.csv"
            cluster_fname = os.path.join(self.output, fname)

        if not os.path.exists(cluster_fname):
            raise ValueError("File does not exists.")

        df = pd.read_csv(cluster_fname, index_col=0)
        self.A.obs["sp_cluster"] = df["cluster"]

        # This set should  be equal to the one
        # stored in the tmcGraph object.
        self.set_of_leaf_nodes = set(df["cluster"])

        self.tf = clock()
        delta = self.tf - self.t0
        txt = ("Elapsed time to load cluster file: " + 
                f"{delta:.2f} seconds.")
        print(txt)


    #=====================================
    def plot_expression_from_node_x_to_node_y(
            self,
            x: int,
            y: int,
            genes: Union[List[str], str],
            ):
        """
        For a given pair of nodes x and y, we compute the
            gene expression path along the path connecting
            those nodes. 
        Make sure that property set_of_leaf_nodes is
            populated with the correct information.

        TODO: Make sure you populate the gene expression
        on each node using the right function.
        
        If you plan to use this function repeatedly,
        then consider calculating the mean expression
        for the whole tree first.

        The corresponding function is:
        populate_tree_with_mean_expression_for_all_markers()
        """

        if isinstance(genes, str):
            list_of_genes = [genes]
        else:
            list_of_genes = genes

        T = self.get_path_from_node_x_to_node_y(x,y)
        path_vec, dist_vec = T
        n_nodes = len(path_vec)
        n_genes = len(list_of_genes)
        exp_mat = np.zeros((n_genes,n_nodes))

        for col,node in enumerate(path_vec):
            g_exp = self.compute_cluster_mean_expression(
                node, list_of_genes)
            exp_mat[:,col] = g_exp

        fig,ax = plt.subplots()

        # bogus_names = ["Gene A", "Gene B"]
        # colors = ["blue", "red"]

        for row, gene in enumerate(list_of_genes):
            ax.plot(dist_vec,
                    exp_mat[row,:],
                    linewidth=3,
                    label=gene,
                    # label=bogus_names[row],
                    # color = colors[row]
                    )

        plt.legend()
        txt = f"From node {x} to node {y}"
        # txt = f"From node X to node Y"
        ax.set_title(txt)
        ax.set_ylabel("Gene expression")
        ax.set_xlabel("Distance (modularity units)")
        plt.ticklabel_format(style='sci',
                             axis='x',
                             scilimits=(0,0))

        fname = "expression_path.pdf"
        fname = os.path.join(self.output, fname)
        fig.savefig(fname, bbox_inches="tight")
        print("Plot has been generated.")

    #=====================================
    def compute_marker_mean_value_for_cell(
            self,
            marker: str,
            cell: str,
            cell_ann_col: str = "cell_annotations",
    ):

        CA = cell_ann_col
        if marker not in self.A.var_names:
            return None

        col_index = self.A.var.index.get_loc(marker)
        mask = self.A.obs[CA] == cell
        mean_exp = self.A.X[mask, col_index].mean()

        return mean_exp
    #=====================================
    def compute_marker_median_value_for_cell_type(
            self,
            marker: str,
            cell_type: str,
            cell_ann_col: str = "cell_annotations",
            ignore_zero: bool = True,
    ):
        """
            Note that this function takes two variables, the 
            marker and the cell type. However, under the
            assumption that a marker only points to a single
            cell type, the output of this function can be
            stored as part of a key-value pair
            (marker, cell type).
        """

        CA = cell_ann_col

        col_index = self.marker_to_column_idx[marker]
        mask = self.A.obs[CA] == cell_type
        values = self.A.X[mask, col_index]

        if sp.issparse(values):
            if ignore_zero:
                values = values.data
            else:
                values = values.toarray().squeeze()
        else:
            if ignore_zero:
                values = values[self.eps < values]

        return np.median(values)

    #=====================================
    def compute_marker_median_and_mad_for_all_cells(
            self,
            marker: str,
    ):
        """
        TODO: Remove this function since
        it appears that it is not being used.
        """

        if marker not in self.A.var_names:
            return None

        col_index = self.A.var.index.get_loc(marker)
        values = self.A.X[:, col_index]

        if sp.issparse(self.A.X):
            values = values.toarray().squeeze()

        median = np.median(values)
        mad    = np.median(values-median)

        return (median, mad)

    #=====================================
    def compute_mean_expression_from_indices(
            self,
            marker: str,
            indices: List,
    ):
        """
        TODO: Remove this function since
        it appears that it is not being used.
        """

        col_index = self.marker_to_column_idx[marker]
        mask = self.A.obs_names.isin(indices)
        vec = self.A.X[mask, col_index]

        return vec.mean()

    #=====================================
    def compute_median_and_mad_exp_from_indices(
            self,
            marker: str,
            indices: list,
            ignore_zero: bool = True,
            only_median: bool = False,
    ):
        """
        For a given marker, we compute the 
        the median and MAD expression for the
        given indices.
        TODO: Make sure you use the correct function to
        compute the gene expression at each node.
        """

        col_index = self.marker_to_column_idx[marker]
        mask = self.A.obs_names.isin(indices)
        values = self.A.X[mask, col_index]

        if sp.issparse(values):
            if ignore_zero:
                values = values.data
            else:
                values = values.toarray().squeeze()
        else:
            if ignore_zero:
                values = values[values.min() < values]

        median = np.median(values)
        values -= median
        mad = np.median(np.abs(values))

        if only_median:
            return median

        return (median, mad)

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
            elimination_ratio: float = -1.,
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
            

    #=================================================
    def check_if_cells_belong_to_group(
            self,
            cells: pd.Series,
            group: str,
            conversion_threshold: float = 0.9,
            cell_ann_col: str = "cell_annotations",
    ):
        """
        The "cells" parameter is a series that contains
        the cell types as values and the indices 
        correspond to the barcodes.

        We want to determine if a set of cells 
        belongs to a group previously defined
        in the CSV file "cell_markers".
        """
        #This is the question we are trying to
        #answer.
        belongs_to_group = False
        CA = cell_ann_col

        #What cells types belong to 
        #the given group?
        x = self.group_to_cell_types[group]
        cell_types_in_group = x

        #Now we are going to iterate over the
        #cells that belong to the majority
        #group. We do this to determine if 
        #the non-majority cells could qualify
        #as a member of the majority group by
        #using a marker for cells of the 
        #majority group.
        for cell_type in cell_types_in_group:
            if belongs_to_group:
                break
            print(f"Are they {cell_type}?")
            print("\t", "Marker ", "Reference ", "Measure ")
            markers = self.cell_type_to_markers[cell_type]

            for marker in markers:
                #Zeros were ignored during the calculations.
                # Why? Because this is the 
                #reference, and we want to make sure that
                #if something is above the reference, then
                #it is likely that it is a member of that
                #cell type.
                x = self.marker_to_median_value_for_cell_type[
                    marker][cell_type]
                marker_value = x
                if marker_value is None:
                    #Nothing to be done.
                    continue

                #Here we do not ignore the zeros. The 
                #reasoning is the same as above. We want to
                #have a high degree of confidence that 
                #these cells are actually of the alleged
                #cell type. #If we do not ignore the zeros,
                #it is more likely to produce a smaller 
                #value, making it harder to exceed the
                #reference.
                x=self.compute_median_and_mad_exp_from_indices(
                    marker, 
                    cells.index, 
                    ignore_zero=False,
                    only_median=True)
                expression_value = x
                print("\t", marker, marker_value, x)
                #Let X be the mean/median expression 
                #value of that marker for the
                #given minority.
                #Let Y be the mean/median expression
                #value of that same marker for
                #the cells in the sample that 
                #are known to express that
                #marker. If X is above Y 
                #multiplied by the conversion
                #threshold, then we add that
                #minority to the majority,
                if marker_value * conversion_threshold < x:
                    belongs_to_group = True
                    print("\t","To convert.")
                    break

        if belongs_to_group:
            self.A.obs[CA].loc[cells.index] = group
            return True
        else:
            print("===============================")
            print(f">>>Cells do not belong to {group}.")
            print("===============================")
            return False

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
        TODO: This function needs to be relocated.
        """
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
                x = self.compute_marker_median_value_for_cell_type(
                    marker, cell_type, ignore_zero=True)
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
    def count_nodes_above_threshold_for_marker(
            self,
            threshold: float,
            marker: str,
    ):
        """
        We traverse the tree using a depth-first approach.
        If the expression of the given marker at the 
        parent node is below the threshold, that node is
        ignored and consequently all its descendants.
        Otherwise, we count that node and push its 
        children into the stack.

        Note that previously the expression values at
        each node should have been computed.
        """
        S = [0]
        count = 0
        while 0 < len(S):
            node = S.pop()
            children = self.G.successors(node)
            exp_value = self.G.nodes[node][marker]
            if exp_value < threshold:
                continue
            count += 1
            for child in children:
                S.append(child)

        return count
    #=====================================
    def load_group_and_cell_type_data(
            self,
            cell_group_path: str,
            cell_ann_col: str = "cell_annotations",
    ):
        """
        Each cell type is associated to a group.
        For a given group we generate a list of 
        cell types.
        """
        self.t0 = clock()

        if not os.path.exists(cell_group_path):
            print(cell_group_path)
            raise ValueError("File does not exists.")

        # We assume all the fields in the data frame 
        # are strings. Hence, number used as labels 
        # will be considered as strings.
        df_cg = pd.read_csv(cell_group_path, dtype=str)
        print("===============================")
        print("Cell to Group file")
        print(df_cg)
        CA = cell_ann_col

        self.cell_type_to_group = {}
        self.cell_types_to_erase = []
        self.group_to_cell_types = ddict(list)

        self.cells_to_be_eliminated = None

        #Cell types
        self.CT = set(self.A.obs[CA])

        #Create the cell to group dictionary and
        #the group to cell dictionary
        for index, row in df_cg.iterrows():
            cell_type = row["Cell"]
            group = row["Group"]

            if cell_type not in self.CT:
                print(f"{cell_type=} not in data set.")
                continue

            if pd.isna(group):
                # If the group is not present, we 
                # assume that the cell type is its
                # own group.
                group = cell_type
            elif group == "0":
                # If the group is zero, then we
                # assume that the cell type has
                # to be eliminated or ignored.
                self.cell_types_to_erase.append(cell_type)
                continue

            self.cell_type_to_group[cell_type] = group
            self.group_to_cell_types[group].append(cell_type)

        self.tf = clock()
        delta = self.tf - self.t0
        txt = ("Elapsed time to load group and cell data: " + 
                f"{delta:.2f} seconds.")
        print(txt)

    #=====================================
    def load_marker_and_cell_type_data(
            self,
            cell_marker_path: str,
            keep_all_markers: bool = False,
    ):
        """
        Each marker is associated to a cell type.
        For a given cell type we generate a list of 
        potential markers to identify that cell.

        >>>Note: In general, first call
        >>>load_group_and_cell_type_data(cell_group_path)

        We use the dictionary
            self.marker_to_mad_threshold
        to select cells whose expression
        of a given marker is above or 
        below the given threshold.
        
        Example of the marker_and_cell_type.CSV file.

        Marker	Cell	    Direction	Threshold
        ITGAM	Monocytes	Above	    5
        CD14	Monocytes	Above	    40
        AIF1	Monocytes	Above	    30
        CSF1R	Monocytes	Above	    40
        CD163	Macrophages	Above	    40
        MRC1	Macrophages	Above	    10

        """

        self.t0 = clock()

        if not os.path.exists(cell_marker_path):
            print(cell_marker_path)
            raise ValueError("File does not exists.")

        # We assume all the fields in the data frame 
        # are strings. Hence, number used as labels 
        # will be considered as strings.
        # If thresholds are present, we need to make
        # sure that they are interpreted as floats.
        D = {"Marker":str,
             "Cell":str,
             "Direction":str,
             "Threshold":float,
             }
        df_cm = pd.read_csv(cell_marker_path, dtype=D)

        has_threshold = False
        if "Threshold" in df_cm.columns:
            has_threshold = True
            if "Direction" in df_cm.columns:
                pass
            else:
                txt = ("Need to specify if you are "
                       "interested in values above "
                       "or below the threshold.")
                raise ValueError(txt)

        print("===============================")
        print("Cell to Marker file")
        print(df_cm)

        #Create the cell to markers dictionary and
        #marker to value dictionary.
        self.cell_type_to_markers = ddict(list)
        self.marker_to_cell_types = ddict(list)
        self.marker_to_column_idx = {}
        self.column_idx_to_marker = {}
        self.marker_to_rank       = {}
        list_of_markers           = []
        list_of_column_idx        = []
        list_of_cell_types        = []

        self.marker_to_mad_threshold = {}
        self.marker_to_mad_direction = {}

        gene_names_lower = self.A.var_names.str.lower()

        for index, row in df_cm.iterrows():

            cell_type = row["Cell"]
            marker    = row["Marker"]
            marker_lower = marker.lower()

            if has_threshold:
                th = row["Threshold"]

                if pd.notna(th):
                    th = self.FDT(th)
                    dr = row["Direction"]
                    self.marker_to_mad_threshold[marker] = th

                    if dr.lower() == "below":
                        dr = "below"
                    elif dr.lower() == "above":
                        dr = "above"
                    else:
                        txt = ("Undefined direction.")
                        raise ValueError(txt)

                    self.marker_to_mad_direction[marker] = dr

            #In case some cell marker is not present
            #in the expression matrix.
            if marker_lower not in gene_names_lower:
                print(f"{marker=} not available.")
                continue
            else:
                col_index = gene_names_lower.get_loc(
                    marker_lower)
                # marker = self.A.var_names[col_index]

            # In case some cell type is not present
            # in the cell_type_to_group dictionary.
            # Note that this dictionary might not be
            # present. Hence, we check the presence 
            # of this attribute.
            if hasattr(self, "cell_type_to_group"):
                if cell_type not in self.cell_type_to_group:
                    print(f"{cell_type=} not available.")
                    if keep_all_markers:
                        print(f"{cell_type=} will be kept.")
                        print(f"{marker=} will be kept.")
                    else:
                        print(f"{marker=} will be ignored.")
                        continue

            self.cell_type_to_markers[cell_type].append(
                marker)
            self.marker_to_cell_types[marker].append(
                cell_type)

            #We do the following check to guarantee that 
            #the list of markers and column indices has 
            #no repetitions.

            # #Note that the list of cell types will be 
            # populated with the cell type that was
            # observed the first time we saw a given marker.

            if col_index in self.column_idx_to_marker:
                continue

            self.column_idx_to_marker[col_index] = marker
            self.marker_to_column_idx[marker] = col_index

            list_of_column_idx.append(col_index)
            list_of_markers.append(marker)
            list_of_cell_types.append(cell_type)

        for rank, marker in enumerate(list_of_markers):
            self.marker_to_rank[marker] = rank

        self.list_of_column_idx = np.array(
            list_of_column_idx, dtype=int)

        self.list_of_markers = np.array(list_of_markers)

        self.list_of_cell_types = np.array(
            list_of_cell_types)

        self.tf = clock()
        delta = self.tf - self.t0
        txt = ("Elapsed time to load marker and cell data: " + 
                f"{delta:.2f} seconds.")
        print(txt)

    #=====================================
    def populate_tree_with_mean_expression_for_all_markers(
            self,
            cell_marker_path: str,
            cell_group_path: Optional[str] = None,
        ):
        """
        This function uses a depth-first approach, where 
        we move from the leaf nodes all the way
        up to the root. 

        We use this traversal to make the computation of
        the mean expression at each node more efficient.

        We store two quantities for the gene expression
        information at each node.

        We store the cumulative sum and the mean expression.

        self.G.nodes[node][
            marker_label_sum] = sum_exp

        self.G.nodes[node][
            marker_label_mean] = mean_exp
        """

        if cell_group_path is not None:
            self.load_group_and_cell_type_data(
                cell_group_path)

        self.load_marker_and_cell_type_data(cell_marker_path)

        self.t0 = clock()

        n_nodes = self.G.number_of_nodes()
        if n_nodes == 0:
            raise ValueError("Empty graph.")

        n_markers = len(self.list_of_markers)
        if n_markers == 0:
            raise ValueError("Empty list of markers.")
        
        marker_sum_vec = np.zeros(n_markers, dtype=self.FDT)

        self.mean_exp_mtx = np.zeros(
            (n_nodes, n_markers), dtype=self.FDT)
        
        # Mean expression matrix
        # Node/Marker Marker 1 Marker 2 ...
        # Node 1
        # Node 2
        # ...

        print("Computing mean expression for all markers...")
        S = [0]
        visited = set()
        while 0 < len(S):
            node = S.pop()
            children = list(self.G.successors(node))
            n_cells = self.G.nodes[node]["size"]
            if node not in visited:
                visited.add(node)
                if 0 < len(children):
                    S.append(node)
                    for child in children:
                        S.append(child)
                    continue

                #Otherwise, it is a leaf node.
                nodes = [node]
                mask = self.A.obs["sp_cluster"].isin(nodes)
                #Generate indices for cells 
                #times list of markers.
                indices = np.ix_(mask,
                                 self.list_of_column_idx)
                values = self.X[indices]
                sum_exp_vec = values.sum(axis=0)
                mean_exp_vec = sum_exp_vec / n_cells
                # Fill the row of the expression 
                # matrix for the corresponding node.
                self.mean_exp_mtx[node] = mean_exp_vec

                #Assign the mean expression to each node.
                #We use an iterator.
                #Each node stores a dictionary.
                it = zip(self.list_of_markers,
                        mean_exp_vec,
                        sum_exp_vec)
                for marker, mean_exp, sum_exp in it:
                    marker_label_sum = marker + "_sum"
                    marker_label_mean = marker + "_mean"
                    self.G.nodes[node][
                        marker_label_sum] = sum_exp
                    self.G.nodes[node][
                        marker_label_mean] = mean_exp
                continue

            # Already visited this node.
            # This means we have already visited 
            # the children of this node.
            marker_sum_vec *= 0
            it = enumerate(self.list_of_markers)
            for idx, marker in it:
                marker_label_sum = marker + "_sum"
                marker_label_mean = marker + "_mean"
                self.G.nodes[node][marker_label_sum] = 0
                marker_sum = 0
                for child in children:
                    x = self.G.nodes[child][marker_label_sum]
                    self.G.nodes[node][marker_label_sum] += x
                    marker_sum_vec[idx] += x
                    marker_sum += x
                marker_mean = marker_sum / n_cells
                self.G.nodes[node][
                    marker_label_mean] = marker_mean
            mean_exp_vec = marker_sum_vec / n_cells
            self.mean_exp_mtx[node] = mean_exp_vec
            continue

        txt = ("Mean expression has been"
               " computed for all markers.")
        print(txt)

        self.tf = clock()
        delta = self.tf - self.t0
        txt = ("Elapsed time to compute node expression: " + 
                f"{delta:.2f} seconds.")
        print(txt)


    #=====================================
    def compute_node_expression_metadata(
            self,
        ):
        """
        Before calling this function we need to call:
        populate_tree_with_mean_expression_for_all_markers()

        Once we have computed the node expression
        for all the relevant markers, we can define
        a distribution for the expression of each marker.

        Compute the minimum,
        maximum, median, and mad after ignoring
        zeros for each distribution.

        The relevant data frames are:

        >>> (1)
        self.node_mad_dist_df
            col = marker + "_mad_bounds"
            col = marker + "_exp_bounds"
            col = marker + "_counts"
        self.node_mad_dist_df = pd.DataFrame(
            data = np.zeros((15,n_markers * 3)), 
            index = None,
            columns = L,
            dtype=self.FDT)

        Note that we have chosen 15 values to be
        consistent with too-many-cells interactive.
        These are the x-axis tick marks.

        >>> (2)
        self.node_exp_stats_df
        self.node_exp_stats_df = pd.DataFrame(
            np.zeros((n_markers, 7)),
            index = self.list_of_markers,
            columns = ["median",
                       "mad",
                       "min",
                       "max",
                       "min_mad",
                       "max_mad",
                       "delta"],
            dtype = self.FDT,
            )
        """
        self.t0 = clock()

        n_steps = 15

        # n_nodes, n_markers = self.mean_exp_mtx.shape
        n_markers = self.mean_exp_mtx.shape[1]
        #We create a sparse version of the node
        #expression matrix to eliminate the zeros. 
        #Note that we use a CSC format, since we 
        #plan to operate on the columns because they 
        #represent the genes for all the nodes.
        mean_exp_mtx_sp = sp.csc_array(self.mean_exp_mtx)

        #Vector with the pointers to identify the 
        #data for each column.
        indptr = mean_exp_mtx_sp.indptr
        # self.median_for_markers = np.zeros(n_markers,
        #                                   dtype=float)
        # self.mad_for_markers = np.zeros(n_markers,
        #                                   dtype=float)
        L = []
        for marker in self.list_of_markers:
            txt = marker
            txt += "_mad_bounds"
            L.append(txt)
            txt = marker
            txt += "_exp_bounds"
            L.append(txt)
            txt = marker
            txt += "_counts"
            L.append(txt)

        #TMCI uses 15 steps, and for each marker 
            # we store 3 pieces of information.
        self.node_mad_dist_df = pd.DataFrame(
            data = np.zeros((n_steps, n_markers * 3)), 
            index = None,
            columns = L,
            dtype=self.FDT)

        unique = self.node_mad_dist_df.columns.unique()
        n_unique_cols = len(unique)
        n_cols = len(self.node_mad_dist_df.columns)
        if n_unique_cols != n_cols:
            raise ValueError("Columns are not unique.")

        self.node_exp_stats_df = pd.DataFrame(
            np.zeros((n_markers, 7)),
            index = self.list_of_markers,
            columns = ["median",
                       "mad",
                       "min",
                       "max",
                       "min_mad",
                       "max_mad",
                       "delta"],
            dtype = self.FDT,
        )

        #We use an iterator.
        it = enumerate(zip(indptr[:-1], indptr[1:]))
        #We iterate over the columns, i.e., the markers.
        #No zeros.
        for idx, (start, end) in it:
            marker = self.list_of_markers[idx]
            data = mean_exp_mtx_sp.data[start:end]
            if len(data) == 0:
                continue
            median = np.median(data)
            min = np.min(data)
            max = np.max(data)
            mad = median_abs_deviation(data)
            min_mad = (min - median) / mad
            max_mad = (max - median) / mad
            delta =  (max - min) / mad / n_steps
            #Marker x Stats
            self.node_exp_stats_df.iloc[idx,0] = median
            self.node_exp_stats_df.iloc[idx,1] = mad
            self.node_exp_stats_df.iloc[idx,2] = min
            self.node_exp_stats_df.iloc[idx,3] = max
            self.node_exp_stats_df.iloc[idx,4] = min_mad
            self.node_exp_stats_df.iloc[idx,5] = max_mad
            self.node_exp_stats_df.iloc[idx,6] = delta
            madR = np.arange(min_mad,
                             max_mad - delta/4,
                             delta)
            col = marker + "_mad_bounds"
            self.node_mad_dist_df.loc[:,col] = madR

            col = marker + "_exp_bounds"
            expR = madR * mad + median
            self.node_mad_dist_df.loc[:,col] = expR

            col = marker + "_counts"
            for b_idx, threshold in enumerate(expR):
                mask = threshold < data
                count = mask.sum()
                self.node_mad_dist_df.loc[b_idx,col] = count


        print(self.node_exp_stats_df)
        # print(self.node_exp_stats_df.loc["FAP",:])
        # L = ["FAP_mad_bounds", "FAP_counts"]
        # print(self.node_mad_dist_df[L])

        self.tf = clock()
        delta = self.tf - self.t0
        txt = ("Elapsed time to compute node metadata: " + 
                f"{delta:.2f} seconds.")
        print(txt)

    #=====================================
    def select_cells_based_on_inequalities(
            self,
            cell_ann_col: str = "",
            n_markers_binary_threshold: int = 3,
        ) -> sc.AnnData:
        """

        For every marker we create a CSV file
        indicating which cells are above or
        below the given MAD threshold.

        In the file that was loaded during the call to
        load_marker_and_cell_type_data()
        the user can specify the MAD thresholds in a 
        column named "Threshold".

        Marker	Cell	    Direction	Threshold
        ITGAM	Monocytes	Above	    5
        CD14	Monocytes	Above	    40
        AIF1	Monocytes	Above	    30
        CSF1R	Monocytes	Above	    40
        CD163	Macrophages	Above	    40
        MRC1	Macrophages	Above	    10
        
        The user also has to specify the direction.

        For example:

        ITGAM	Monocytes	Above	    5
        ...

        means that we want to find cells whose
        ITGAM expression is Above the value 
        resulting from multiplying the MAD for
        ITGAM by 5 and adding to that the 
        corresponding median expression.

        We then create a CSV file for each marker with
        a MAD threshold. The cells that are above the
        threshold are written to the file. 
        
        We also indicate the node (cluster)
        membership, the expression value and
        the corresponding number of MADs from
        the median for each cell.

        Ex.

            Node	Expression	ExpressionAsMADs
        C1	82	    0.96506536	21.496866
	    C2  841	    1.4206275	32.1569

        An additional file is also created
        fname = "remaining_cells_after_intersection.csv"
        which has the intersection of all the cells
        whose expression satisfy the given thresholds.
        """

        n_cells = self.A.shape[0]
        mask_intersection = np.full(n_cells, True)
        n_markers = len(self.marker_to_mad_threshold)

        dict_iterator = self.marker_to_mad_threshold.items()

        # If we only have two markers, then we can create
        # a high-high, high-low, low-high and 
        # low-low classification.
        list_of_masks = []
        list_of_markers = []

        for marker, threshold in dict_iterator:
            # print(marker, threshold)
            list_of_markers.append(marker)
            direction = self.marker_to_mad_direction[marker]
            mad = self.node_exp_stats_df.loc[marker,
                                             "mad"]
            median = self.node_exp_stats_df.loc[marker,
                                                "median"]
            marker_exp = mad * threshold + median
            matrix_col = self.marker_to_column_idx[marker]
            vec = self.A.X[:,matrix_col]

            if sp.issparse(self.A.X):
                vec = vec.toarray().squeeze()

            if direction == "above":
                mask = marker_exp <= vec
            elif direction == "below":
                mask = vec <= marker_exp
            else:
                raise ValueError("Unexpected direction")

            if 1 < n_markers <= n_markers_binary_threshold:
                list_of_masks.append(mask)

            mask_intersection &= mask
            df = self.A.obs.sp_cluster.loc[mask]
            df = df.to_frame(name="Node")
            df["Expression"] = vec[mask]

            df["ExpressionAsMADs"] = (vec[mask] - median)
            df["ExpressionAsMADs"] /= mad
            if 0 < len(cell_ann_col):
                cell_types = self.A.obs[cell_ann_col]
                df["CellType"] = cell_types.loc[mask]

            fname  = f"{marker}_exp_{direction}"
            fname += f"_MAD_threshold.csv"
            fname = os.path.join(self.output, fname)
            df.to_csv(fname, index=True)
            
        fname = "remaining_cells_after_intersection.csv"
        fname = os.path.join(self.output, fname)
        df = self.A.obs.sp_cluster.loc[mask_intersection]
        df = df.to_frame(name="Node")

        self.A.obs["Intersection"] = mask_intersection

        if 0 < len(cell_ann_col):
            cell_types = self.A.obs[cell_ann_col]
            df["CellType"]=cell_types.loc[mask_intersection]

        #Write the file of intersections.
        df.to_csv(fname, index=True)

        if 1 < n_markers <= n_markers_binary_threshold:

            df = self.A.obs.sp_cluster
            df = df.to_frame(name="Node")

            if 0 < len(cell_ann_col):
                df["CellType"] = self.A.obs[cell_ann_col]

            df["Class"] = ""
            mask = np.ones(n_cells, dtype=bool)
            vec_classes = df["Class"].values
            exp_states  = ("High", "Low")

            def add_category(states: Tuple[str],
                             level: int,
                             vec_bool: ArrayLike,
                             state_name: str,
                             vec_str: ArrayLike,
                ):

                if level == n_markers:
                    #Assign the final name
                    #to the cells that satisfy 
                    #the class condition.
                    vec_str[vec_bool] = state_name

                    return None


                marker_name = list_of_markers[level]


                for state in states:

                    if 0 == level:
                        new_state_name = state_name
                    else:
                        new_state_name = f"{state_name}-"

                    new_state_name += f"{marker_name}-{state}"
                    new_vec_bool = vec_bool.copy()

                    if state == "High":
                        new_vec_bool &= list_of_masks[level]

                    elif state == "Low":
                        new_vec_bool &= ~list_of_masks[level]

                    else:
                        raise ValueError("Unknown state.")

                    add_category(states,
                                 level+1,
                                 new_vec_bool,
                                 new_state_name,
                                 vec_str)

            #Initiate the recursive call.
            add_category(exp_states,
                         0,
                         mask,
                         "",
                         vec_classes,
            )

            #Update the column of classes.
            df["Class"] = vec_classes

            print(df["Class"].value_counts())
            print(df["Class"].value_counts(normalize=True))

            fname = "cell_classes_from_markers.csv"
            fname = os.path.join(self.output, fname)

            #Write the file of 
            df.to_csv(fname, index=True)

            self.A.obs["MarkerClass"] = vec_classes

        #Return an AnnData object whose cells
        return self.A[mask_intersection].copy()









    #=====================================
    def count_connected_nodes_above_threshold_for_attribute(
            self,
            threshold: float,
            attribute: str,
    ):
        """
        This function will proceed using a breadth-first
        approach. Everytime a node has the attribute 
        above the threshold, we increase the count by one
        and add the children to the deque.
        """
        DQ = deque()
        DQ.append(0)
        count = 0

        while 0 < len(DQ):
            node = DQ.popleft()
            value = self.G.nodes[node][attribute]

            if value < threshold:
                continue
            count += 1
            children = self.G.successors(node)

            for child in children:
                DQ.append(child)

        return count

    #=====================================
    def compute_cell_types(
            self,
            mad_threshold: float = 1.,
    ):
        """
        TODO: Identify questionable cells and doublets.
        This function aims to fully annotate a collection
        of cells based on the provided cell markers.
        """
        n_markers = len(self.list_of_markers)
        mad_exp_df = pd.DataFrame(
            index=self.A.obs_names,
            columns=self.list_of_markers,
            )
        above_threshold_df = pd.DataFrame(
            index=self.A.obs_names,
            columns=self.list_of_markers,
            )
        sorted_marker_df = pd.DataFrame(
            index=self.A.obs_names,
            columns=np.arange(1,n_markers+1),
            )
        sorted_cell_type_df = pd.DataFrame(
            index=self.A.obs_names,
            columns=np.arange(1,n_markers+1),
            )
        sorted_groups_df = pd.DataFrame(
            index=self.A.obs_names,
            columns=np.arange(1,n_markers+1),
            )

        for k, marker in enumerate(self.list_of_markers):
            mad_exp_df[marker]         = ""
            above_threshold_df[marker] = ""
            sorted_marker_df[k+1]      = ""
            sorted_cell_type_df[k+1]   = ""
            sorted_groups_df[k+1]      = ""


        list_of_groups = []
        for cell_type in self.list_of_cell_types:
            group = self.cell_type_to_group[cell_type]
            list_of_groups.append(group)

        self.list_of_groups = np.array(list_of_groups)


        list_of_status = []
        groups = []

        # Iterate over cells.
        n_cells = self.X.shape[0]
        for row_idx, row in enumerate(tqdm(self.X,
                                           total=n_cells)):

            # if row_idx == 10:
            #     break

            #Normal status
            status = "Normal"

            if sp.issparse(self.X):
                vec = row.toarray().squeeze()
            else:
                vec = row.copy()

            m_exp = vec[self.list_of_column_idx]
            m_exp -= self.node_exp_stats_df.loc[:,"median"]
            m_exp /= self.node_exp_stats_df.loc[:,"mad"]

            #Compute who is above the MAD threshold.
            above_th = mad_threshold < m_exp

            mad_exp_df.iloc[row_idx] = m_exp
            above_threshold_df.iloc[row_idx] = above_th

            # Get the indices going 
            # from the highest to the lowest
            # deviation from the median.
            indices = np.argsort(m_exp)
            indices = indices[::-1]

            sorted_markers = self.list_of_markers[indices]
            sorted_marker_df.iloc[row_idx] = sorted_markers

            # Note that by construction, for each marker
            # we have a (most likely) unique cell type
            # present in the list self.list_of_cell_types.

            sorted_cell_types = self.list_of_cell_types[
                indices]
            x = sorted_cell_types
            sorted_cell_type_df.iloc[row_idx] = x

            sorted_groups = self.list_of_groups[indices]
            x = sorted_groups
            sorted_groups_df.iloc[row_idx] = x

            # Here we check if a cell belongs to 
            # two different cell groups. By definitions,
            # this groups are disjoint. For example,
            # we do not expect a cell to belong to
            # the Fibroblast category and the Lymphocyte
            # category.

            #Sort the above threshold candidates
            above_th = above_th[indices]
            if above_th.any():
                pass
            else:
                #This cell is low for all markers.
                status = "AllLow"
                list_of_status.append(status)
                continue

            # Filter for those groups that are above
            # threshold.

            filtered_groups = sorted_groups[above_th]
            unique_groups = np.unique(filtered_groups)

            if 1 < len(unique_groups):
                status = "MultipleGroups"

            list_of_status.append(status)


        fname = "MAD_exp_values.csv"
        fname = os.path.join(self.output, fname)
        mad_exp_df.to_csv(fname, index=True)

        fname = "above_MAD_threshold_status.csv"
        fname = os.path.join(self.output, fname)
        above_threshold_df.to_csv(fname, index=True)

        fname = "sorted_markers.csv"
        fname = os.path.join(self.output, fname)
        sorted_marker_df.to_csv(fname, index=True)

        fname = "sorted_cell_types.csv"
        fname = os.path.join(self.output, fname)
        sorted_cell_type_df.to_csv(fname, index=True)

        fname = "sorted_groups.csv"
        fname = os.path.join(self.output, fname)
        sorted_groups_df.to_csv(fname, index=True)

        tmc_ct = "TMC_cell_type"
        self.A.obs[tmc_ct] = sorted_cell_type_df[1]
        self.generate_cell_annotation_file(
            tmc_ct, tag="cell_types")

        tmc_marker = "TMC_marker"
        self.A.obs[tmc_marker] = sorted_marker_df[1]
        self.generate_cell_annotation_file(
            tmc_marker, tag="cell_markers")

        tmc_group = "TMC_group"
        self.A.obs[tmc_group] = sorted_groups_df[1]
        self.generate_cell_annotation_file(
            tmc_group, tag="cell_groups")

        status_str = "Status"
        self.A.obs[tmc_group] = list_of_status
        self.generate_cell_annotation_file(
            status_str, tag="group_status")

    #=====================================
    def plot_marker_distributions(
            self,
    ):
        """
        For every marker in the list of markers
        we create an interactive plot (HTML) of 
        the marker expression distribution 
        using MADs (median absolute deviations).
        The y-axis indicates how many nodes are
        above a specific threshold. The x-axis
        describes the thresholds.
        """
        fig = go.Figure()
        n_markers = len(self.list_of_markers)
        max_markers = n_markers

        def create_marker_mask(n_markers, idx):
            L = []
            for k in range(n_markers):
                if k == idx:
                    L.append(True)
                else:
                    L.append(False)
            return L

        list_of_dict = []
        mask = create_marker_mask(max_markers, -1)
        D = dict(
            label="",
            method="update",
            args=[{"visible": mask},
                  {"title": "Gene",
                   "annotations": []}]
        )
        list_of_dict.append(D)
        for rank, marker in enumerate(self.list_of_markers):

            if rank == max_markers:
                break

            mask = create_marker_mask(max_markers, rank)

            x_label = marker + "_mad_bounds"
            y_label = marker + "_counts"

            x_data = self.node_mad_dist_df[x_label]
            y_data = self.node_mad_dist_df[y_label]

            fig.add_trace(
                go.Scatter(x=x_data,
                           y=y_data,
                           name=marker,
                           line=dict(color="#0000FF"),
                           visible=False,
                           )
            )

            D = dict(
                label=marker,
                method="update",
                args=[{"visible": mask},
                      {"title": marker,
                       "annotations": []}]
            )

            list_of_dict.append(D)
        
        fig.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    buttons = list_of_dict,
                )
            ]
        )

        title = "Node expression distribution"
        fig.update_layout(
            title=title,
            xaxis_title = "threshold (MADs from the median)",
            yaxis_title = "Number of nodes above threshold",
            font=dict(family="monospace", size=18),           
            hoverlabel=dict(
                font=dict(family="monospace", size=18)
                ),
        )
        # fig.update_layout(title_text="Node exp. distribution")
        fname = "distributions.html"
        fname = os.path.join(self.output, fname)
        fig.write_html(fname)


        # remote = [
        # 1364, 427, 350, 292, 249, 199, 156, 127, 93, 55,
        # 33, 12, 3, 2, 1,]

    #=====================================
    def compute_branch_diameter(
            self,
            node: int,
    ):
        """
        """
        self.t0 = clock()

        n_nodes = self.G.number_of_nodes()
        if n_nodes == 0:
            raise ValueError("Empty graph.")

        txt = f"Computing the diameter for branch at {node=}"
        print(txt)
        S = [node]
        start = node
        visited = set()
        while 0 < len(S):
            node = S.pop()
            children = list(self.G.successors(node))
            n_cells = self.G.nodes[node]["size"]
            if node not in visited:
                visited.add(node)
                if 0 < len(children):
                    S.append(node)
                    for child in children:
                        S.append(child)
                    continue

                #Otherwise, it is a leaf node.
                #It has no modularity.
                self.G.nodes[node]["D"] = (0,[node])
                continue

            # We have already visited this node.
            # This means we have already processed
            # the children.
            (Q1, L1) = self.G.nodes[children[0]]["D"]
            (Q2, L2) = self.G.nodes[children[1]]["D"]
            Q_current = self.G.nodes[node]["Q"] / 2
            if Q2 < Q1:
                Q_current += Q1
                L = L1.copy()
            else:
                Q_current += Q2
                L = L2.copy()
            L.append(node)
            T = (Q_current,L)
            self.G.nodes[node]["D"] = T
            continue

        #Post-processing
        node = start
        children = list(self.G.successors(node))
        Q_current = self.G.nodes[node]["Q"]
        child1 = children[0]
        child2 = children[1]

        (Q1, L1) = self.G.nodes[child1]["D"]
        (Q2, L2) = self.G.nodes[child2]["D"]
        Q_total = Q_current + Q1 + Q2

        print(f"======================================")
        print(f"Diameter for branch at {node}: {Q_total}")
        print(f"Maximum attained between nodes: "
              f"{L1[0]} and {L2[0]}")
        print(f"======================================")
        print("Summary:")
        print(f"Distance from {child1} to {L1[0]}: {Q1}")
        print(f"Distance from {child2} to {L2[0]}: {Q2}")
        print(f"Modularity at {node}: {Q_current}")
        print(f"Distance from {L1[0]} to {L2[0]}: {Q_total}")
        print(f"Path from {L1[0]} to {child1}: {L1}")
        print(f"Path from {L2[0]} to {child2}: {L2}")

        self.tf = clock()
        delta = self.tf - self.t0
        txt = ("Elapsed time to compute node expression: " + 
                f"{delta:.2f} seconds.")
        print(txt)

    #=====================================
    def annotate_with_celltypist(
            self,
            cell_ann_col: str = "cell_annotations",
            model_kind: str = "Immune_All_High",
            use_majority_voting: bool = False,
    ):
        """
        Instead of using celltypist separately,
        we facilitate its use through TMC 
        a la Python.
        """
        B = sc.AnnData(self.A.X.copy())

        vec = B.X.expm1().sum(axis=1) - 1e4
        vec = np.abs(vec)

        if 0.01 < np.max(vec):
            print("Pre-processing matrix.")
            sc.pp.normalize_total(B, target_sum=1e4)
            sc.pp.log1p(B)

        print("Matrix is ready.")
            
        B.obs_names = self.A.obs_names
        B.var_names = self.A.var_names

        model_str = model_kind
        if model_kind.endswith(".pkl"):
            pass
        else:
            model_str += ".pkl"

        CT.models.download_models(force_update=False,
                                  model = model_str)
        ct_model = CT.models.Model.load(model = model_str)
        prediction = CT.annotate(
            B,
            model = ct_model,
            majority_voting = use_majority_voting,
        )
        col = "predicted_labels"
        if use_majority_voting:
            col = "majority_voting"
        B.obs[cell_ann_col] = prediction.predicted_labels[col]
        fname = "celltypist_annotations.csv"
        fname = os.path.join(self.output, fname)
        B.obs[cell_ann_col].to_csv(fname, index = True)
        self.A.obs[cell_ann_col] = B.obs[cell_ann_col]
        print(self.A.obs[cell_ann_col])

    #=====================================
    def quantify_modularity_heterogeneity(
            self,
            list_of_branches: List[int] = [0],
            tag: str = "modularity_distribution",
            file_format: str = "pdf",
            show_column_totals: bool = False,
            use_log_y: bool = False,
            color = "blue",
    ):
        """
        This function plots the modularity distribution 
        for all the nodes that belong to the branches 
        specified in the list of branches.
        """

        if 0 == len(list_of_branches):
            print("Nothing to be done.")
            return

        nodes = set()
        total_n_cells = 0
        print("Working with branches:")
        for branch in list_of_branches:
            print(branch)
            nodes.add(branch)
            nodes.update(
                list(nx.descendants(self.G, branch))
            )
            # total_n_cells += self.G.nodes[branch]["size"]
        
        list_of_Q     = []
        list_of_nodes = []
        vec_cells     = []

        for node in nodes:
            if 0 < self.G.out_degree(node):
                #This is not a leaf node.
                Q = self.G.nodes[node]["Q"]
                list_of_nodes.append(node)
                list_of_Q.append(Q)
            else:
                #This is a leaf node.
                n_cells = self.G.nodes[node]["size"]
                vec_cells.append(n_cells)
                total_n_cells += n_cells

        print(f"Total # of cells: {total_n_cells}")
        
        # ============== Diversity ==============
        vec_cells  = np.array(vec_cells, dtype=self.FDT)
        vec_cells /= total_n_cells

        # q = 0 ==> Total number of species
        n_species = len(vec_cells)
        q_0 = n_species

        # q = 1 ==> Shannon's diversity index
        shannon = entropy(vec_cells)
        q_1 = np.exp(shannon)

        # q = 2 ==> Simpson's diversity index
        simpson = np.sum(vec_cells**2)
        q_2 = 1 / simpson

        # q = infty ==> Max
        max_p = np.max(vec_cells)
        q_inf = 1 / np.max(vec_cells)

        indices = ["Richness",
                   "Shannon",
                   "Simpson",
                   "MaxProp",
                   "q = 0",
                   "q = 1",
                   "q = 2",
                   "q = inf",
                   ]

        results = [n_species, shannon, simpson, max_p,
                   q_0, q_1, q_2, q_inf]

        df = pd.DataFrame(results,
                          index=indices,
                          columns = ["value"])

        print(df)
        fname = "diversity_indices.csv"
        fname = os.path.join(self.output, fname)
        df.to_csv(fname, index=True)


        # ============== Modularity ==============
        df = pd.DataFrame(list_of_Q,
                          index = list_of_nodes,
                          columns = ["modularity"] )
        df.index.name = "node"

        fname = "modularity_distribution.csv"
        fname = os.path.join(self.output, fname)
        df.to_csv(fname, index=True)

        Q_total = df["modularity"].sum()

        # ============== Distributions ==============
        fig,ax = plt.subplots()
        counts, edges, bars = ax.hist(list_of_Q, color=color)
        if show_column_totals:
            plt.bar_label(bars)
        ax.set_xlabel("Modularity (Q)")
        ax.set_ylabel("# of nodes")
        txt = f"Cumulative modularity: {Q_total:.2E}"
        ax.set_title(txt)
        plt.ticklabel_format(style="sci",
                             axis="x",
                             scilimits=(0,0))
        if use_log_y:
            plt.yscale("log")
        fname = tag + "." + file_format
        fname = os.path.join(self.output, fname)
        fig.savefig(fname, bbox_inches="tight")

    #=====================================
    def easy_plot(
        self,
        cell_ann_col: str,
        draw_modularity: bool = False,
        draw_node_numbers: bool = False,
        ):
        """
        This function can be used to plot 
        your toomanycells tree if you already
        have generated the outputs.
        """

        self.generate_cell_annotation_file(cell_ann_col)

        labels_path = os.path.join(
            self.output,
            "cell_annotation_labels.csv")

        haskell = TMCHaskell(
            self.output,
            tmc_tree_path = self.output,
            path_to_cell_annotations=labels_path,
            draw_modularity = draw_modularity,
            draw_node_numbers = draw_node_numbers,
        )

        haskell.run()

    #=====================================
    def plot_with_tmc_a_la_haskell(
            self,
            tmc_tree_path: str,
            matrix_path: str = "",
            list_of_genes: List[str] = [],
            use_threshold: bool = False,
            high_low_colors: List[str] = ["purple",
                                          "red",
                                          "blue",
                                          "aqua"],
            gene_colors: List[str] = [],
            annotation_colors: List[str] = [],
            method: str = "MadMedian",
            tree_file_name: str = "tree.svg",
            threshold: float = 1.5,
            saturation: float = 1.5,
            output_folder: str = "tmc_haskell",
            feature_column: str = "1",
            draw_modularity: bool = False,
            path_to_cell_annotations: str = "",
            draw_node_numbers: bool = False,
                                   ):
        haskell = TMCHaskell(
            self.output,
            tmc_tree_path,
            matrix_path,
            list_of_genes,
            use_threshold,
            high_low_colors,
            gene_colors,
            annotation_colors,
            method,
            tree_file_name,
            threshold,
            saturation,
            output_folder,
            feature_column,
            draw_modularity,
            path_to_cell_annotations,
            draw_node_numbers,
        )

        haskell.run()

    #=====================================
    def clean_tree(
            self,
            cell_ann_col: str = "cell_annotations",
    ):
        """
        """

        self.tmcGraph.eliminate_cell_type_outliers(
            cell_ann_col)
        self.A = self.tmcGraph.A
        self.tmcGraph.rebuild_graph_after_removing_cells()
        self.tmcGraph.rebuild_tree_from_graph()
        self.tmcGraph.store_outputs()

    #=====================================
    def load_graph(
            self,
            json_fname: str = "graph.json",
            load_clusters_file: bool = True,
            store_in_uns_dict: bool = False,
        ):
        """
        Load graph from JSON file.
        """

        self.tmcGraph.load_graph(json_fname,
                                 load_clusters_file)
        # After loading the graph, the set of leaf
        # nodes gets populated internally.
        x = self.tmcGraph.set_of_leaf_nodes

        self.G = self.tmcGraph.G
        self.set_of_leaf_nodes = x

        if store_in_uns_dict:
            self.A.uns["tmc_graph"] = self.G
            self.A.uns["tmc_leaf_nodes"] = x

    #=====================================
    def redefine_output_folder(
            self,
            path: str,
            ):
        """
        In case one wants to change the 
        output folder for all the relevant
        objects.
        """
        os.makedirs(path, exist_ok = True)
        self.output = path
        self.tmcGraph.output = path

    #=====================================
    def isolate_cells_from_branches(
            self,
            path_to_csv_file: str = "",
            list_of_branches: List[int] = [],
            branch_column: str = "node",
            generate_cell_id_file: bool = False,
            cell_id_fname: str = "cell_ids.csv",
        ):

        """
        Isolate the cells from all the given
        branches (collection of nodes).
        The function returns an AnnData object
        with only the isolated cells.
        """

        mask = self.tmcGraph.isolate_cells_from_branches(
            path_to_csv_file,
            list_of_branches,
            branch_column,
        )

        adata = self.A[mask].copy()

        if generate_cell_id_file:

            fname = os.path.join(self.output, cell_id_fname)
            adata.obs["sp_cluster"].to_csv(fname, index=True)

        return adata

    #=====================================
    def plot_embedding(
        self,
        color_column: str,
        color_map: pd.DataFrame,
        file_name: str,
        ):
        """
        """
        obj = MultiPlotter(self.A, self.output)

        obj.plot_embedding(
            color_column = color_column,
            color_map = color_map,
            file_name = file_name,)

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
        Use the majority to annotate the cells
        that belong to a specific category.
        """
        obj = CellAnnotation(self.G, self.A, self.output)
        obj.homogenize_leaf_nodes(
            cell_ann_col,
            upper_threshold,
            change_below_this,
            minimum_level,
            change_all,
            labels_to_change,
        )


    #====END=OF=CLASS=====================

