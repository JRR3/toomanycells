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
import subprocess
import numpy as np
import scanpy as sc
import pandas as pd
# import seaborn as sb
from tqdm import tqdm
import networkx as nx
import matplotlib as mpl
import celltypist as CT
from typing import Union
from typing import Optional
from os.path import dirname
from scipy.io import mmread
from scipy.io import mmwrite
from collections import deque
from scipy import sparse as sp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from time import perf_counter as clock
from sklearn.preprocessing import normalize
from collections import defaultdict as ddict
from scipy.stats import median_abs_deviation
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.sparse.linalg import eigsh as Eigen_Hermitian
from sklearn.feature_extraction.text import TfidfTransformer

#Matplotlib parameters.
mpl.use("agg")
mpl.rcParams["figure.dpi"]=600
mpl.rcParams["pdf.fonttype"]=42

font = {'weight' : 'normal', 'size'   : 18}
mpl.rc("font", **font)

sys.path.insert(0, dirname(__file__))
from common import MultiIndexList

#=====================================================
class TooManyCells:
    """
    This class focuses on one aspect of the original \
        Too-Many-Cells tool, the clustering.\ 
        Features such as normalization, \
        dimensionality reduction and many others can be \
        applied using functions from libraries like \ 
        Scanpy, or they can be implemented locally. This \
        implementation also allows the possibility of \
        new features with respect to the original \
        Too-Many-Cells. For example, imagine you want to \
        continue partitioning fibroblasts until you have \
        at most a given number of cells, even if the \
        modularity becomes negative, but for CD8+ T-cells \
        you do not want to have partitions with less \
        than 100 cells. This can be easily implemented \
        with a few conditions using the cell annotations \
        in the .obs data frame of the AnnData object.\

    With regards to visualization, we recommend \
        using the too-many-cells-interactive tool. \
        You can find it at:\ 
        https://github.com/schwartzlab-methods/\
        too-many-cells-interactive.git\
        Once installed, you can use the function \
        visualize_with_tmc_interactive() to \
        generate the visualization. You will need \
        path to the installation folder of \
        too-many-cells-interactive.


    """
    #=================================================
    def __init__(self,
            input: Union[sc.AnnData, str],
            output: Optional[str] = "",
            input_is_matrix_market: Optional[bool] = False,
            use_full_matrix: Optional[bool] = False,
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

        if isinstance(input, TooManyCells):

            #Clone the given TooManyCells object.
            self.A = input.A.copy()
            self.G = input.G.copy()
            S = input.set_of_leaf_nodes.copy()
            self.set_of_leaf_nodes = S

        elif isinstance(input, str):
            self.source = os.path.abspath(input)
            if self.source.endswith('.h5ad'):
                self.t0 = clock()
                self.A = sc.read_h5ad(self.source)
                self.tf = clock()
                delta = self.tf - self.t0
                txt = ('Elapsed time for loading: ' +
                        f'{delta:.2f} seconds.')
                print(txt)
            else:
                if input_is_matrix_market:
                    self.convert_mm_from_source_to_anndata()
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

        if not os.path.exists(output):
            os.makedirs(output)

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

        self.FDT = np.float64
        txt = f"Data will be treated as: {self.FDT}."
        print(txt)

        #Create a copy to avoid direct modifications
        #of the original count matrix X.
        #Note that we are making sure that the 
        #sparse matrix has the CSR format. This
        #is relevant when we normalize.
        if sp.issparse(self.A.X):
            #Compute the density of the matrix
            rho = self.A.X.nnz / np.prod(self.A.X.shape)
            #If more than 50% of the matrix is occupied,
            #we generate a dense version of the matrix.
            sparse_threshold = 0.50
            if use_full_matrix or sparse_threshold < rho:
                self.is_sparse = False
                self.X = self.A.X.toarray()
                txt = ("Using a dense representation" 
                       " of the count matrix.")
                print(txt)
                self.X = self.X.astype(self.FDT)
            else:
                self.is_sparse = True
                #Make sure we use a CSR array format.
                self.X = sp.csr_array(self.A.X,
                                       dtype=self.FDT,
                                       copy=True)
        else:
            #The matrix is dense.
            print("The matrix is dense.")
            self.is_sparse = False
            self.X = self.A.X.copy()
            self.X = self.X.astype(self.FDT)

        self.n_cells, self.n_genes = self.A.shape

        if self.n_cells < 3:
            raise ValueError("Too few observations (cells).")

        print(self.A)

        #Location of the matrix data for TMCI
        self.tmci_mtx_dir = ""

        self.spectral_clustering_has_been_called = False

        self.cells_to_be_eliminated = None

        # We use a deque to offer the possibility of breadth-
        # versus depth-first. Our current implementation
        # uses depth-first to be consistent with the 
        # numbering scheme of TooManyCellsInteractive.
        self.DQ = deque()

        #Map a node to the path in the
        #binary tree that connects the
        #root node to the given node.
        self.node_to_path = {}

        #Map a node to a list of indices
        #that provide access to the JSON
        #structure.
        self.node_to_j_index = {}

        #the JSON structure representation
        #of the tree.
        self.J = MultiIndexList()

        self.node_counter = 0

        #The threshold for modularity to 
        #accept a given partition of a set
        #of cells.
        self.eps = 1e-9

        self.use_twopi_cmd   = True
        self.verbose_mode    = False

    #=====================================
    def normalize_sparse_rows(self):
        """
        Divide each row of the count matrix by the \
            given norm. Note that this function \
            assumes that the matrix is in the \
            compressed sparse row format.
        """

        print("Normalizing rows.")


        for i, row in enumerate(self.X):
            data = row.data.copy()
            row_norm  = np.linalg.norm(
                data, ord=self.similarity_norm)
            data /= row_norm
            start = self.X.indptr[i]
            end   = self.X.indptr[i+1]
            self.X.data[start:end] = data

    #=====================================
    def normalize_dense_rows(self):
        """
        Divide each row of the count matrix by the \
            given norm. Note that this function \
            assumes that the matrix is dense.
        """

        print('Normalizing rows.')

        for row in self.X:
            row /= np.linalg.norm(row,
                                  ord=self.similarity_norm)

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
    def estimate_n_of_iterations(self) -> int:
        """
        We assume a model of the form \
        number_of_iter = const * N^exponent \
        where N is the number of cells.
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
            shift_similarity_matrix:Optional[float] = 0,
            normalize_rows:Optional[bool] = False,
            similarity_function:Optional[str]="cosine_sparse",
            similarity_norm: Optional[float] = 2,
            similarity_power: Optional[float] = 1,
            similarity_gamma: Optional[float] = None,
            use_eig_decomp: Optional[bool] = False,
            use_tf_idf: Optional[bool] = False,
            tf_idf_norm: Optional[str] = None,
            tf_idf_smooth: Optional[str] = True,
            svd_algorithm: Optional[str] = "randomized"):
        """
        This function computes the partitions of the \
                initial cell population and continues \
                until the modularity of the newly \
                created partitions is nonpositive.
        """

        svd_algorithms = ["randomized","arpack"]
        if svd_algorithm not in svd_algorithms:
            raise ValueError("Unexpected SVD algorithm.")

        if similarity_norm < 1:
            raise ValueError("Unexpected similarity norm.")
        self.similarity_norm = similarity_norm

        if similarity_gamma is None:
            # gamma = 1 / (number of features)
            similarity_gamma = 1 / self.X.shape[1]
        elif similarity_gamma <= 0:
            raise ValueError("Unexpected similarity gamma.")

        if similarity_power <= 0:
            raise ValueError("Unexpected similarity power.")

        similarity_functions = []
        similarity_functions.append("cosine_sparse")
        similarity_functions.append("cosine")
        similarity_functions.append("neg_exp")
        similarity_functions.append("laplacian")
        similarity_functions.append("gaussian")
        similarity_functions.append("div_by_sum")
        if similarity_function not in similarity_functions:
            raise ValueError("Unexpected similarity fun.")


        #In case the user wants to call this function again.
        self.spectral_clustering_has_been_called = True

        #TF-IDF section
        if use_tf_idf:

            t0 = clock()
            print("Using inverse document frequency (IDF).")

            if tf_idf_norm is None:
                pass 
            else:
                print("Using term frequency normalization.")
                tf_idf_norms = ["l2","l1"]
                if tf_idf_norm not in tf_idf_norms:
                    raise ValueError("Unexpected tf norm.")

            tf_idf_obj = TfidfTransformer(
                norm=tf_idf_norm,
                smooth_idf=tf_idf_smooth)

            self.X = tf_idf_obj.fit_transform(self.X)
            if self.is_sparse:
                pass
            else:
                #If the matrix was originally dense
                #and the tf_idf function changed it
                #to sparse, then convert to dense.
                if sp.issparse(self.X):
                    self.X = self.X.toarray()

            tf = clock()
            delta = tf - t0
            txt = ("Elapsed time for IDF build: " +
                    f"{delta:.2f} seconds.")
            print(txt)

        #Normalization section
        use_cos_sp = similarity_function == "cosine_sparse"
        use_dbs = similarity_function == "div_by_sum"
        if normalize_rows or use_cos_sp or use_dbs:
            t0 = clock()

            if self.is_sparse:
                self.normalize_sparse_rows()
            else:
                self.normalize_dense_rows()

            tf = clock()
            delta = tf - t0
            txt = ("Elapsed time for normalization: " +
                    f"{delta:.2f} seconds.")
            print(txt)

        #Similarity section.
        print(f"Working with {similarity_function=}")

        if similarity_function == "cosine_sparse":

            self.trunc_SVD = TruncatedSVD(
                    n_components=2,
                    n_iter=5,
                    algorithm=svd_algorithm)

        else:
            #Use a similarity function different from
            #the cosine_sparse similarity function.

            t0 = clock()
            print("Building similarity matrix ...")
            n_rows = self.X.shape[0]
            max_workers = os.cpu_count()
            n_workers = 1
            if n_rows < 500:
                pass
            elif n_rows < 5000:
                if 8 < max_workers:
                    n_workers = 8
            elif n_rows < 50000:
                if 16 < max_workers:
                    n_workers = 16
            else:
                if 25 < max_workers:
                    n_workers = 25
            print(f"Using {n_workers=}.")

        if similarity_function == "cosine_sparse":
            pass
        elif similarity_function == "cosine":
            #( x @ y ) / ( ||x|| * ||y|| )
            def sim_fun(x,y):
                cos_sim = x @ y
                x_norm = np.linalg.norm(x, ord=2)
                y_norm = np.linalg.norm(y, ord=2)
                cos_sim /= (x_norm * y_norm)
                return cos_sim

            self.X = pairwise_kernels(self.X,
                                        metric="cosine",
                                        n_jobs=n_workers)

        elif similarity_function == "neg_exp":
            #exp(-||x-y||^power * gamma)
            def sim_fun(x,y):
                delta = np.linalg.norm(
                    x-y, ord=similarity_norm)
                delta = np.power(delta, similarity_power)
                return np.exp(-delta * similarity_gamma)

            self.X = pairwise_kernels(
                self.X,
                metric=sim_fun,
                n_jobs=n_workers)

        elif similarity_function == "laplacian":
            #exp(-||x-y||^power * gamma)
            def sim_fun(x,y):
                delta = np.linalg.norm(
                    x-y, ord=1)
                delta = np.power(delta, 1)
                return np.exp(-delta * similarity_gamma)

            self.X = pairwise_kernels(
                self.X,
                metric="laplacian",
                n_jobs=n_workers,
                gamma = similarity_gamma)

        elif similarity_function == "gaussian":
            #exp(-||x-y||^power * gamma)
            def sim_fun(x,y):
                delta = np.linalg.norm(
                    x-y, ord=2)
                delta = np.power(delta, 2)
                return np.exp(-delta * similarity_gamma)

            self.X = pairwise_kernels(
                self.X,
                metric="rbf",
                n_jobs=n_workers,
                gamma = similarity_gamma)

        elif similarity_function == "div_by_sum":
            #1 - ( ||x-y|| / (||x|| + ||y||) )^power
            #The rows should have been previously normalized.
            def sim_fun(x,y):
                delta = np.linalg.norm(
                    x-y, ord=similarity_norm)
                x_norm = np.linalg.norm(
                    x, ord=similarity_norm)
                y_norm = np.linalg.norm(
                    y, ord=similarity_norm)
                delta /= (x_norm + y_norm)
                delta = np.power(delta, 1)
                value =  1 - delta
                return value

            if self.similarity_norm == 1:
                lp_norm = "l1"
            elif self.similarity_norm == 2:
                lp_norm = "l2"
            else:
                txt = "Similarity norm should be 1 or 2."
                raise ValueError(txt)

            self.X = pairwise_distances(self.X,
                                        metric=lp_norm,
                                        n_jobs=n_workers)
            self.X *= -0.5
            self.X += 1

        if similarity_function != "cosine_sparse":

            if shift_similarity_matrix != 0:
                print(f"Similarity matrix will be shifted.")
                print(f"Shift: {shift_similarity_matrix}.")
                self.X += shift_similarity_matrix
            
            print("Similarity matrix has been built.")
            tf = clock()
            delta = tf - t0
            delta /= 60
            txt = ("Elapsed time for similarity build: " +
                    f"{delta:.2f} minutes.")
            print(txt)


        self.use_eig_decomp = use_eig_decomp

        self.t0 = clock()

        #===========================================
        #=============Main=Loop=====================
        #===========================================
        node_id = self.node_counter

        #Initialize the array of cells to partition
        rows = np.array(range(self.X.shape[0]))

        #Initialize the deque
        # self.DQ.append((rows, None))
        # self.DQ.append(rows)

        #Initialize the graph
        self.G.add_node(node_id, size=len(rows))

        #Path to reach root node.
        self.node_to_path[node_id] = str(node_id)

        #Indices to reach root node.
        self.node_to_j_index[node_id] = (1,)

        #Update the node counter
        self.node_counter += 1

        #============STEP=1================Cluster(0)

        p_node_id = node_id

        if similarity_function == "cosine_sparse":
            Q,S = self.compute_partition_for_sp(rows)
        else:
            Q,S = self.compute_partition_for_gen(rows)

        if self.eps < Q:
            #Modularity is above threshold, and
            #thus each partition will be 
            #inserted into the deque.

            D = self.modularity_to_json(Q)

            #Update json index
            self.J.append(D)
            self.J.append([])
            # self.J.append([[],[]])
            # j_index = (1,)

            self.G.nodes[node_id]["Q"] = Q

            for indices in S:
                T = (indices, p_node_id)
                self.DQ.append(T)

        else:
            #Modularity is below threshold and 
            #therefore this partition will not
            #be considered.
            txt = ("All cells belong" 
                    " to the same partition.")
            print(txt)
            return

        max_n_iter = self.estimate_n_of_iterations()

        self.print_message_before_clustering()

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
                # determine if the node will 
                if similarity_function == "cosine_sparse":
                    Q,S = self.compute_partition_for_sp(rows)
                else:
                    Q,S = self.compute_partition_for_gen(rows)

                # If the parent node is 0, then the path is
                # "0".
                current_path = self.node_to_path[p_node_id]

                #Update path for the new node
                new_path = current_path 
                new_path += '/' + str(node_id) 
                self.node_to_path[node_id]=new_path

                # If the parent node is 0, then j_index is
                # (1,)
                j_index = self.node_to_j_index[p_node_id]

                n_stored_blocks = len(self.J[j_index])
                self.J[j_index].append([])
                #Update the j_index. For example, if
                #j_index = (1,) and no blocks have been
                #stored, then the new j_index is (1,0).
                #Otherwise, it is (1,1).
                j_index += (n_stored_blocks,)

                #Include new node into the graph.
                self.G.add_node(node_id, size=len(rows))

                #Include new edge into the graph.
                self.G.add_edge(p_node_id, node_id)

                if self.eps < Q:
                    #Modularity is above threshold, and
                    #thus each partition will be 
                    #inserted into the deque.

                    D = self.modularity_to_json(Q)
                    self.J[j_index].append(D)
                    self.J[j_index].append([])
                    j_index += (1,)

                    # We only store the modularity of nodes
                    # whose modularity is above threshold.
                    self.G.nodes[node_id]["Q"] = Q

                    # Update the j_index for the newly 
                    # created node. (1,0,1)
                    self.node_to_j_index[node_id] = j_index

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
                    #Also include the path to reach that node.
                    c = self.cluster_column_index
                    self.A.obs.iloc[rows, c] = node_id

                    reversed_path = self.reverse_path(
                        new_path)
                    p = self.path_column_index
                    self.A.obs.iloc[rows, p] = reversed_path

                    self.set_of_leaf_nodes.add(node_id)

                    #Update the JSON structure for 
                    #a leaf node.
                    L = self.cells_to_json(rows)
                    self.J[j_index].append(L)
                    self.J[j_index].append([])

                pbar.update()

            #==============END OF WHILE==============
            pbar.total = pbar.n
            self.final_n_iter = pbar.n
            pbar.refresh()

        self.tf = clock()
        self.delta_clustering = self.tf - self.t0
        self.delta_clustering /= 60
        txt = ("Elapsed time for clustering: " +
                f"{self.delta_clustering:.2f} minutes.")
        print(txt)

    #=====================================
    def compute_partition_for_sp(self, rows: np.ndarray
    ) -> tuple:
    #) -> tuple[float, np.ndarray]:
        """
        Compute the partition of the given set\
            of cells. The rows input \
            contains the indices of the \
            rows we are to partition. \
            The algorithm computes a truncated \
            SVD and the corresponding modularity \
            of the newly created communities.
        """

        if self.verbose_mode:
            print(f'I was given: {rows=}')

        partition = []
        Q = 0

        n_rows = len(rows) 
        #print(f"Number of cells: {n_rows}")

        #If the number of rows is less than 3,
        #we keep the cluster as it is.
        if n_rows < 3:
            return (Q, partition)

        B = self.X[rows,:]
        ones = np.ones(n_rows)
        partial_row_sums = B.T.dot(ones)
        #1^T @ B @ B^T @ 1 = (B^T @ 1)^T @ (B^T @ 1)
        L = partial_row_sums @ partial_row_sums - n_rows
        #These are the row sums of the similarity matrix
        row_sums = B @ partial_row_sums
        #Check if we have negative entries before computing
        #the square root.
        # if  neg_row_sums or self.use_eig_decomp:
        zero_row_sums_mask = np.abs(row_sums) < self.eps
        has_zero_row_sums = zero_row_sums_mask.any()
        has_neg_row_sums = (row_sums < -self.eps).any() 

        if has_zero_row_sums:
            print("We have zero row sums.")
            row_sums[zero_row_sums_mask] = 0

        if has_neg_row_sums and has_zero_row_sums:
            txt = "This matrix cannot be processed."
            print(txt)
            txt = "Cannot have negative and zero row sums."
            raise ValueError(txt)

        if  has_neg_row_sums:
            #This means we cannot use the fast approach
            #We'll have to build a dense representation
            # of the similarity matrix.
            if 5000 < n_rows:
                print("The row sums are negative.")
                print("We will use a full eigen decomp.")
                print(f"The block size is {n_rows}.")
                print("Warning ...")
                txt = "This operation is very expensive."
                print(txt)
            laplacian_mtx  = B @ B.T
            row_sums_mtx   = sp.diags(row_sums)
            laplacian_mtx  = row_sums_mtx - laplacian_mtx

            #This is a very expensive operation
            #since it computes all the eigenvectors.
            inv_row_sums   = 1/row_sums
            inv_row_sums   = sp.diags(inv_row_sums)
            laplacian_mtx  = inv_row_sums @ laplacian_mtx
            eig_obj = np.linalg.eig(laplacian_mtx)
            eig_vals = eig_obj.eigenvalues
            eig_vecs = eig_obj.eigenvectors
            idx = np.argsort(np.abs(np.real(eig_vals)))
            #Get the index of the second smallest eigenvalue.
            idx = idx[1]
            W = np.real(eig_vecs[:,idx])
            W = np.squeeze(np.asarray(W))

        elif self.use_eig_decomp or has_zero_row_sums:
            laplacian_mtx  = B @ B.T
            row_sums_mtx   = sp.diags(row_sums)
            laplacian_mtx  = row_sums_mtx - laplacian_mtx
            try:
                #if the row sums are negative, this 
                #step could fail.
                E_obj = Eigen_Hermitian(laplacian_mtx,
                                        k=2,
                                        M=row_sums_mtx,
                                        sigma=0,
                                        which="LM")
                eigen_val_abs = np.abs(E_obj[0])
                #Identify the eigenvalue with the
                #largest magnitude.
                idx = np.argmax(eigen_val_abs)
                #Choose the eigenvector corresponding
                # to the eigenvalue with the 
                # largest magnitude.
                eigen_vectors = E_obj[1]
                W = eigen_vectors[:,idx]
            except:
                #This is a very expensive operation
                #since it computes all the eigenvectors.
                if 5000 < n_rows:
                    print("We will use a full eigen decomp.")
                    print(f"The block size is {n_rows}.")
                    print("Warning ...")
                    txt = "This operation is very expensive."
                    print(txt)
                inv_row_sums   = 1/row_sums
                inv_row_sums   = sp.diags(inv_row_sums)
                laplacian_mtx  = inv_row_sums @ laplacian_mtx
                eig_obj = np.linalg.eig(laplacian_mtx)
                eig_vals = eig_obj.eigenvalues
                eig_vecs = eig_obj.eigenvectors
                idx = np.argsort(np.abs(np.real(eig_vals)))
                idx = idx[1]
                W = np.real(eig_vecs[:,idx])
                W = np.squeeze(np.asarray(W))


        else:
            #This is the fast approach.
            #It is fast in the sense that the 
            #operations are faster if the matrix
            #is sparse, i.e., O(n) nonzero entries.

            d = 1/np.sqrt(row_sums)
            D = sp.diags(d)
            C = D @ B
            W = self.trunc_SVD.fit_transform(C)
            singular_values = self.trunc_SVD.singular_values_
            idx = np.argsort(singular_values)
            #Get the singular vector corresponding to the
            #second largest singular value.
            W = W[:,idx[0]]


        mask_c1 = 0 < W
        mask_c2 = ~mask_c1

        #If one partition has all the elements
        #then return with Q = 0.
        if mask_c1.all() or mask_c2.all():
            return (Q, partition)

        masks = [mask_c1, mask_c2]

        for mask in masks:
            n_rows_msk = mask.sum()
            partition.append(rows[mask])
            ones_msk = ones * mask
            row_sums_msk = B.T.dot(ones_msk)
            O_c = row_sums_msk @ row_sums_msk - n_rows_msk
            L_c = ones_msk @ row_sums  - n_rows_msk
            Q += O_c / L - (L_c / L)**2

        if self.verbose_mode:
            print(f'{Q=}')
            print(f'I found: {partition=}')
            print('===========================')

        return (Q, partition)

    #=====================================
    def compute_partition_for_gen(self, rows: np.ndarray
    ) -> tuple:
    #) -> tuple[float, np.ndarray]:
        """
        Compute the partition of the given set\
            of cells. The rows input \
            contains the indices of the \
            rows we are to partition. \
            The algorithm computes a truncated \
            SVD and the corresponding modularity \
            of the newly created communities.
        """

        if self.verbose_mode:
            print(f'I was given: {rows=}')

        partition = []
        Q = 0

        n_rows = len(rows) 
        #print(f"Number of cells: {n_rows}")

        #If the number of rows is less than 3,
        #we keep the cluster as it is.
        if n_rows < 3:
            return (Q, partition)

        S = self.X[np.ix_(rows, rows)]
        ones = np.ones(n_rows)
        row_sums = S.dot(ones)
        row_sums_mtx   = sp.diags(row_sums)
        laplacian_mtx  = row_sums_mtx - S
        L = np.sum(row_sums) - n_rows

        zero_row_sums_mask = np.abs(row_sums) < self.eps
        has_zero_row_sums = zero_row_sums_mask.any()
        has_neg_row_sums = (row_sums < -self.eps).any() 

        if has_zero_row_sums:
            print("We have zero row sums.")
            row_sums[zero_row_sums_mask] = 0

        if has_neg_row_sums and has_zero_row_sums:
            txt = "This matrix cannot be processed."
            print(txt)
            txt = "Cannot have negative and zero row sums."
            raise ValueError(txt)

        if has_neg_row_sums:
            #This is a very expensive operation
            #since it computes all the eigenvectors.
            if 5000 < n_rows:
                print("The row sums are negative.")
                print("We will use a full eigen decomp.")
                print(f"The block size is {n_rows}.")
                print("Warning ...")
                txt = "This operation is very expensive."
                print(txt)
            inv_row_sums   = 1/row_sums
            inv_row_sums   = sp.diags(inv_row_sums)
            laplacian_mtx  = inv_row_sums @ laplacian_mtx
            eig_obj = np.linalg.eig(laplacian_mtx)
            eig_vals = eig_obj.eigenvalues
            eig_vecs = eig_obj.eigenvectors
            idx = np.argsort(np.abs(np.real(eig_vals)))
            idx = idx[1]
            W = np.real(eig_vecs[:,idx])
            W = np.squeeze(np.asarray(W))

        else:
            #Nonnegative row sums.
            try:
                E_obj = Eigen_Hermitian(laplacian_mtx,
                                        k=2,
                                        M=row_sums_mtx,
                                        sigma=0,
                                        which="LM")
                eigen_val_abs = np.abs(E_obj[0])
                #Identify the eigenvalue with the
                #largest magnitude.
                idx = np.argmax(eigen_val_abs)
                #Choose the eigenvector corresponding
                # to the eigenvalue with the 
                # largest magnitude.
                eigen_vectors = E_obj[1]
                W = eigen_vectors[:,idx]

            except:
                #This is a very expensive operation
                #since it computes all the eigenvectors.
                if 5000 < n_rows:
                    print("We will use a full eigen decomp.")
                    print(f"The block size is {n_rows}.")
                    print("Warning ...")
                    txt = "This operation is very expensive."
                    print(txt)
                inv_row_sums   = 1/row_sums
                inv_row_sums   = sp.diags(inv_row_sums)
                laplacian_mtx  = inv_row_sums @ laplacian_mtx
                eig_obj = np.linalg.eig(laplacian_mtx)
                eig_vals = eig_obj.eigenvalues
                eig_vecs = eig_obj.eigenvectors
                idx = np.argsort(np.abs(np.real(eig_vals)))
                #Get the index of the second smallest 
                #eigenvalue.
                idx = idx[1]
                W = np.real(eig_vecs[:,idx])
                W = np.squeeze(np.asarray(W))


        mask_c1 = 0 < W
        mask_c2 = ~mask_c1

        #If one partition has all the elements
        #then return with Q = 0.
        if mask_c1.all() or mask_c2.all():
            return (Q, partition)

        masks = [mask_c1, mask_c2]

        for mask in masks:
            n_rows_msk = mask.sum()
            partition.append(rows[mask])
            ones_msk = ones * mask
            row_sums_msk = S @ ones_msk
            O_c = ones_msk @ row_sums_msk - n_rows_msk
            L_c = ones_msk @ row_sums  - n_rows_msk
            Q += O_c / L - (L_c / L)**2

        if self.verbose_mode:
            print(f'{Q=}')
            print(f'I found: {partition=}')
            print('===========================')

        return (Q, partition)

    #=====================================
    def store_outputs(
            self,
            cell_ann_col: Optional[str] = "cell_annotations",
            store_tree_svg: Optional[bool] = True,
            ):
        """
        Store the outputs and plot the branching tree.

        File outputs:

        cluster_list.json: The json file containing a list 
        of clusters. 

        cluster_tree.json: The json file containing the 
        output tree in a recursive format.

        graph.dot: A dot file of the tree. It includes the 
        modularity and the size.

        node_info.csv: Size and modularity of each node.

        clusters.csv: The cluster membership for each cell.

        """

        self.t0 = clock()


        fname = 'graph.dot'
        dot_fname = os.path.join(self.output, fname)

        if store_tree_svg:
            nx.nx_agraph.write_dot(self.G, dot_fname)

        #Write cell to node data frame.
        self.write_cell_assignment_to_csv()
        self.convert_graph_to_json()
        self.write_cluster_list_to_json()

        #Store the cell annotations in the output folder.
        if 0 < len(cell_ann_col):
            if cell_ann_col in self.A.obs.columns:
                self.generate_cell_annotation_file(
                    cell_ann_col)
            else:
                txt = "Annotation column does not exists."
                #raise ValueError(txt)

        print(self.G)

        #Number of cells for each node
        size_list = []
        #Modularity for each node
        Q_list = []
        #Node label
        node_list = []

        for node, attr in self.G.nodes(data=True):
            node_list.append(node)
            size_list.append(attr['size'])
            if 'Q' in attr:
                Q_list.append(attr['Q'])
            else:
                Q_list.append(np.nan)

        #Write node information to CSV
        D = {'node': node_list, 'size':size_list, 'Q':Q_list}
        df = pd.DataFrame(D)
        fname = 'node_info.csv'
        fname = os.path.join(self.output, fname)
        df.to_csv(fname, index=False)

        if self.use_twopi_cmd and store_tree_svg:
            self.plot_radial_tree_from_dot_file()

        self.tf = clock()
        delta = self.tf - self.t0
        txt = ('Elapsed time for storing outputs: ' +
                f'{delta:.2f} seconds.')
        print(txt)


    #=====================================
    def convert_mm_from_source_to_anndata(self):
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

        print('Loading data from .mtx file.')
        print('Note that we assume the format:')
        print('genes=rows and cells=columns.')

        fname = None
        for f in os.listdir(self.source):
            if f.endswith('.mtx'):
                fname = f
                break

        if fname is None:
            raise ValueError('.mtx file not found.')

        fname = os.path.join(self.source, fname)
        mat = mmread(fname)
        #Remember that the input matrix has
        #genes for rows and cells for columns.
        #Thus, just transpose.
        self.A = mat.T.tocsr()

        fname = 'barcodes.tsv'
        print(f'Loading {fname}')
        fname = os.path.join(self.source, fname)
        df_barcodes = pd.read_csv(
                fname, delimiter='\t', header=None)
        barcodes = df_barcodes.loc[:,0].tolist()

        fname = 'genes.tsv'
        print(f'Loading {fname}')
        fname = os.path.join(self.source, fname)
        df_genes = pd.read_csv(
                fname, delimiter='\t', header=None)
        genes = df_genes.loc[:,0].tolist()

        self.A = sc.AnnData(self.A)
        self.A.obs_names = barcodes
        self.A.var_names = genes

        self.tf = clock()
        delta = self.tf - self.t0
        txt = ('Elapsed time for loading: ' + 
                f'{delta:.2f} seconds.')

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
    def write_cluster_list_to_json(self):
        """
        This function creates a JSON file that indicates \
            the assignment of each cell to a specific \
            cluster. 
        """
        fname = 'cluster_list.json'
        fname = os.path.join(self.output, fname)
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

        s = str(master_list)
        replace_dict = {' ':'', "'":'"'}
        pattern = '|'.join(replace_dict.keys())
        regexp  = re.compile(pattern)
        fun = lambda x: replace_dict[x.group(0)] 
        obj = regexp.sub(fun, s)
        with open(fname, 'w') as output_file:
            output_file.write(obj)


    #=====================================
    def convert_graph_to_json(self):
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
        s = str(self.J)
        replace_dict = {' ':'', 'None':'null', "'":'"'}
        pattern = '|'.join(replace_dict.keys())
        regexp  = re.compile(pattern)
        fun = lambda x: replace_dict[x.group(0)] 
        obj = regexp.sub(fun, s)
        with open(fname, 'w') as output_file:
            output_file.write(obj)

    #=====================================
    def generate_cell_annotation_file(self,
            cell_ann_col: Optional[str] = "cell_annotations",
            tag: Optional[str]="cell_annotation_labels"
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
        ca.index.names = ['item']
        ca = ca.rename('label')
        ca.to_csv(fname, index=True)

    #=====================================
    def create_data_for_tmci(
            self,
            tmci_mtx_dir: Optional[str] = "tmci_mtx_data",
            list_of_genes: Optional[list] = [],
            path_to_genes: Optional[str] = "",
            create_matrix: Optional[bool] = True,
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

        var_names = []
        col_indices = []

        if 0 < len(path_to_genes):
            df = pd.read_csv(path_to_genes, header=0)
            #The first column should contain the genes.
            list_of_genes = df.iloc[:,0].to_list()

        if 0 < len(list_of_genes):

            for gene in list_of_genes:
                if gene not in self.A.var.index:
                    continue
                var_names.append(gene)
                col_index = self.A.var.index.get_loc(gene)
                col_indices.append(col_index)
    
            G_mtx = self.A.X[:,col_indices]

        else:
            #If not list is provided, use all the genes.
            var_names = self.A.var_names
            G_mtx = self.A.X

        L = [var_names,var_names]
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
            index=False)

        # Matrix
        if create_matrix:
            matrix_f = "matrix.mtx"
            matrix_f = os.path.join(self.tmci_mtx_dir,
                                    matrix_f)
            mmwrite(matrix_f, sp.coo_matrix(G_mtx.T))

    #=====================================
    def visualize_with_tmc_interactive(self,
            path_to_tmc_interactive: str,
            use_column_for_labels: Optional[str] = "",
            port: Optional[int] = 9991,
            include_matrix_data: Optional[bool] = False,
            tmci_mtx_dir: Optional[str] = "",
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

        mmwrite(mtx_path, sp.coo_matrix(m))

    #=====================================
    def load_graph(
            self,
            dot_fname: Optional[str]="",
            ):
        """
        Load the dot file. Note that when loading the data,
        the attributes of each node are assumed to be 
        strings. Hence, we have to convert them to 
        integers for the case of number of cells, and to
        float for the case of modularity.
        """

        self.t0 = clock()


        if len(dot_fname) == 0:
            fname = 'graph.dot'
            dot_fname = os.path.join(self.output, fname)

        if not os.path.exists(dot_fname):
            raise ValueError("File does not exists.")

        self.G = nx.nx_agraph.read_dot(dot_fname)
        self.G = nx.DiGraph(self.G)
        n_nodes = self.G.number_of_nodes()

        # Change string labels to integers.
        D = {}
        for k in range(n_nodes):
            D[str(k)] = k

        self.G = nx.relabel_nodes(self.G, D, copy=True)

        #We convert the number of cells of each node to
        #integer. We also convert the modularity to float.
        for node in self.G.nodes():
            size = self.G.nodes[node]["size"]
            self.G.nodes[node]["size"] = int(size)
            if "Q" in self.G.nodes[node]:
                Q = self.G.nodes[node]["Q"]
                self.G.nodes[node]["Q"] = self.FDT(Q)

        print(self.G)

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
            genes: Union[list, str],
            output_list: Optional[bool] = False,
            ):

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

        if isinstance(genes, str):
            is_string = True
            list_of_genes = [genes]
        else:
            list_of_genes = genes

        exp_vec = []
        mean_exp = 0

        for gene in list_of_genes:

            if gene not in self.A.var.index:
                raise ValueError(f"{gene=} was not found.")

            col_index = self.A.var.index.get_loc(gene)

            mask = self.A.obs["sp_cluster"].isin(nodes)
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
            cluster_file_path: Optional[str]="",
            ):
        """
        Load the cluster file.
        """

        self.t0 = clock()

        if 0 < len(cluster_file_path):
            cluster_fname = cluster_file_path

        else:
            fname = 'clusters.csv'
            cluster_fname = os.path.join(self.output, fname)

        if not os.path.exists(cluster_fname):
            raise ValueError("File does not exists.")

        df = pd.read_csv(cluster_fname, index_col=0)
        self.A.obs["sp_cluster"] = df["cluster"]

        self.set_of_leaf_nodes = set(df["cluster"])

        self.tf = clock()
        delta = self.tf - self.t0
        txt = ("Elapsed time to load cluster file: " + 
                f"{delta:.2f} seconds.")


    #=====================================
    def plot_expression_from_node_x_to_node_y(
            self,
            x: int,
            y: int,
            genes: Union[list, str],
            ):
        """
        For a given pair of nodes x and y, we compute the \
            gene expression path along the path connecting\
            those nodes. 
        Make sure that property set_of_leaf_nodes is\
            populated with the correct information.
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
    def plot_radial_tree_from_dot_file(
            self,
            dot_fname: Optional[str] = "",
    ):
        if len(dot_fname) == 0:
            fname = 'graph.dot'
            dot_fname = os.path.join(self.output, fname)
        else:
            if not os.path.exists(dot_fname):
                raise ValueError("DOT file not found.")

        fname = 'output_graph.svg'
        fname = os.path.join(self.output, fname)

        command = ['twopi',
                '-Groot=0',
                '-Goverlap=true',
                '-Granksep=2',
                '-Tsvg',
                dot_fname,
                '>',
                fname,
                ]
        command = ' '.join(command)
        p = subprocess.call(command, shell=True)

    #=====================================
    def compute_marker_mean_value_for_cell(
            self,
            marker: str,
            cell: str,
            cell_ann_col: Optional[str] = "cell_annotations",
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
            cell_ann_col: Optional[str] = "cell_annotations",
            ignore_zero: Optional[bool] = True,
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
            indices: list,
    ):

        col_index = self.marker_to_column_idx[marker]
        mask = self.A.obs_names.isin(indices)
        vec = self.A.X[mask, col_index]

        return vec.mean()

    #=====================================
    def compute_median_and_mad_exp_from_indices(
            self,
            marker: str,
            indices: list,
            ignore_zero: Optional[bool] = True,
            only_median: Optional[bool] = False,
    ):
        """
        Aug 13, 2024
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
            cell_ann_col: Optional[str] = "cell_annotations",
            clean_threshold: Optional[float] = 0.8,
            favor_minorities: Optional[bool] = False,
            conversion_threshold: Optional[float] = 0.9,
            confirmation_threshold: Optional[float] = 0.9,
            elimination_ratio: Optional[float] = -1.,
            homogeneous_leafs: Optional[bool] = False,
            follow_parent: Optional[bool] = False,
            follow_majority: Optional[bool] = False,
            no_mixtures: Optional[bool] = False,
            storage_path: Optional[str] = "stable_tree",
            max_n_iter: Optional[int] = 100,
    ):
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
            

    #=====================================
    def check_leaf_homogeneity(
            self,
            cell_ann_col: Optional[str] = "cell_annotations",
    ):

        CA = cell_ann_col

        for node in self.G.nodes:
            if 0 < self.G.out_degree(node):
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
    def homogenize_leaf_nodes(
            self,
            cell_ann_col: Optional[str] = "cell_annotations",
            follow_parent: Optional[bool] = False,
            follow_majority: Optional[bool] = False,
    ):

        if follow_parent == follow_majority:
            print("Homogeneous leafs strategy:")
            raise ValueError("Strategy has to be unique.")

        CA = cell_ann_col
        elim_set = set()

        # for node in self.G.nodes:
            # if 0 < self.G.out_degree(node):
            #     continue

        if len(self.set_of_leaf_nodes) == 0:
            raise ValueError("Empty set of leaf nodes.")

        for node in self.set_of_leaf_nodes:
            parent = next(self.G.predecessors(node))
            #print(f"{parent}-->{node}")

            #Child
            mask = self.A.obs["sp_cluster"].isin([node])
            S = self.A.obs[CA].loc[mask]
            vc = S.value_counts(normalize=True)
            child_majority = vc.index[0]
            child_ratio = vc.iloc[0]

            if child_ratio == 1:
                #The node is already homogeneous
                continue

            if follow_parent:
                #Parent
                mask = self.A.obs["sp_cluster"].isin([parent])
                S = self.A.obs[CA].loc[mask]
                vc = S.value_counts(normalize=True)
                parent_majority = vc.index[0]
    
                #Who is different from the parent?
                mask = S != parent_majority
                Q = S.loc[mask]
                elim_set.update(Q.index)
                continue

            if follow_majority:
                #Who is different from the child's majority?
                mask = S != child_majority
                Q = S.loc[mask]
                elim_set.update(Q.index)
                continue

        return elim_set

                    
    #=====================================
    def erase_cells_from_json_file(
            self,
            json_file_path: Optional[str] = "",
            target_json_file_path: Optional[str] = "",
            modify_anndata: Optional[bool] = False,
    ):

        #{'_barcode':
        #{'unCell': 'CAGCTGGCACGGTAGA-176476-OM'},
        #'_cellRow': {'unRow': 29978}}
        if len(json_file_path) == 0:
            folder = "tmc_outputs"
            source = os.getcwd()
            fname = "cluster_tree.json"
            source_fname = os.path.join(
                source, folder, fname)
            fname = "pruned_cluster_tree.json"
            target_fname = os.path.join(
                source, folder, fname)

        else:
            source_fname = json_file_path
            target_fname = target_json_file_path

        with open(source_fname, "r") as f:
            source = f.readline()

        list_of_regexp = []
        replace_dict = {}

        for k, barcode in enumerate(
            self.cells_to_be_eliminated):

            txt = '[{][^{]+[{][a-zA-Z":]+[ ]?["]'
            # txt += "(?P<barcode>"
            txt += barcode
            # txt += ")"
            txt += '["][}][^}]+[}]{2}[,]?'
            list_of_regexp.append(txt)
            replace_dict[barcode] = ""

        pattern = "|".join(list_of_regexp)
        regexp = re.compile(pattern)
        fun = lambda x: ""
        obj = regexp.sub(fun, source)

        with open(target_fname, "w") as output_file:
            output_file.write(obj)

        if modify_anndata:
            #This will eliminate all the
            #cells labeled with an X from the 
            #anndata object.
            mask = self.A.obs.index.isin(
                self.cells_to_be_eliminated)
            self.A = self.A[~mask].copy()


    #=================================================
    def check_if_cells_belong_to_group(
            self,
            cells: pd.Series,
            group: str,
            conversion_threshold: Optional[float] = 0.9,
            cell_ann_col: Optional[str] = "cell_annotations",
    ):
        """
        The cells parameter is a series that contains
        the cell types as values and the indices 
        correspond to the barcodes.
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
            cell_ann_col: Optional[str] = "cell_annotations",
            clean_threshold: Optional[float] = 0.8,
            favor_minorities: Optional[bool] = False,
            conversion_threshold: Optional[float] = 0.9,
            confirmation_threshold: Optional[float] = 0.9,
            elimination_ratio: Optional[float] = -1.,
            homogeneous_leafs: Optional[bool] = False,
            follow_parent: Optional[bool] = False,
            follow_majority: Optional[bool] = False,
            no_mixtures: Optional[bool] = False,
    ):
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
            cell_ann_col: Optional[str] = "cell_annotations",
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

        df_cg = pd.read_csv(cell_group_path)
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
                group = cell_type
            elif group == "0":
                self.cell_types_to_erase.append(cell_type)
                continue

            self.cell_type_to_group[cell_type] = group
            self.group_to_cell_types[group].append(cell_type)

        self.tf = clock()
        delta = self.tf - self.t0
        txt = ("Elapsed time to load group and cell data: " + 
                f"{delta:.2f} seconds.")

    #=====================================
    def load_marker_and_cell_type_data(
            self,
            cell_marker_path: str,
            keep_all_markers: Optional[bool] = False,
    ):
        """
        Each marker is associated to a cell type.
        For a given cell type we generate a list of 
        potential markers to identify that cell.
        """

        self.t0 = clock()

        if not os.path.exists(cell_marker_path):
            print(cell_marker_path)
            raise ValueError("File does not exists.")

        df_cm = pd.read_csv(cell_marker_path)
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

        for index, row in df_cm.iterrows():

            cell_type = row["Cell"]
            marker = row["Marker"]

            #In case some cell marker is not present
            #in the expression matrix.
            if marker not in self.A.var_names:
                print(f"{marker=} not available.")
                continue

            #In case some cell type is not present
            #in the expression matrix.
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

            col_index = self.A.var.index.get_loc(marker)
            #We do the following check to guarantee that 
            #the list of markers and column indices has 
            #no repetitions.
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

    #=====================================
    def populate_tree_with_mean_expression_for_all_markers(
            self,
            cell_group_path: str,
            cell_marker_path: str,
    ):
        """
        """
        self.load_group_and_cell_type_data(cell_group_path)
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

        # print(self.mean_exp_mtx.shape)
        # print(self.A.X.shape)
        # for node in self.G.nodes():
        #     for marker in self.list_of_markers:
        #         mk_sum = marker + "_sum"
        #         mk_mean = marker + "_mean"
        #         self.G.nodes[node][mk_sum] = 0
        #         self.G.nodes[node][mk_mean] = 0

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


    #=====================================
    def populate_tree_with_mean_expression_for_all_markers_s(
            self,
            cell_group_path: str,
            cell_marker_path: str,
    ):
        """
        This is the slower original version of the function
        populate_tree_with_mean_expression_for_all_markers()
        """
        self.t0 = clock()

        self.load_group_and_cell_type_data(cell_group_path)

        self.load_marker_and_cell_type_data(
            cell_marker_path,
            keep_all_markers=True)

        n_nodes = self.G.number_of_nodes()
        if n_nodes == 0:
            raise ValueError("Empty graph.")

        n_markers = len(self.list_of_markers)
        if n_markers == 0:
            raise ValueError("Empty list of markers.")

        self.mean_exp_mtx = np.zeros(
            (n_nodes, n_markers), dtype=self.FDT)

        # print(self.mean_exp_mtx.shape)
        # print(self.A.X.shape)

        print("Computing mean expression for all markers...")
        for node in tqdm(self.G.nodes()):

            nodes = nx.descendants(self.G, node)

            if len(nodes) == 0:
                #This is a leaf node.
                nodes = [node]
            else:
                #Make sure these are leaf nodes.
                nodes = self.set_of_leaf_nodes.intersection(
                    nodes)

            mask = self.A.obs["sp_cluster"].isin(nodes)
            #Generate indices for cells 
            #times list of markers.
            indices = np.ix_(mask, self.list_of_column_idx)
            values = self.X[indices]
            mean_exp_vec = values.mean(axis=0)
            #Fill the row of the expression matrix for the
            #corresponding node.
            self.mean_exp_mtx[node] = mean_exp_vec

            #Assign the mean expression to each node.
            #We use an iterator.
            #Each node stores a dictionary.
            it = zip(self.list_of_markers, mean_exp_vec)
            for marker, mean_exp in it:
                self.G.nodes[node][marker] = mean_exp

        txt = ("Mean expression has been"
               " computed for all markers.")
        print(txt)

        self.tf = clock()
        delta = self.tf - self.t0
        txt = ("Elapsed time to compute node expression: " + 
                f"{delta:.2f} seconds.")


    #=====================================
    def compute_node_expression_metadata(
            self,
    ):
        """
        Compute the minimum,
        maximum, median, and mad after ignoring
        zeros.
        """
        self.t0 = clock()

        n_nodes, n_markers = self.mean_exp_mtx.shape
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

        self.node_mad_dist_df = pd.DataFrame(
            data = np.zeros((15,n_markers * 3)), 
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
        #We iterate over the columns.
        #No zeros.
        for idx, (start, end) in it:
            marker = self.list_of_markers[idx]
            data = mean_exp_mtx_sp.data[start:end]
            median = np.median(data)
            min = np.min(data)
            max = np.max(data)
            mad = median_abs_deviation(data)
            min_mad = (min - median) / mad
            max_mad = (max - median) / mad
            delta =  (max - min) / mad / 15
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
        print(self.node_exp_stats_df.loc["FAP",:])
        L = ["FAP_mad_bounds", "FAP_counts"]
        print(self.node_mad_dist_df[L])

        self.tf = clock()
        delta = self.tf - self.t0
        txt = ("Elapsed time to compute node metadata: " + 
                f"{delta:.2f} seconds.")

    #=====================================
    def count_connected_nodes_above_threshold_for_attribute(
            self,
            threshold: float,
            attribute: str,
    ):
        """
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
            mad_threshold: Optional[float] = 1.,
    ):
        """
        TODO: Identify questionable cells and doublets.
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

        for k, marker in enumerate(self.list_of_markers):
            mad_exp_df[marker]         = ""
            above_threshold_df[marker] = ""
            sorted_marker_df[k+1]      = ""
            sorted_cell_type_df[k+1]   = ""


        # Iterate over cells.
        for row_idx, row in enumerate(tqdm(self.X)):
            # if row_idx == 10:
            #     break
            if sp.issparse(self.X):
                vec = row.toarray().squeeze()
            else:
                vec = row.copy()
            m_exp = vec[self.list_of_column_idx]
            m_exp -= self.node_exp_stats_df.loc[:,"median"]
            m_exp /= self.node_exp_stats_df.loc[:,"mad"]
            above_th = mad_threshold < m_exp

            mad_exp_df.iloc[row_idx] = m_exp
            above_threshold_df.iloc[row_idx] = above_th

            indices = np.argsort(m_exp)
            indices = indices[::-1]

            sorted_markers = self.list_of_markers[indices]
            sorted_marker_df.iloc[row_idx] = sorted_markers

            sorted_cell_types = self.list_of_cell_types[
                indices]
            x = sorted_cell_types
            sorted_cell_type_df.iloc[row_idx] = x

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

        tmc_ct = "TMC_cell_type"
        self.A.obs[tmc_ct] = sorted_cell_type_df[1]
        self.generate_cell_annotation_file(
            tmc_ct, tag="cell_types")

        tmc_marker = "TMC_marker"
        self.A.obs[tmc_marker] = sorted_marker_df[1]
        self.generate_cell_annotation_file(
            tmc_marker, tag="cell_markers")

    #=====================================
    def plot_marker_distributions(
            self,
    ):
        """
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

        print(f"Diameter for branch at {node}: {Q_total}")
        print(f"Maximum attained between nodes: "
              f"{L1[0]} and {L2[0]}")
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

    #=====================================
    def annotate_with_celltypist(
            self,
            cell_ann_col: Optional[str] = "cell_annotations",
            model_kind: Optional[str] = "Immune_All_High",
            use_majority_voting: Optional[bool] = False,
    ):
        """
        """
        B = sc.AnnData(self.A.X.copy())
        # if sp.issparse(B.X):
        #     B.X = sp.csr_array(B)

        # vec = B.X.exp1m().sum(axis=1) - 1e4
        # vec = np.abs(vec)

        # if 0.01 < np.max(vec):
        sc.pp.normalize_total(B, target_sum=10000)
        sc.pp.log1p(B)
            # print("Matrix is ready to be used.")
            
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



            

    #====END=OF=CLASS=====================

