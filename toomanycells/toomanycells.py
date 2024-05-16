#########################################################
#Princess Margaret Cancer Research Tower
#Schwartz Lab
#Javier Ruiz Ramirez
#May 2024
#########################################################
#This is a Python implementation of the command line 
#tool too-many-cells.
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7439807/
#########################################################
#Questions? Email me at: javier.ruizramirez@uhn.ca
#########################################################
from typing import Optional
from typing import Union
import networkx as nx
from scipy import sparse as sp
from scipy.sparse.linalg import eigsh as Eigen_Hermitian
from scipy.io import mmread
from time import perf_counter as clock
import anndata as ad
from anndata import AnnData
import numpy as np
import pandas as pd
import re
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
from collections import deque
import os
from os.path import dirname
import subprocess
from tqdm import tqdm
import sys

#sys.path.insert(0, dirname(__file__))
from .common import MultiIndexList

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
            input: Union[AnnData, str],
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

        if isinstance(input, str):
            self.source = os.path.abspath(input)
            if self.source.endswith('.h5ad'):
                self.t0 = clock()
                self.A = ad.read_h5ad(self.source)
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
                            self.A = ad.read_h5ad(fname)
                            self.tf = clock()
                            delta = self.tf - self.t0
                            txt = ('Elapsed time for ' +
                                   'loading: ' +
                                    f'{delta:.2f} seconds.')
                            print(txt)
                            break

        elif isinstance(input, AnnData):
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
        n_cols = len(self.A.obs.columns)
        self.A.obs['sp_cluster'] = -1
        self.A.obs['sp_path']    = ""

        t = self.A.obs.columns.get_loc("sp_cluster")
        self.cluster_column_index = t
        t = self.A.obs.columns.get_loc("sp_path")
        self.path_column_index = t

        self.delta_clustering = 0
        self.final_n_iter     = 0

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
                txt = ("Values will be converted to" 
                       " float32.")
                print(txt)
                self.X = self.X.astype(np.float32)
            else:
                self.is_sparse = True
                #Make sure we use a CSR format.
                self.X = sp.csr_matrix(self.A.X,
                                       dtype=np.float32,
                                       copy=True)
        else:
            #The matrix is dense.
            print("The matrix is dense.")
            self.is_sparse = False
            self.X = self.A.X.copy()
            txt = ("Values will be converted to" 
                   " float32.")
            print(txt)
            self.X = self.X.astype(np.float32)

        self.n_cells, self.n_genes = self.A.shape

        if self.n_cells < 3:
            raise ValueError("Too few observations (cells).")

        print(self.A)

        #We use a deque to enforce a breadth-first traversal.
        self.Dq = deque()

        #We use a directed graph to enforce the parent
        #to child relation.
        self.G = nx.DiGraph()

        self.set_of_leaf_nodes = set()

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

        print('Normalizing rows.')


        #It's just an alias.
        mat = self.X

        for i in range(self.n_cells):
            row = mat.getrow(i)
            nz = row.data
            row_norm  = np.linalg.norm(
                nz, ord=self.similarity_norm)
            row = nz / row_norm
            mat.data[mat.indptr[i]:mat.indptr[i+1]] = row

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

        node_id = self.node_counter

        #Initialize the array of cells to partition
        rows = np.array(range(self.X.shape[0]))

        #Initialize the deque
        self.Dq.append((rows, node_id))

        #Initialize the graph
        self.G.add_node(node_id, size=len(rows))

        #Path to reach root node.
        self.node_to_path[node_id] = str(node_id)

        #Indices to reach root node.
        self.node_to_j_index[node_id] = None

        #Update the node counter
        self.node_counter += 1

        max_n_iter = self.estimate_n_of_iterations()

        self.print_message_before_clustering()

        with tqdm(total=max_n_iter) as pbar:
            while 0 < len(self.Dq):
                rows, node_id = self.Dq.popleft()
                if similarity_function == "cosine_sparse":
                    Q,S = self.compute_partition_for_sp(rows)
                else:
                    Q,S = self.compute_partition_for_gen(rows)
                current_path = self.node_to_path[node_id]
                j_index = self.node_to_j_index[node_id]
                if self.eps < Q:

                    D = self.modularity_to_json(Q)
                    if j_index is None:
                        self.J.append(D)
                        self.J.append([[],[]])
                        j_index = (1,)
                    else:
                        self.J[j_index].append(D)
                        self.J[j_index].append([[],[]])
                        j_index += (1,)

                    self.G.nodes[node_id]['Q'] = Q

                    for k,indices in enumerate(S):
                        new_node = self.node_counter
                        self.G.add_node(new_node,
                                size=len(indices))
                        self.G.add_edge(node_id, new_node)
                        T = (indices, new_node)
                        self.Dq.append(T)

                        #Update path for the new node
                        new_path = current_path 
                        new_path += '/' + str(new_node) 
                        self.node_to_path[new_node]=new_path

                        seq = j_index + (k,)
                        self.node_to_j_index[new_node] = seq

                        self.node_counter += 1

                else:
                    if j_index is None:
                        txt = ("All cells belong" 
                               " to the same partition.")
                        print(txt)
                        continue
                    #Update the relation between a set of
                    #cells and the corresponding leaf node.
                    #Also include the path to reach that node.
                    c = self.cluster_column_index
                    self.A.obs.iloc[rows, c] = node_id

                    reversed_path = self.reverse_path(
                        current_path)
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
    def store_outputs(self,
                   load_dot_file: Optional[bool]=False):
        """
        Plot the branching tree. If the .dot file already\
            exists, one can specify such condition with \
            the flag `load_dot_file=True`. This function \
            also generates two CSV files. One is the \
            clusters_hm.csv file, which stores the \
            relation between cell ids and the cluster they \
            belong. The second is node_info_hm.csv, which \
            provides information regarding the number of \
            cells belonging to that node and its \
            modularity if it has children. Lastly, a file \
            named cluster_tree_hm.json is produced, which \
            stores the tree structure in the JSON format. \
            This last file can be used with too-many-cells \
            interactive.
        """

        self.t0 = clock()


        fname = 'graph.dot'
        dot_fname = os.path.join(self.output, fname)

        if load_dot_file:
            self.G = nx.nx_agraph.read_dot(dot_fname)
            self.G = nx.DiGraph(self.G)
            self.G = nx.convert_node_labels_to_integers(
                    self.G)
        else:
            nx.nx_agraph.write_dot(self.G, dot_fname)
            #Write cell to node data frame.
            self.write_cell_assignment_to_csv()
            self.convert_graph_to_json()

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
        fname = 'node_info_hm.csv'
        fname = os.path.join(self.output, fname)
        df.to_csv(fname, index=False)

        if self.use_twopi_cmd:

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

            self.tf = clock()
            delta = self.tf - self.t0
            txt = ('Elapsed time for plotting: ' +
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

        self.A = AnnData(self.A)
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
        fname = 'clusters_hm.csv'
        fname = os.path.join(self.output, fname)
        labels = ['sp_cluster','sp_path']
        df = self.A.obs[labels]
        df.index.names = ['cell']
        df = df.rename(columns={'sp_cluster':'cluster',
                                'sp_path':'path'})
        df.to_csv(fname, index=True)

    #=====================================
    def convert_graph_to_json(self):
        """
        The graph structure stored in the attribute\
            self.J has to be formatted into a \
            JSON file. This function takes care\
            of that task. The output file is \
            named 'cluster_tree_hm.json' and is\
            equivalent to the 'cluster_tree.json'\
            file produced by too-many-cells.
        """
        fname = "cluster_tree_hm.json"
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
            column: str) -> None:
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
        fname = 'cell_annotation_labels.csv'
        #ca = cell_annotations
        ca = self.A.obs[column].copy()
        ca.index.names = ['item']
        ca = ca.rename('label')
        fname = os.path.join(self.output, fname)
        self.cell_annotations_path = fname
        ca.to_csv(fname, index=True)

    #=====================================
    def visualize_with_tmc_interactive(self,
            path_to_tmc_interactive: str,
            use_column_for_labels: str = "",
            port: Optional[int] = 9991) -> None:
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

        fname = "cluster_tree_hm.json"
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

        command = [
                bash_exec,
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
        print("The app will start loading after pressing Enter.")
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


    #====END=OF=CLASS=====================

