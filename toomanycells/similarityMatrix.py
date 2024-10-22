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
import matplotlib as mpl
from typing import Optional
from scipy import sparse as sp
from time import perf_counter as clock
from sklearn.feature_extraction.text import TfidfTransformer

class SimilarityMatrix:

    #=====================================
    def __init__(
            self,
            matrix: np.ndarray,
    ):
        self.X = matrix
        self.is_sparse = sp.issparse(self.X)

    #=====================================
    def compute_similarity_matrix(
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
    ):

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
        similarity_functions.append("div_by_delta_max")
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
            max_workers = int(os.cpu_count())
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
            # D(x,y) = 1 - ||x-y|| / (||x|| + ||y||)

            # If the vectors have unit norm, then
            # D(x,y) = 1 - ||x-y|| / (||x|| + ||y||)

            # The rows should have been previously normalized.

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

        elif similarity_function == "div_by_delta_max":
            # Let M be the diameter of the set S.
            # M = max_{x,y in S} {||x-y||}
            # D(x,y) = 1 - ||x-y|| / M
            # Note that this quantity is zero
            # when ||x-y|| equals M and is 
            # equal to 1 only when x = y.

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


            if shift_until_nonnegative:
                min_value = self.X.min()
                if min_value < 0:
                    shift_similarity_matrix = -min_value
                    print(f"Similarity matrix will be shifted.")
                    print(f"Shift: {shift_similarity_matrix}.")
                    self.X += shift_similarity_matrix

            elif shift_similarity_matrix != 0:
                print(f"Similarity matrix will be shifted.")
                print(f"Shift: {shift_similarity_matrix}.")
                self.X += shift_similarity_matrix

            if store_similarity_matrix:
                matrix_fname = "similarity_matrix.npy"
                matrix_fname = os.path.join(self.output,
                                            matrix_fname)
                np.save(matrix_fname, self.X)

            
            print("Similarity matrix has been built.")
            tf = clock()
            delta = tf - t0
            delta /= 60
            txt = ("Elapsed time for similarity build: " +
                    f"{delta:.2f} minutes.")
            print(txt)

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

    # #=====================================
    # def modularity_to_json(self, Q:float):
    #     return {'_item': None,
    #             '_significance': None,
    #             '_distance': Q}

    # #=====================================
    # def cell_to_json(self, cell_name, cell_number):
    #     return {'_barcode': {'unCell': cell_name},
    #             '_cellRow': {'unRow': cell_number}}

    # #=====================================
    # def cells_to_json(self,rows):
    #     L = []
    #     for row in rows:
    #         cell_id = self.A.obs.index[row]
    #         D = self.cell_to_json(cell_id, row)
    #         L.append(D)
    #     return {'_item': L,
    #             '_significance': None,
    #             '_distance': None}
