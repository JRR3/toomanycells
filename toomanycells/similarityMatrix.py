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
from time import perf_counter as clock

from numpy import ix_ as np_ix
from numpy import sum as np_sum
from numpy import inf as np_inf
from numpy import abs as np_abs
from numpy import save as np_save
from numpy import sqrt as np_sqrt
from numpy import ones as np_ones
from numpy import diag as np_diag
from numpy import array as np_array
from numpy import zeros as np_zeros
from numpy import argmax as np_argmax
from numpy.random import rand as np_rand
from numpy import fill_diagonal as np_fill_diagonal

from numpy.linalg import norm as np_norm
from scipy.linalg import norm as sp_norm

from typing import List
from typing import Tuple
from typing import Optional
from numpy.typing import ArrayLike

from scipy.sparse import issparse
from scipy.sparse import diags as sp_diags

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels


from sklearn.metrics.pairwise import euclidean_distances


from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigs as EigSpGen
from scipy.sparse.linalg import eigsh as EigSpHer


class SimilarityMatrix:

    #=====================================
    def __init__(
            self,
            matrix: ArrayLike,
            use_hermitian_method: bool = False,
            svd_algorithm: str = "arpack",
            output: str = "",
            modularity_default: float = 0,
            verbose_mode: bool = False,
            float_data_type: float = float,
    ):
        self.X: ArrayLike = matrix
        self.is_sparse = issparse(self.X)
        self.similarity_norm: float
        self.trunc_SVD = None
        self.use_hermitian_method = use_hermitian_method
        self.svd_algorithm = svd_algorithm
        self.modularity_default = modularity_default

        list_of_svd_algorithms = ["randomized", "arpack"]
        if svd_algorithm not in list_of_svd_algorithms:
            raise ValueError("Unexpected SVD algorithm.")

        self.verbose_mode = verbose_mode
        self.eps = 1e-9
        self.FDT = float_data_type
        self.output = output
        self.add_eps_to_zero = False
        self.threshold_to_full = 200
        self.compute_partition = None
        
    #=====================================
    def compute_vector_of_norms(
            self,
            lp_norm:float = 2,
        ) -> ArrayLike:
        """
        """
        if issparse(self.X):

            from scipy.linalg import norm as sp_norm

            vec = sp_norm(
                self.X,
                ord=lp_norm,
                axis=1,
            )
        else:

            from numpy.linalg import norm as np_norm

            vec = np_norm(
                self.X,
                ord=lp_norm,
                axis=1,
            )

        return vec


    #=====================================
    def compute_similarity_matrix(
            self,
            shift_similarity_matrix: float = 0,
            shift_until_nonnegative: bool = False,
            store_similarity_matrix: bool = False,
            normalize_rows: bool = False,
            similarity_function: str = "cosine_sparse",
            similarity_norm: float = 2,
            similarity_gamma: Optional[float] = None,
            use_tf_idf: bool = False,
            tf_idf_norm: Optional[str] = None,
            tf_idf_smooth: bool = False,
            plot_similarity_matrix: bool = False,
            use_exact_diameter: bool = False,
            use_adaptive_diameter: bool = True,
    ):

        if similarity_norm < 1:
            raise ValueError("Unexpected similarity norm.")
        self.similarity_norm = similarity_norm

        if similarity_gamma is None:
            # gamma = 1 / (number of features)
            similarity_gamma = 1 / self.X.shape[1]
        elif similarity_gamma <= 0:
            raise ValueError("Unexpected similarity gamma.")

        similarity_functions = []
        similarity_functions.append("cosine_sparse")
        similarity_functions.append("dnes_sparse")
        similarity_functions.append("cosine")
        similarity_functions.append("laplacian")
        similarity_functions.append("gaussian")
        similarity_functions.append("div_by_sum")
        similarity_functions.append("div_by_delta_max")

        if similarity_function not in similarity_functions:
            raise ValueError("Unexpected similarity fun.")

        t0_build_sim_matrix = clock()
        print("building similarity matrix ...")

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

            from sklearn.feature_extraction.text import TfidfTransformer

            tf_idf_obj = TfidfTransformer(
                norm=tf_idf_norm,
                smooth_idf=tf_idf_smooth,
            )

            self.X = tf_idf_obj.fit_transform(self.X)
            if self.is_sparse:
                pass
            else:
                #If the matrix was originally dense
                #and the tf_idf function changed it
                #to sparse, then convert to dense.
                if issparse(self.X):
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

            fun_name = "compute_partition_for_cosine_sparse"
            self.compute_partition = getattr(self, fun_name)

            from sklearn.decomposition import TruncatedSVD

            self.trunc_SVD = TruncatedSVD(
                n_components=2,
                n_iter=5,
                algorithm=self.svd_algorithm
            )

        elif similarity_function == "dnes_sparse":

            fun_name = "compute_partition_for_dnes_sparse"
            self.compute_partition = getattr(self, fun_name)

            print(fun_name)

            vec = self.compute_vector_of_norms()
            self.norm_sq_vec = vec * vec

            print("Vector of norms has been computed.")

            self.use_exact_diameter = use_exact_diameter
            self.use_adaptive_diameter = use_adaptive_diameter

            #The diameter is computed within the 
            #definition of the LinearOperator.

            x = self.compute_sq_diameter_for_observations()
            #print(f"The sq_diameter is approx.: {x}")
            self.inv_diam_sq = 1 / x

        else:
            #Use a similarity function different from
            #the cosine_sparse similarity function.

            n_rows = self.X.shape[0]

            import os.cpu_count as cpu_count
            max_workers = int(cpu_count())

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

        if similarity_function == "cosine":

            #( x @ y ) / ( ||x|| * ||y|| )
            # This function is not translation invariant.
            #We set the # of jobs to 1 since the cosine
            #similarity uses the BLAS library,
            #which is already multithreaded.
            n_workers = 1

            fun_name = "compute_partition_for_dense_matrix"
            self.compute_partition = getattr(self, fun_name)

            self.X = pairwise_kernels(self.X,
                                        metric="cosine",
                                        n_jobs=n_workers)

        elif similarity_function == "laplacian":

            #exp(-||x-y||_1 * gamma)
            # This function is translation invariant.
            # The Laplacian kernel only offers the l1 norm.

            fun_name = "compute_partition_for_dense_matrix"
            self.compute_partition = getattr(self, fun_name)

            self.X = pairwise_kernels(
                self.X,
                metric="laplacian",
                n_jobs=n_workers,
                gamma = similarity_gamma)

        elif similarity_function == "gaussian":
            #exp(-||x-y||_2^2 * gamma)
            # This function is translation invariant.
            # The Gaussian kernel only offers the l2 norm.

            fun_name = "compute_partition_for_dense_matrix"
            self.compute_partition = getattr(self, fun_name)

            self.X = pairwise_kernels(
                self.X,
                metric="rbf",
                n_jobs=n_workers,
                gamma = similarity_gamma)

        elif similarity_function == "div_by_sum":
            # D(x,y) = 1 - ||x-y|| / (||x|| + ||y||)
            # This function is not translation invariant.

            # If the vectors have unit norm, then
            # D(x,y) = 1 - ||x-y|| / 2

            # If the user chooses this function, then
            # the row vectors are automatically normalized.

            fun_name = "compute_partition_for_dense_matrix"
            self.compute_partition = getattr(self, fun_name)

            if self.similarity_norm == 1:
                lp_norm = "l1"
            elif self.similarity_norm == 2:
                lp_norm = "euclidean"
                #We set the # of jobs to 1 since the 
                #euclidean similarity uses the BLAS library,
                #which is already multithreaded.
                n_workers = 1
            else:
                txt = "Similarity norm should be 1 or 2."
                raise ValueError(txt)

            self.X = pairwise_distances(self.X,
                                        metric=lp_norm,
                                        n_jobs=n_workers)
            self.X *= -0.5
            self.X += 1

        elif similarity_function == "div_by_delta_max":
            # Let Phi be the diameter of the set S.
            # Phi = max_{x,y in S} {||x-y||}
            # D(x,y) = 1 - ( ||x-y|| / Phi )^2
            # This function is translation invariant.
            # Note that D(x,y) is zero
            # when ||x-y|| equals M and is 
            # equal to 1 only when x = y.

            fun_name = "compute_partition_for_dense_matrix"
            self.compute_partition = getattr(self, fun_name)

            if self.similarity_norm == 1:
                lp_norm = "l1"
            elif self.similarity_norm == 2:
                lp_norm = "euclidean"
                #We set the # of jobs to 1 since the 
                #euclidean similarity uses the BLAS library,
                #which is already multithreaded.
                n_workers = 1
            else:
                txt = "Similarity norm should be 1 or 2."
                raise ValueError(txt)

            self.X = pairwise_distances(self.X,
                                        metric=lp_norm,
                                        n_jobs=n_workers)

            diam = self.X.max()
            print(f"The diameter is: {diam}")
            self.X *= self.X
            self.X *= -1 / (diam * diam)
            self.X += 1


        if not issparse(self.X):

            if shift_until_nonnegative:
                min_value = self.X.min()
                if min_value < 0:
                    shift_similarity_matrix = -min_value
                    txt="Similarity matrix will be shifted."
                    print(txt)
                    txt=f"Shift: {shift_similarity_matrix}."
                    print(txt)
                    self.X += shift_similarity_matrix

            elif shift_similarity_matrix != 0:
                print(f"Similarity matrix will be shifted.")
                print(f"Shift: {shift_similarity_matrix}.")
                self.X += shift_similarity_matrix

            if store_similarity_matrix:

                matrix_fname = "similarity_matrix.npy"

                from os.path import join as os_path_join
                matrix_fname = os_path_join(
                    self.output, matrix_fname)
                np_save(matrix_fname, self.X)

            if plot_similarity_matrix:
                self.plot_similarity_matrix()

        print("Similarity matrix has been built.")
        tf_build_sim_matrix = clock()
        delta = tf_build_sim_matrix - t0_build_sim_matrix
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
            row_norm  = np_norm(
                data, ord=self.similarity_norm)

            if row_norm < self.eps:
                continue

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

            row_norm = np_norm(
                    row, ord=self.similarity_norm)

            if row_norm < self.eps:
                continue

            row /= row_norm

    #=====================================
    def compute_partition_for_cosine_sparse(
            self,
            rows: ArrayLike,
        )->Tuple[float, List[ArrayLike]]:
        """
        This function is for sparse matrices.

        Compute the partition of the given set
        of cells. The rows argument
        contains the indices of the
        rows that we are going to partition.

        The algorithm computes a truncated
        SVD and the corresponding modularity
        of the newly created communities.
        """

        if self.verbose_mode:
            print(f'I was given: {rows=}')


        n_rows = len(rows) 
        #print(f"Number of cells: {n_rows}")

        if n_rows == 1:
            Q = -np_inf
            partition = []
            return (Q, partition)

        B = self.X[rows,:]
        ones = np_ones(n_rows, dtype=self.FDT)

        # partial_row_sums = B.T.dot(ones)
        #1^T @ B @ B^T @ 1 = (B^T @ 1)^T @ (B^T @ 1)
        # L_all = partial_row_sums @ partial_row_sums - n_rows
        #These are the row sums of the similarity matrix
        # row_sums = B @ partial_row_sums

        row_sums = B @ (B.T @ ones)

        #We will use the similarity operator for the
        #modularity computation.
        similarity_op = None

        #Check if we have negative entries before computing
        #the square root.
        # if  neg_row_sums or self.use_hermitian_method:
        zero_row_sums_mask = np_abs(row_sums) < self.eps
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

        n_cols     = B.shape[1]
        condition  = n_rows == 2
        condition |= n_cols == 2
        condition |= has_neg_row_sums
        condition |= has_zero_row_sums
        condition |= self.use_hermitian_method

        if condition:

            T=self.generate_cosine_operators(rows)
            similarity_op = T[0]
            row_sums_op   = T[1]
            laplacian_op  = T[2]

        if  n_rows == 2:

            #The linear operators are full matrices.

            if similarity_op[0,1] <= 0:
                #Since the similarity is nonpositive
                #we separate the cells.
                #Note that for nonnegative data, this
                #condition will be satisfied only when
                #the vectors are orthogonal.

                partition = [np_array([rows[0]]),
                             np_array([rows[1]])]
                #By assigning each cell to a separate
                #community with a single edge between
                #them, we get Q = -1/2
                Q = -1/2
            else:
                #Group the cells into a single set.
                Q         = -np_inf
                partition = []

            return (Q, partition)

        elif has_neg_row_sums or condition:

            masks = self.compute_masks_from_operators(
                row_sums_op,
                laplacian_op,
            )

        else:

            masks = self.compute_masks_from_matrix(
                B,
                row_sums,
            )

        if similarity_op is None:

            # We create the operator
            #S_op @ vec = B @ (B.T @ vec)
            similarity_op = self.generate_cosine_operators(
                rows,
                only_similarity=True,
            )

        return self.compute_modularity_and_partition(
            rows,
            masks,
            similarity_op,
            row_sums,
        )

    #=====================================
    def generate_cosine_operators(
            self,
            rows: ArrayLike,
            only_similarity: bool = False,
        )->Tuple[LinearOperator]:
        """
        Generate a linear operator that describes
        the matrix:
        S(x,y) = <x,y> / sqrt( <x,x> * <y,y> )
        """
        B               = self.X[rows,:]
        n_rows          = len(rows)
        float_data_type = self.FDT
        ones            = np_ones(n_rows,
                                  dtype=float_data_type)

        if n_rows < self.threshold_to_full:
            #We use the full matrix.
            S_op = B @ B.T

            if issparse(S_op):
                S_op = S_op.toarray()
            
            #In-place modification
            np_fill_diagonal(S_op, 1)

            if only_similarity:
                return S_op

            row_sums = S_op @ ones

            D_op = np_diag(row_sums)

            L_op = D_op - S_op

            return (S_op, D_op, L_op)

        #-------------------------------------------
        #Similarity operator

        def mat_vec_prod_for_sim_op(vec: ArrayLike):
            return B @ (B.T @ vec)

        S_op = LinearOperator(
            dtype=float_data_type,
            shape=(n_rows,n_rows),
            matvec=mat_vec_prod_for_sim_op,
            rmatvec=mat_vec_prod_for_sim_op,
        )
        #-------------------------------------------

        if only_similarity:
            return S_op

        #-------------------------------------------
        # Diagonal operator
        row_sums = S_op @ ones

        def mat_vec_prod_for_diag_op(vec: ArrayLike):
            return row_sums * vec

        D_op = LinearOperator(
            dtype=float_data_type,
            shape=(n_rows,n_rows),
            matvec=mat_vec_prod_for_diag_op,
            rmatvec=mat_vec_prod_for_diag_op,
        )
        #-------------------------------------------

        #-------------------------------------------
        # Laplace operator

        def mat_vec_prod_for_lap_op(vec: ArrayLike):
            return row_sums * vec - B @ (B.T @ vec)

        L_op =  LinearOperator(
            dtype=float_data_type,
            shape=(n_rows,n_rows),
            matvec=mat_vec_prod_for_lap_op,
            rmatvec=mat_vec_prod_for_lap_op,
        )
        #-------------------------------------------

        return (S_op, D_op, L_op)


    #=====================================
    def generate_dnes_operators(
            self,
            rows: ArrayLike,
            only_similarity: bool = False,
        )->Tuple[LinearOperator]:
        """
        Generate a linear operator that describes
        the matrix:
        1 - ||x-y||^2 / ||x-y||^2_{max}
        """
        # Expose objects to be used so that the 
        # closure does not take unnecessary material.
        B               = self.X[rows,:]
        norms_sq        = self.norm_sq_vec[rows]
        n_rows          = len(rows)
        float_data_type = self.FDT
        ones            = np_ones(
            n_rows, dtype=float_data_type)

        #If the number of rows is less than 200,
        #we use a direct method.
        if n_rows < self.threshold_to_full:
            #We use the full matrix.

            # raise ValueError("XXX")
            S_op = euclidean_distances(B, squared=True)

            diam_sq = S_op.max()

            if n_rows == 2:
                # This is used to enforce the
                # partition of the two cells.
                diam_sq *= 1.1

            if self.use_adaptive_diameter:
                scaling = 1 / diam_sq
            else:
                scaling = self.inv_diam_sq

            S_op *= -scaling
            S_op += 1

            if only_similarity:
                return S_op

            row_sums = S_op @ ones

            D_op = np_diag(row_sums)

            L_op = D_op - S_op

            return (S_op, D_op, L_op)


        scaling = 0
        if self.use_adaptive_diameter:
            x = self.compute_sq_diameter_for_observations(B)
            #print(f"Calculated {diam=:.2e}")
            scaling = 1 / x
        else:
            scaling = self.inv_diam_sq

        #-------------------------------------------
        # Similarity operator

        def mat_vec_prod_for_sim_op(vec: ArrayLike):
            norm_sq = norms_sq * (ones @ vec)
            norm_sq += ones * (norms_sq @ vec)
            norm_sq -= 2 * B @ (B.T @ vec)
            norm_sq *= scaling
            return ones * (ones @ vec) - norm_sq

        S_op = LinearOperator(
            dtype=float_data_type,
            shape=(n_rows,n_rows),
            matvec=mat_vec_prod_for_sim_op,
            rmatvec=mat_vec_prod_for_sim_op,
        )
        #-------------------------------------------

        if only_similarity:
            return S_op

        #-------------------------------------------
        # Diagonal operator

        row_sums = S_op @ ones
        def mat_vec_prod_for_diag_op(vec: ArrayLike):
            return row_sums * vec

        D_op = LinearOperator(
            dtype=float_data_type,
            shape=(n_rows,n_rows),
            matvec=mat_vec_prod_for_diag_op,
            rmatvec=mat_vec_prod_for_diag_op,
        )
        #-------------------------------------------


        #-------------------------------------------
        # Laplace operator
        def mat_vec_prod_for_lap_op(vec: ArrayLike):
            norm_sq = norms_sq * (ones @ vec)
            norm_sq += ones * (norms_sq @ vec)
            norm_sq -= 2 * B @ (B.T @ vec)
            norm_sq *= scaling
            x = ones * (ones @ vec) - norm_sq
            return row_sums * vec - x

        L_op =  LinearOperator(
            dtype=float_data_type,
            shape=(n_rows,n_rows),
            matvec=mat_vec_prod_for_lap_op,
            rmatvec=mat_vec_prod_for_lap_op,
        )
        #-------------------------------------------

        return (S_op, D_op, L_op)

    #=====================================
    def compute_masks_from_operators(
            self,
            row_sums_op: LinearOperator,
            laplacian_op: LinearOperator,
            max_resolution: bool = False,
        )->Tuple[List[bool]]:
        """
        If we set max_resolution to True, we
        will separate pairs of cells as long
        as their similarity is not equal to 1.
        """
        #Do we have only 2 cells?
        if laplacian_op.shape[0] == 2:

            if max_resolution:
                vec = np_array([0,1])
                second_column = laplacian_op @ vec
                top_right_entry = second_column[0]
                if -1 + self.eps < top_right_entry:
                    #We split the cells.
                    W = np_array([1,-1])
                else:
                    #We DO NOT split the cells.
                    W = np_array([1,1])
            else:
                W = np_array([1,1])

            return self.compute_masks_from_eigenvector(W)

        try:
            E_obj = EigSpHer(
                laplacian_op,
                k=2,
                M=row_sums_op,
                maxiter=200,
                which="SM",
            )

            eigen_val_abs = np_abs(E_obj[0])
            #Identify the eigenvalue with the
            #largest magnitude.
            idx = np_argmax(eigen_val_abs)
            #Choose the eigenvector corresponding
            # to the eigenvalue with the 
            # largest magnitude.
            eigen_vectors = E_obj[1]
            W = eigen_vectors[:,idx]

        except:

            print("Warning! Hermitian solver failed.")
            print("Warning! Using General Solver.")

            print(f"{laplacian_op.shape}")
            print(f"{row_sums_op.shape}")

            print(f"{laplacian_op}")
            print(f"{row_sums_op}")

            E_obj = EigSpGen(
                laplacian_op,
                k=2,
                M=row_sums_op,
                which="SM",
            )

            eigen_val_abs = np_abs(E_obj[0])
            #Identify the eigenvalue with the
            #largest magnitude.
            idx = np_argmax(eigen_val_abs)
            #Choose the eigenvector corresponding
            # to the eigenvalue with the 
            # largest magnitude.
            eigen_vectors = E_obj[1]
            W = eigen_vectors[:,idx]

        return self.compute_masks_from_eigenvector(W)


    #=====================================
    def compute_masks_from_eigenvector(
            self,
            W: ArrayLike,
        )->Tuple[List[bool]]:

        #Convert zeros into positive values.
        if self.add_eps_to_zero:
            mask = W <=0
            if mask.all():
                print("Let W be the partition vector.")
                print("All elements of W are nonpositive.")
                print("W += epsilon.")
                W += self.eps

        mask_c1 = self.eps / 2 < W
        mask_c2 = ~mask_c1

        return (mask_c1, mask_c2)

    #=====================================
    def compute_masks_from_matrix(
            self,
            B: ArrayLike,
            row_sums: ArrayLike,
        )->Tuple[List[bool]]:
        """
        This function is specific for the 
        cosine similarity.
        """

        #This is the fast approach.
        #It is fast in the sense that the 
        #operations are faster if the matrix
        #is sparse, i.e., O(n) nonzero entries.
        d = 1 / np_sqrt(row_sums)
        D = sp_diags(d)
        C = D @ B
        W = self.trunc_SVD.fit_transform(C)
        singular_values=self.trunc_SVD.singular_values_

        # Singular values are ordered from
        # largest to smallest.
        # We want the second largest singular value.
        idx = 1
        sigma = singular_values[idx]

        # order = np_argsort(singular_values)
        # idx = order[0]

        if sigma < self.eps:
            # This could be a problem.
            n_rows = B.shape[0]

            print(f"Warning: sigma < eps")
            print(f"{sigma=}")
            print(f"{n_rows=}")
            print(f"{row_sums=}")
            print(f"{singular_values=}")

        #Get the singular vector corresponding to the
        #second largest singular value.
        W = W[:,idx]

        return self.compute_masks_from_eigenvector(W)

    #=====================================
    def compute_modularity_and_partition(
            self,
            rows: ArrayLike,
            masks: Tuple[List[bool]],
            similarity_op: LinearOperator,
            row_sums: Optional[ArrayLike] = None,
        )->Tuple[float, List[ArrayLike]]:

        partition = []
        Q         = self.modularity_default

        #If one partition has all the elements
        #then return with Q = 0.
        if masks[0].all() or masks[1].all():
            Q = -np_inf
            return (Q, partition)

        n_rows = similarity_op.shape[0]
        float_data_type = self.FDT
        ones = np_ones(n_rows, dtype=float_data_type)

        if row_sums is None:
            row_sums = similarity_op @ ones

        L_all = np_sum(row_sums) - n_rows
        # print(f"L_all inside multiomic:{L_all}")

        if np_abs(L_all) < self.eps:

            print(">>> Warning: L_all < epsilon.")
            print(">>> L_all += epsilon.")
            L_all += self.eps

        for mask in masks:

            n_rows_msk = mask.sum()
            partition.append(rows[mask])
            ones_msk = ones * mask
            row_sums_msk = similarity_op @ ones_msk
            O_c = ones_msk @ row_sums_msk - n_rows_msk
            L_c = ones_msk @ row_sums - n_rows_msk
            # Modularity
            Q += O_c / L_all - (L_c / L_all)**2

        if self.verbose_mode:
            print(f'{Q=}')
            print(f'I found: {partition=}')
            print('===========================')

        return (Q, partition)


    #=====================================
    def compute_partition_for_dense_matrix(self, 
                                  rows: ArrayLike,
        )->Tuple[float, List[ArrayLike]]:
        """
        Compute the partition of the given set
        of cells. The rows input
        contains the indices of the
        rows we are to partition.
        We assume that the matrix is full.
        For sparse matrices consider 
        a sparse generator.
        """

        if self.verbose_mode:
            print(f"I was given: {rows=}")

        n_rows = len(rows) 
        #print(f"Number of cells: {n_rows}")

        #If the number of rows is one,
        #we keep the cluster as it is.
        if n_rows == 1:
            Q = -np_inf
            partition = []
            return (Q, partition)

        #Note: self.X is a cell by cell similarity
        #matrix. Hence, we only need the block that
        #corresponds to the cells given by the rows.

        similarity_op = self.X[np_ix(rows, rows)]
        ones          = np_ones(n_rows, dtype=self.FDT)
        row_sums      = similarity_op @ ones 
        row_sums_op   = np_diag(row_sums)
        laplacian_op  = row_sums_op - similarity_op

        masks = self.compute_masks_from_operators(
            row_sums_op,
            laplacian_op,
        )

        return self.compute_modularity_and_partition(
            rows,
            masks,
            similarity_op,
            row_sums,
        )


    #=====================================
    def compute_sq_diameter_for_observations(
            self,
            matrix: Optional[ArrayLike] = None,
        )->float:
        """
        Assuming every row vector is a 
        point in R^n, we estimate the diameter 
        of that set.
        """

        if matrix is None:
            matrix = self.X

        if self.use_exact_diameter:
            distance_matrix = euclidean_distances(
                matrix,
                squared=True,
            )
            return distance_matrix.max()

        # Else, we use the approximation.
        centroid  = matrix.mean(axis=0)

        # fun = lambda x: np_norm(x-centroid, ord=2)
        # fun = lambda x: (x-centroid)**2

        #We use this function to avoid computing a sqrt.
        def sq_euc_dist(x: ArrayLike)->float:
            y = x - centroid
            return y @ y

        max_norm_sq = max(map(sq_euc_dist, matrix))

        diameter_sq =  4 * max_norm_sq

        # print(f"Approx. sq_diam: {diameter_sq}")

        return diameter_sq

    #=====================================
    def compute_partition_for_dnes_sparse(
            self,
            rows: ArrayLike,
        )->Tuple[float, List[ArrayLike]]:
        """
        Compute the partition of the given set
        of cells. The rows input
        contains the indices of the
        rows we are to partition.
        We assume that the matrix is sparse.
        For dense matrices consider 
        a dense generator.
        """

        if self.verbose_mode:
            print(f"I was given: {rows=}")


        n_rows = len(rows) 
        #print(f"Number of cells: {n_rows}")

        #If the number of rows is one
        #we keep the cluster as it is.
        if n_rows == 1:
            print("We have just one row!")
            Q = -np_inf
            partition = []
            return (Q, partition, None)

        T = self.generate_dnes_operators(rows)
        similarity_op = T[0]
        row_sums_op   = T[1]
        laplacian_op  = T[2]

        ones = np_ones(n_rows, dtype=self.FDT)
        row_sums = similarity_op @ ones

        zero_row_sums_mask = np_abs(row_sums) < self.eps
        has_zero_row_sums = zero_row_sums_mask.any()
        has_neg_row_sums = (row_sums < -self.eps).any() 

        if has_neg_row_sums:
            print("The similarity matrix "
                  "has negative row sums")
            Q = -np_inf
            print(f"{n_rows=}")
            print(f"{row_sums.min()=}")
            raise ValueError("Negative row sums for dnes.")

        if has_zero_row_sums:
            print("Warning: We have zero row sums.")
            row_sums[zero_row_sums_mask] = 0

        masks = self.compute_masks_from_operators(
            row_sums_op,
            laplacian_op,
        )

        return self.compute_modularity_and_partition(
            rows,
            masks,
            similarity_op,
            row_sums,
        )


    #=====================================
    def plot_similarity_matrix(self):

        #Matplotlib parameters.
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        mpl.use("agg")
        mpl.rcParams["figure.dpi"]=600
        # mpl.rcParams["pdf.fonttype"]=42
        mpl.rc("pdf", fonttype=42)
        font = {'weight' : 'normal', 'size'   : 18}
        mpl.rc("font", **font)

        fig, ax = plt.subplots()
        cax = ax.matshow(self.X, cmap="plasma")
        ax.axis("off")
        fig.colorbar(cax)
        #ax.colorbar()
        fname = "matrix.png"

        from os.path import join as os_path_join
        fname = os_path_join(self.output, fname)

        fig.savefig(fname, bbox_inches="tight")
        print("Matrix image has been produced.")


            




