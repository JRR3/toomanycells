# too-many-cells (à la Python)


[![image](https://img.shields.io/pypi/v/toomanycells.svg)](https://pypi.python.org/pypi/toomanycells)

### It's [Scanpy](https://github.com/scverse/scanpy) friendly!

**A python package for spectral clustering based on the powerful suite of tools named [too-many-cells](https://github.com/GregorySchwartz/too-many-cells). In essence, you can use toomanycells to partition a data set in the form of a matrix of integers or floating point numbers into clusters. The rows represent observations and the columns are the features. Initially, toomanycells will partition your data set into two subsets, trying to maximize the differences between the two. Subsequently, it will reapply that same criterion to each subset and will continue bifurcating until the [modularity](https://en.wikipedia.org/wiki/Modularity_(networks)) of the parent becomes negative, implying that the current subset is fairly homogeneous, and consequently suggesting that further partitioning is not warranted. Thus, when the process finishes, you end up with a tree structure of your data set, where the leaves represent the clusters. As mentioned earlier, you can use this tool with any kind of data. However, a common application is to classify cells and therefore you can provide an [AnnData](https://anndata.readthedocs.io/en/latest/) object. You can read about this application in this [Nature Methods paper](https://www.nature.com/articles/s41592-020-0748-5). **


-   Free software: BSD 3-Clause License
-   Documentation: https://JRR3.github.io/toomanycells

## Dependencies

Make sure you have installed the graph visualization library [Graphviz](https://www.graphviz.org). For example, if you want to use conda, then do the following.
```
conda install anaconda::graphviz
```
Or, if you are using Linux, you can do
```
sudo apt install libgraphviz-dev
```

## Installation

Just type
```
pip install toomanycells
```
in your home environment. If you want to install an updated version, then use the following flag.
```
pip install toomanycells -U
```
Make sure you have the latest version. If not,
run the previous command again.

## Quick run
If you want to see a concrete example of how
to use toomanycells, check out the jupyter 
notebook [demo](./toomanycells_demo.ipynb).

## Usage
1. First import the module as follows
   ```
   from toomanycells import TooManyCells as tmc
   ```
2. If you already have an [AnnData](https://anndata.readthedocs.io/en/latest/) object `A` loaded into memory, then you can create a TooManyCells object with
   ```
   tmc_obj = tmc(A)
   ```
   However, if you want the output folder to be a directory that is not the current working directory, then you can specify the path as follows
   ```
   tmc_obj = tmc(A, output_directory)
   ```
3. If instead of providing an AnnData object you want to provide the directory where your data is located, you can use the syntax
   ```
   tmc_obj = tmc(input_directory, output_directory)
   ```
4. If your input directory has a file in the [matrix market format](https://math.nist.gov/MatrixMarket/formats.html), then you have to specify this information by using the following flag
   ```
   tmc_obj = tmc(input_directory, output_directory, input_is_matrix_market=True)
   ```
   Under this scenario, the `input_directory` must contain a `.mtx` file, a `barcodes.tsv` file (the observations), and a `genes.tsv` (the features).
5. Once your data has been loaded successfully, you can start the clustering process with the following command
   ```
   tmc_obj.run_spectral_clustering()
   ```
   In my desktop computer processing a data set with ~90K cells (observations) and ~30K genes (features) took a little less than 6 minutes in 1809 iterations. For a larger data set like the [Tabula Sapiens](https://figshare.com/articles/dataset/Tabula_Sapiens_release_1_0/14267219?file=40067134) with 483,152 cells and 58,870 genes (14.51 GB in zip format) the total time was about 50 minutes in the same computer.
   ![Progress bar example](https://github.com/JRR3/toomanycells/blob/main/tests/tabula_sapiens_time.png)
    
7. At the end of the clustering process the `.obs` data frame of the AnnData object should have two columns named `['sp_cluster', 'sp_path']` which contain the cluster labels and the path from the root node to the leaf node, respectively.
   ```
   tmc_obj.A.obs[['sp_cluster', 'sp_path']]
   ```
8. To generate the outputs, just call the function
   ```
   tmc_obj.store_outputs()
   ```
   This call will generate a PDF of the tree and a DOT file for the graph, two CSV files that describe the clusters and the information of each node, and a JSON file that contains the tree structure. If you already have a DOT file and only want to plot the tree and store the information of each node, you can use the following call
   ```
   tmc_obj.store_outputs(load_dot_file=True)
   ```
9. If you want to visualize your results in a dynamic platform, I strongly recommend the tool [too-many-cells-interactive](https://github.com/schwartzlab-methods/too-many-cells-interactive?tab=readme-ov-file). To use it, first make sure that you have Docker Compose and Docker. One simple way of getting the two is by installing [Docker Desktop](https://docs.docker.com/compose/install/). If you use [Nix](https://search.nixos.org/packages), simply add the packages `pkgs.docker` and `pkgs.docker-compose` to your configuration or `home.nix` file and run
```
home-manager switch
```
9. If you installed Docker Desktop you probably don't need to follow this step. However, under some distributions the following two commands have proven to be essential.
```
sudo dockerd
```
to start the daemon service for docker containers and
```
sudo chmod 666 /var/run/docker.sock
```
to let Docker read and write to that location.

10. Now clone the repository 
   ```
   git clone https://github.com/schwartzlab-methods/too-many-cells-interactive.git
   ```
   and store the path to the `too-many-cells-interactive` folder in a variable, for example `path_to_tmc_interactive`. Also, you will need to identify a column in your `AnnData.obs` data frame that has the labels for the cells. Let's assume that the column name is stored in the variable `cell_annotations`. Lastly, you can provide a port number to host your visualization, for instance `port_id=1234`. Then, you can call the function
   ```
   tmc_obj.visualize_with_tmc_interactive(
            path_to_tmc_interactive,
            cell_annotations,
            port_id)
   ```
   The following visualization corresponds to the data set with ~90K cells (observations).
   ![Visualization example](https://github.com/JRR3/toomanycells/blob/main/tests/example_1.png)
   
   And this is the visualization for the Tabula Sapiens data set with ~480K cells.
   ![Visualization example](https://github.com/JRR3/toomanycells/blob/main/tests/tmci_tabula_sapiens.png)

## What is the time complexity of toomanycells (à la Python)?
To answer that question we have created the following benchmark. We tested the performance of toomanycells in 20 data sets having the following number of cells: 6360, 10479, 12751, 16363, 23973, 32735, 35442, 40784, 48410, 53046, 57621, 62941, 68885, 76019, 81449, 87833, 94543, 101234, 107809, 483152. The range goes from thousands of cells to almost half a million cells. These are the results.
![Visualization example](https://github.com/JRR3/toomanycells/blob/main/tests/log_linear_time.png)
![Visualization example](https://github.com/JRR3/toomanycells/blob/main/tests/log_linear_iter.png)
As you can see, the program behaves linearly with respect to the size of the input. In other words, the observations fit the model $T = k\cdot N^p$, where $T$ is the time to process the data set, $N$ is the number of cells, $k$ is a constant, and $p$ is the exponent. In our case $p\approx 1$. Nice!

## Similarity functions
So far we have assumed that the similarity matrix 
$S$ is
computed by calculating the cosine of the angle 
between each observation. Concretely, if the 
matrix of observations is $B$ ($m\times n$), the $i$-th row
of $B$ is $x = B(i,:)$ and the $j$-th row of $B$ 
is $y=B(j,:)$, then 
$$S(i,j)=\frac{x\cdot y}{||x||_2\cdot ||y||_2}$$
However, this is not the only similarity 
function. We will list all the available
similarity functions and how to call them.
### Cosine (sparse)
If your matrix is sparse, i.e., the number of nonzero
entries is proportional to the number of samples ($m$),
and you want to use the cosine similarity, then use the
following instruction.
```
tmc_obj.run_spectral_clustering(
   similarity_function="cosine_sparse")
```
By default we use the Halko-Martinsson-Tropp algorithm 
to compute the truncated singular value decomposition.
However, the ARPACK library (written in Fortran)
is also available.
```
tmc_obj.run_spectral_clustering(
   similarity_function="cosine_sparse",
   svd_algorithm="arpack")
```
If $B$ has negative entries, it is possible
to get negative entries for $S$. This could
in turn produce negative row sums for $S$. 
If that is the case,
the convergence to a solution could be extremely slow.
However, if you use the non-sparse version of this
function, we provide a reasonable solution to this problem.
### Cosine
If your matrix is dense, 
and you want to use the cosine similarity, then use the
following instruction.
```
tmc_obj.run_spectral_clustering(
   similarity_function="cosine")
```
The same comment about negative entries applies here.
However, there is a simple solution. While shifting
the matrix of observations can drastically change the
interpretation of the data, shifting the similarity 
matrix is actually a reasonable method to remove negative
entries. The reason is that shifting by a constant is
an order-preserving transformation, which implies
that if the similarity between $x$ and $y$ is
less than the similarity between $u$ and $w$, i.e.,
$S(x,y)<S(u,w)$, then $S(x,y)+1 < S(u,w)+1$. Samples
have no natural order, but similarities do.
To shift the (dense) similarity matrix by $s=1$, use the 
following instruction.
```
tmc_obj.run_spectral_clustering(
   similarity_function="cosine",
   shift_similarity_matrix=1)
```
Note that since the range of the cosine similarity
is $[-1,1]$, the shifted range with $s=1$ becomes $[0,2]$.

### Laplacian
The similarity matrix is
$$S(x,y)=\exp(-\gamma\cdot ||x-y||_1)$$
This is an example:
```
tmc_obj.run_spectral_clustering(
   similarity_function="laplacian",
   similarity_gamma=0.01)
```
This function is very sensitive to $\gamma$. Hence, an
inadequate choice can result in poor results or 
no convergence. If you obtain poor results, try using  
a smaller value for $\gamma$.

### Gaussian
The similarity matrix is
$$S(x,y)=\exp(-\gamma\cdot ||x-y||_2^2)$$
This is an example:
```
tmc_obj.run_spectral_clustering(
   similarity_function="gaussian",
   similarity_gamma=0.001)
```
As before, this function is very sensitive to $\gamma$. 
Note that the norm is squared. Thus, it transforms
big differences between $x$ and $y$ into very small
quantities.

### Divide by the sum
The similarity matrix is 
$$S(x,y)=1-\frac{||x-y||_p}{||x||_p+||y||_p},$$
where $p =1$ or $p=2$. The rows 
of the matrix are normalized (unit norm)
before computing the similarity.
This is an example:
```
tmc_obj.run_spectral_clustering(
   similarity_function="div_by_sum")
```

## Additional features
### TF-IDF
If you want to use the inverse document
frequency (IDF) normalization, then use
```
tmc_obj.run_spectral_clustering(
   similarity_function="some_sim_function",
   use_tf_idf=True)
```
If you also want to normalize the frequencies to
unit norm, then use
```
tmc_obj.run_spectral_clustering(
   similarity_function="some_sim_function",
   use_tf_idf=True,
   tf_idf_norm="l2")
```
to use the $2$-norm, or
```
tmc_obj.run_spectral_clustering(
   similarity_function="some_sim_function",
   use_tf_idf=True,
   tf_idf_norm="l1")
```
to use the $1$-norm.

### Normalization
Sometimes normalizing your matrix
of observations can improve the
performance of some routines. 
To normalize the row, use the following instruction.
```
tmc_obj.run_spectral_clustering(
   similarity_function="some_sim_function",
   normalize_rows=True)
```
Be default, the $2$-norm is used. To 
use any other $p$-norm, use
```
tmc_obj.run_spectral_clustering(
   similarity_function="some_sim_function",
   normalize_rows=True,
   similarity_norm=p)
```


## Acknowledgments
I would like to thank 
the Schwartz lab (GW) for 
letting me explore different
directions and also Christie Lau for
providing multiple test 
cases to improve this 
implementation. 