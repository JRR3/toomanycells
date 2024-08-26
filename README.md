# too-many-cells (à la Python)


[![image](https://img.shields.io/pypi/v/toomanycells.svg)](https://pypi.python.org/pypi/toomanycells)

### It's [Scanpy](https://github.com/scverse/scanpy) friendly!

** A python package for spectral clustering based on the
powerful suite of tools named
[too-many-cells](https://github.com/GregorySchwartz/too-many-cells).
In essence, you can use toomanycells to partition a data set
in the form of a matrix of integers or floating point numbers
into clusters, where members of a cluster are similar to each
other. The rows represent observations and the
columns are the features. However, sometimes just knowing the
clusters is not sufficient. Often, we are insterested on the
relationships between the clusters, and this tool can help 
you visualize the clusters as leaf nodes of a tree, where the
branches illustrate the trajectories that have to be followed
to reach a particular cluster. Initially, this tool will
partition your data set into two subsets (each subset is a 
node of the tree), trying to maximize
the differences between the two. Subsequently, it will
reapply that same criterion to each subset (node) and will 
continue bifurcating until the
[modularity](https://en.wikipedia.org/wiki/Modularity_(networks))
of the node that is about to be partitioned becomes less 
than a given threshold value ($10^{-9}$ by default), 
implying that the elements belonging
to the current node are fairly homogeneous, 
and consequently suggesting
that further partitioning is not warranted. Thus, when the
process finishes, you end up with a tree structure of your
data set, where the leaves represent the clusters. As
mentioned earlier, you can use this tool with any kind of
data. However, a common application is to classify cells and
therefore you can provide an
[AnnData](https://anndata.readthedocs.io/en/latest/) object.
You can read about this application in this [Nature Methods
paper](https://www.nature.com/articles/s41592-020-0748-5).**


-   Free software: GNU AFFERO GENERAL PUBLIC LICENSE
-   Documentation: https://JRR3.github.io/toomanycells

## Dependencies

Make sure you have installed the graph visualization library
[Graphviz](https://www.graphviz.org). For example, if you
want to use conda, then do the following.
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
in your home environment. If you want to install an updated
version, then use the following flag.
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

2. If you already have an 
[AnnData](https://anndata.readthedocs.io/en/latest/) 
object `A` loaded into memory, then you can create a 
TooManyCells object with
   ```
   tmc_obj = tmc(A)
   ```
   In this case the output folder will be called 
   `tmc_outputs`.  However, if you want the output folder to
   be a particular directory, then you can specify the path 
   as follows
   ```
   tmc_obj = tmc(A, output_directory)
   ```
3. If instead of providing an AnnData object you want to
provide the directory where your data is located, you can use
the syntax
   ```
   tmc_obj = tmc(input_directory, output_directory)
   ```

4. If your input directory has a file in the [matrix market
format](https://math.nist.gov/MatrixMarket/formats.html),
then you have to specify this information by using the
following flag
   ```
   tmc_obj = tmc(input_directory, output_directory, input_is_matrix_market=True)
   ```
Under this scenario, the `input_directory` must contain a
`.mtx` file, a `barcodes.tsv` file (the observations), and
a `genes.tsv` (the features).

5. Once your data has been loaded successfully, you can start the clustering process with the following command
   ```
   tmc_obj.run_spectral_clustering()
   ```
In my desktop computer processing a data set with ~90K
cells (observations) and ~30K genes (features) took a
little less than 6 minutes in 1809 iterations. For a
larger data set like the [Tabula
Sapiens](https://figshare.com/articles/dataset/Tabula_Sapiens_release_1_0/14267219?file=40067134)
with 483,152 cells and 58,870 genes (14.51 GB in zip
format) the total time was about 50 minutes in the same
computer. ![Progress bar example](https://github.com/JRR3/toomanycells/blob/main/tests/tabula_sapiens_time.png)
    
6. At the end of the clustering process the `.obs` data frame of the AnnData object should have two columns named `['sp_cluster', 'sp_path']` which contain the cluster labels and the path from the root node to the leaf node, respectively.
   ```
   tmc_obj.A.obs[['sp_cluster', 'sp_path']]
   ```
7. To generate the outputs, just call the function
   ```
   tmc_obj.store_outputs()
   ```
This call will generate a graphical representation of the 
tree (`output_graph.svg`), a DOT file
containing the nodes and edges of the graph (`graph.dot`), 
one CSV file that describes the cluster
information (`clusters.csv`), another CSV file containing 
the information of each node (`node_info.csv`), and two 
JSON files.  One relates cells to clusters 
(`cluster_list.json`), and the other has the 
full tree structure (`cluster_tree.json`). You need this
last file for TMCI.

For those who may have problems installing `pygraphviz`,
you can still store the main outputs using the call
   ```
   tmc_obj.store_outputs(store_tree_svg=False)
   ```
Note that in this case you will not be able to 
generate the `output_graph.svg` and `graph.dot`
files. However, the `cluster_tree.json` file, which is the 
most important file, will still be generated, and
you can continue working with this tutorial.

8. If you already have a DOT file you can load it with
   ```
   tmc_obj.load_graph(dot_fname="some_path")
   ```
or plot it with
   ```
   tmc_obj.plot_radial_tree_from_dot_file(
      dot_fname="some_path")
   ```
9. If you want to visualize your results in a dynamic
platform, I strongly recommend the tool
[too-many-cells-interactive](https://github.com/schwartzlab-methods/too-many-cells-interactive?tab=readme-ov-file).
To use it, first make sure that you have Docker Compose and
Docker. One simple way of getting the two is by installing
[Docker Desktop](https://docs.docker.com/compose/install/).
If you use [Nix](https://search.nixos.org/packages), simply
add the packages `pkgs.docker` and `pkgs.docker-compose` to
your configuration or `home.nix` file and run
```
home-manager switch
```
10. If you installed Docker Desktop you probably don't need to
follow this step. However, under some distributions the
following two commands have proven to be essential. Use
```
sudo dockerd
```
to start the daemon service for docker containers and
```
sudo chmod 666 /var/run/docker.sock
```
to let Docker read and write to that location.

11. Now clone the repository 
   ```
   git clone https://github.com/schwartzlab-methods/too-many-cells-interactive.git
   ```
and store the path to the `too-many-cells-interactive`
folder in a variable, for example
`path_to_tmc_interactive`. Also, you will need to identify
a column in your `AnnData.obs` data frame that has the
labels for the cells. Let's assume that the column name is
stored in the variable `cell_annotations`. Lastly, you can
provide a port number to host your visualization, for
instance `port_id=1234`. Then, you can call the function
   ```
   tmc_obj.visualize_with_tmc_interactive(
            path_to_tmc_interactive,
            cell_annotations,
            port_id)
   ```
The following visualization corresponds to the data set
with ~90K cells (observations). ![Visualization example](https://github.com/JRR3/toomanycells/blob/main/tests/example_1.png)
   
And this is the visualization for the Tabula Sapiens data set with ~480K cells.
![Visualization example](https://github.com/JRR3/toomanycells/blob/main/tests/tmci_tabula_sapiens.png)

## What is the time complexity of toomanycells (à la Python)?
To answer that question we have created the following benchmark. We tested the performance of toomanycells in 20 data sets having the following number of cells: 6360, 10479, 12751, 16363, 23973, 32735, 35442, 40784, 48410, 53046, 57621, 62941, 68885, 76019, 81449, 87833, 94543, 101234, 107809, 483152. The range goes from thousands of cells to almost half a million cells. These are the results.
![Visualization example](https://github.com/JRR3/toomanycells/blob/main/tests/log_linear_time.png)
![Visualization example](https://github.com/JRR3/toomanycells/blob/main/tests/log_linear_iter.png)
As you can see, the program behaves linearly with respect to the size of the input. In other words, the observations fit the model $T = k\cdot N^p$, where $T$ is the time to process the data set, $N$ is the number of cells, $k$ is a constant, and $p$ is the exponent. In our case $p\approx 1$. Nice!

## Cell annotation
### CellTypist
When visualizing the tree, we often are interested on
observing how different cell types distribute across the
branches of the tree. In case your AnnData object lacks
a cell annotation column in the ``obs`` data frame, or 
if you already have one but you want to try a different 
method, we have created a wrapper function that calls 
![CellTypist](https://www.celltypist.org/). Simply 
write
```
   tmc_obj.annotate_with_celltypist(
           column_label_for_cell_annotations,
   )
```
and the ```obs``` data frame of your AnnData object will 
have a column named like the string stored under the
```column_label_for_cell_annotations``` variable.
By default we use the ```Immune_All_High``` celltypist 
model that contains 32 cell types. If you want to use
another model, simply write
```
   tmc_obj.annotate_with_celltypist(
           column_label_for_cell_annotations,
           celltypist_model,
   )
```
where ```celltypist_model``` describes the type of model
to use by the library. For example, if this 
variable is equal to ```Immune_All_Low```, then the number 
of possible cell types increases to 98.
For a complete list of all the models, see the following
![list](https://www.celltypist.org/models). Lastly,
if you want to use the fact that transcriptionally similar
cells are likely to cluster together, you can assign the cell 
type labels on a cluster-by-cluster basis
 rather than a cell-by-cell basis. To activate this 
 feature, use the call
```
   tmc_obj.annotate_with_celltypist(
           column_label_for_cell_annotations,
           celltypist_model,
           use_majority_voting = True,
   )
```
### Median absolute deviation classification
Work in progress...

## Similarity functions
So far we have assumed that the similarity matrix 
$S$ is
computed by calculating the cosine of the angle 
between each observation. Concretely, if the 
matrix of observations is $B$ ($m\times n$), the $i$-th row
of $B$ is $x = B(i,:)$, and the $j$-th row of $B$ 
is $y=B(j,:)$, then the similarity between $x$ and
$y$ is
$$S(x,y)=\frac{x\cdot y}{||x||_2\cdot ||y||_2}.$$
However, this is not the only way to compute
a similarity matrix. We will list all the available
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
interpretation of the data because each column lives
in a different (gene) space, shifting the similarity 
matrix is actually a reasonable method to remove negative
entries. The reason is that similarities live in an 
ordered space and shifting by a constant is
an order-preserving transformation. Equivalently,
if the similarity between $x$ and $y$ is
less than the similarity between $u$ and $w$, i.e.,
$S(x,y) < S(u,w)$, then $S(x,y)+s < S(u,w)+s$ for
any constant $s$. The raw data
have no natural order, but similarities do.
To shift the (dense) similarity matrix by $s=1$, use the 
following instruction.
```
tmc_obj.run_spectral_clustering(
   similarity_function="cosine",
   shift_similarity_matrix=1)
```
Note that since the range of the cosine similarity
is $[-1,1]$, the shifted range for $s=1$ becomes $[0,2]$.
The shift transformation can also be applied to any of the subsequent
similarity matrices.

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

## Normalization

### TF-IDF
If you want to use the inverse document
frequency (IDF) normalization, then use
```
tmc_obj.run_spectral_clustering(
   similarity_function="some_sim_function",
   use_tf_idf=True)
```
If you also want to normalize the frequencies to
unit norm with the $2$-norm, then use
```
tmc_obj.run_spectral_clustering(
   similarity_function="some_sim_function",
   use_tf_idf=True,
   tf_idf_norm="l2")
```
If instead you want to use the $1$-norm, then
replace "l2" with "l1".

### Simple normalization
Sometimes normalizing your matrix
of observations can improve the
performance of some routines. 
To normalize the rows, use the following instruction.
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


## Gene expression along a path
### Introduction
Imagine you have the following tree structure after 
running toomanycells. 
![Tree path](https://github.com/JRR3/toomanycells/blob/main/tests/tree_path_example.svg)
Further, assume that the colors denote different classes
satisfying specific properties.  We want to know how the
expression of two genes, for instance, `Gene S` and `Gene T`,
fluctuates as we move from node $X$ (lower left side of the tree), which is rich
in `Class B`, to node $Y$ (upper left side of the tree), which is rich in `Class
C`. To compute such quantities, we first need to define the
distance between nodes. 
### Distance between nodes
Assume we have a (parent) node $P$ with
two children nodes $C_1$ and $C_2$. Recall that the modularity of 
$P$ indicates the strength of separation between the cell
populations of $C_1$ and $C_2$. 
A large the modularity indicates strong connections,
i.e., high similarity, within each cluster $C_i$,
and also implies weak connections, i.e., low similarity, between 
$C_1$ and $C_2$. If the modularity at $P$ is $Q(P)$, we define
the distance between $C_1$ and $C_2$ as $$d(C_1,C_2) = Q(P).$$
We also define $d(C_i,P) = Q(P)/2$. Note that with 
those definitions we have that 
```math
d(C_1,C_2)=d(C_1,P)
+d(P,C_2)=Q(P)/2+Q(P)/2=Q(P),
```
as expected. Now that we know how to calculate the
distance between a node and its parent or child, let 
$X$ and $Y$ be two distinct nodes, and let
${(N_{i})}_{i=0}^{n}$ be the sequence of nodes that describes
the unique path between them satisfying:

1. $N_0 = X$,
2. $N_n=Y$,
3. $N_i$ is a direct relative of $N_{i+1}$, i.e., 
$N_i$ is either a child or parent of $N_{i+1}$,
4. $N_i \neq N_j$ for $i\neq j$.

Then, the distance between $X$ and $Y$ is given by 
```math
d(X,Y) =
\sum_{i=0}^{n-1} d(N_{i},N_{i+1}).
```
### Gene expression
We define the expression
of `Gene G` at a node $N$, $Exp(G,N)$, as the mean expression
of `Gene G` considering all the cells that belong to node
$N$. Hence, given the sequence of nodes 
```math 
(N_i)_{i=0}^{n}
```
we can compute the corresponding gene
expression sequence 
```math
(E_{i})_{i=0}^{n}, \quad E_i = Exp(G,N_i).
```
### Cumulative distance
Lastly, since we are interested in plotting the
gene expression as a function of the distance with respect to
the node $X$, we define the sequence of real numbers
```math 
(D_{i})_{i=0}^{n}, \quad D_{i} = d(X,N_{i}).
```
### Summary
1. The sequence of nodes between $X$ and $Y$
$${(N_{i})}_{i=0}^{n}$$
2. The sequence of gene expression levels between $X$ and $Y$
$${(E_{i})}_{i=0}^{n}$$
3. And the sequence of distances with respect to node $X$
$${(D_{i})}_{i=0}^{n}$$

The final plot is simply $E_{i}$ versus $D_{i}$. An example
is given in the following figure.
### Example
![Gene expression](https://github.com/JRR3/toomanycells/blob/main/tests/exp_path_test.svg)

Note how the expression of `Gene A` is high relative to
that of `Gene B` at node $X$, and as we move
farther towards 
node $Y$ the trend is inverted and now `Gene B` is 
highly expressed relative to `Gene A` at node $Y$.

## Acknowledgments
I would like to thank 
the Schwartz lab (GW) for 
letting me explore different
directions and also Christie Lau for
providing multiple test 
cases to improve this 
implementation. 
