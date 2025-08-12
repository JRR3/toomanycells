# too-many-cells (à la Python)


[![image](https://img.shields.io/pypi/v/toomanycells.svg)](https://pypi.python.org/pypi/toomanycells)

### It's [Scanpy](https://github.com/scverse/scanpy) friendly!
### Please remember to [cite](https://doi.org/10.1093/gigascience/giae056) us

**A Python package for spectral clustering** based on the
powerful suite of tools named
[too-many-cells](https://github.com/GregorySchwartz/too-many-cells).
Note, **this is not a wrapper**.
In essence, you can use toomanycells to partition a data set
in the form of a matrix of integers or floating point numbers
into clusters, where members of a cluster are similar to each
other under a given similarity function.
The rows represent observations and the
columns are the features. However, sometimes just knowing the
clusters is not sufficient. Often, we are interested on the
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
paper](https://www.nature.com/articles/s41592-020-0748-5).


-   Free software: GNU AFFERO GENERAL PUBLIC LICENSE
-   Documentation: https://JRR3.github.io/toomanycells

## Dependencies
Version 1.0.40 no longer requires Graphviz. Thus,
no need to install a separate C library!

## Virtual environments
To have control of your working environment 
you can use a python virtual environment, which 
can help you keep only the packages you need in 
one location. In bash or zsh you can simply
type
```
python -m venv /path/to/new/virtual/environment
```
To activate it you simply need
```
source pathToTheVirtualEnvironment/bin/activate
```
To deactivate the environment use the intuitive
```
deactivate
```

## Installation

Caveat: I have tested the following steps in
**Python 3.9.18**. For other versions, things might
be different.

In theory, just typing
```
pip install toomanycells
```
in your home or custom environment 
should work. However, for reproducibility,
here is the list of packages I had in my virtual 
environment in 2024:
```
anndata==0.10.9
array_api_compat==1.8
celltypist==1.6.3
certifi==2024.8.30
charset-normalizer==3.3.2
click==8.1.7
contourpy==1.3.0
cycler==0.12.1
et-xmlfile==1.1.0
exceptiongroup==1.2.2
fonttools==4.54.1
get-annotations==0.1.2
h5py==3.11.0
idna==3.10
igraph==0.11.6
importlib_resources==6.4.5
joblib==1.4.2
kiwisolver==1.4.7
legacy-api-wrap==1.4
leidenalg==0.10.2
llvmlite==0.43.0
matplotlib==3.9.2
natsort==8.4.0
networkx==3.2.1
numba==0.60.0
numpy==2.0.2
openpyxl==3.1.5
packaging==24.1
pandas==2.2.3
patsy==0.5.6
pillow==10.4.0
plotly==5.24.1
pynndescent==0.5.13
pyparsing==3.1.4
python-dateutil==2.9.0.post0
pytz==2024.2
requests==2.32.3
scanpy==1.10.3
scikit-learn==1.5.2
scipy==1.13.1
seaborn==0.13.2
session-info==1.0.0
six==1.16.0
statsmodels==0.14.3
stdlib-list==0.10.0
tenacity==9.0.0
texttable==1.7.0
threadpoolctl==3.5.0
toomanycells==1.0.52
tqdm==4.66.5
tzdata==2024.2
umap-learn==0.5.6
urllib3==2.2.3
zipp==3.20.2
```
If you want to install an updated
version, then please use the following 
approach.
```
pip install -U --no-deps toomanycells
```
Note that we are requiring to keep all the 
dependencies as they are. Otherwise they would
get upgraded and that could potentially **break**
the installation.

To install packages based on a list of requirements,
i.e., the packages you want installed with the specific
version, then use
```
pip install -r requirements.txt
```
where `requirements.txt` is a list
like the one shown the above block of code.

Make sure you have the latest version. If not,
run the previous command again.

## Quick run (needs to be updated)
If you want to see a concrete example of how
to use toomanycells, check out the jupyter 
notebook [demo](./toomanycells_demo.ipynb).

## Quick plotting
If you are already familiar with toomanycells and
want to generate a quick plot (an SVG) of your tree after 
calling 
   ```
   tmc_obj.run_spectral_clustering()
   ```
then use the following call
   ```
   tmc_obj.store_outputs(
       cell_ann_col="name_of_the_column",
       plot_tree=True,
   )
   ```
   or
   ```
   tmc_obj.store_outputs(
       cell_ann_col="name_of_the_column",
       plot_tree=True,
       draw_modularity=False,
       draw_node_numbers=False,
   )
   ```
with the appropriate flags.
If you already have the outputs and you 
just want to plot, then simply call
   ```
   tmc_obj.easy_plot(
      cell_ann_col="name_of_the_column",
   )
   ```
   or
   ```
   tmc_obj.easy_plot(
      cell_ann_col="name_of_the_column",
      draw_modularity=False,
      draw_node_numbers=False,
   )
   ```
with the appropriate flags,
where `name_of_the_column` is the name of the AnnData.obs 
column that contains the cell annotations.
This function will look for the outputs in the 
folder that you defined for the `output_directory`
as shown in step 2.
Note that this function relies on
[too-many-cells](https://gregoryschwartz.github.io/too-many-cells/)
(à la Haskell). So you need to have it installed. If you work
within the cluster of your organization, maybe it has 
already been installed, and you could load it as follows.
   ```
   module add too-many-cells
   ```
Otherwise, I recommend you to install it with Nix.

## Generating a matrix market file
Sometimes you want to generate a matrix market
file from a set of genes so that you can visualize
them with other tools. The function that will help you 
with this is called `create_data_for_tmci`. 
Just for context,
imagine you are interested in two genes, `COL1A1`, and
`TCF21`. Moreover, imagine that you also want to include
in your matrix another feature located in the `obs` data
frame called `total_counts`. Finally, assume that the
column that contains the labels for your cells is called
`cell_annotations`. Please use this a template for your
specific needs.
   ```
   from toomanycells import TooManyCells as tmc
   tmc_obj = tmc(A)
   tmc_obj.run_spectral_clustering()                                 
   tmc_obj.store_outputs(cell_ann_col="cell_annotations")            
   list_of_genes = []                                            
   list_of_genes.append("COL1A1")                                
   list_of_genes.append("TCF21")                                 
   list_of_genes.append("total_counts")                          
   tmc_obj.create_data_for_tmci(list_of_genes=list_of_genes)    
   ```
These lines of code will produce for you
a folder named `tmci_mtx_data` with the expected outputs.

## Starting from scratch
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
   as follows.
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
   tmc_obj = tmc(input_directory,
                 output_directory,
                 input_is_matrix_market=True)
   ```
Under this scenario, the `input_directory` must contain a
`.mtx` file, a `barcodes.tsv` file (the observations), and
a `genes.tsv` (the features).

5. Once your data has been loaded successfully, 
you can start the clustering process with the following command
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
   or
   ```
   tmc_obj.store_outputs(
       cell_ann_col="name_of_the_column",
       plot_tree=True,
       )
   ```
   to include a plot of the graph.

This call will generate JSON file
containing the nodes and edges of the graph (`graph.json`), 
one CSV file that describes the cluster
information (`clusters.csv`), another CSV file containing 
the information of each node (`node_info.csv`), and two 
JSON files.  One relates cells to clusters 
(`cluster_list.json`), and the other has the 
full tree structure (`cluster_tree.json`). You need this
last file for too-many-cells interactive (TMCI).

8. If you already have the `graph.json` file you can 
load it with
   ```
   tmc_obj.load_graph(json_fname="some_path")
   ```
## Visualization with TMCI
9. If you want to visualize your results in a dynamic
platform, I strongly recommend the tool
[too-many-cells-interactive](https://github.com/schwartzlab-methods/too-many-cells-interactive?tab=readme-ov-file).
To use it, first make sure that you have Docker Compose and
Docker. One simple way of getting the two is by installing
[Docker Desktop](https://docs.docker.com/compose/install/).
Note that with MacOS the instructions are slightly different.
If you use [Nix](https://search.nixos.org/packages), simply
add the packages `pkgs.docker` and `pkgs.docker-compose` to
your configuration or `home.nix` file and run
```
home-manager switch
```
10.  If you installed Docker Desktop you probably don't need to
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


11.  Now clone the repository 
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

## Running TMCI independently
In case you already have the outputs for TMCI, but you
want to visualize a specific set of genes on top of
your tree, you are going to need the expression matrix
corresponding to those genes in the matrix marker format.
You will also need a list of genes and the barcodes.
All of that can be easily achieved with toomanycells
(à la Python) after loading your matrix or AnnData
object. If you are interested in only a few genes,
you can call 
   ```
   tmc_obj.create_data_for_tmci(
      list_of_genes = ["G1","G2",...,"Gn"]
   )
   ```
   where `G1`,`G2`,...,`Gn`, are the labels
   of the genes of interest. If instead you have
   a table of genes stored as a text file, then
   use the call
   ```
   tmc_obj.create_data_for_tmci(
      path_to_genes = "path/to/genes.csv"
   )
   ```
   Lastly, if you want to write all the available genes
   to a matrix, then simply call
   ```
   tmc_obj.create_data_for_tmci()
   ```
   but note that this could take a considerable
   amount of time, depending on how many genes
   are in your matrix.
   After calling this function, you will
   have a new folder called `tmci_mtx_data`
   which will contain the aforementioned files.
   It is also important to mention that you need
   a file wiht the labels

   ```
   ./start-and-load.sh \
    --matrix-dir /path_to/tmci_mtx_data \
    --tree-path /path_to/cluster_tree.json \
    --label-path /path_to/cell_annotations.csv \
    --port 2025 \
    --debug
   ```

## What is the time complexity of toomanycells (à la Python)?
To answer that question we have created the following
benchmark. We tested the performance of toomanycells in 20
data sets having the following number of cells: 6,360, 10,479,
12,751, 16,363, 23,973, 32,735, 35,442, 40,784, 48,410, 53,046,
57,621, 62,941, 68,885, 76,019, 81,449, 87,833, 94,543, 101,234,
107,809, and 483,152. The range goes from thousands of cells to
almost half a million cells. 
These are the results.
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
[CellTypist](https://www.celltypist.org/). Simply 
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
[list](https://www.celltypist.org/models). Lastly,
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

## Graph operations

### Selecting cells through branches

Imagine you have a tree structure 
of your data like the one shown below.
![Branches](https://github.com/JRR3/toomanycells/blob/main/tests/4plex_data_full_2_branches.svg)
If you want to isolate all the cells that belong to branches
261 and 2, and produce an AnnData object with those cells,
simply use the following call

```
   adata = tmc_obj.isolate_cells_from_branches(
      list_of_branches=[261,2])
```

If you have a CSV file that specifies the branches,
then use the following call

```
   adata = tmc_obj.isolate_cells_from_branches(
    path_to_csv_file="list_of_branches.csv",
    branch_column="node",
   )
```

The name of the column that contains the branches
or nodes is specified through the keyword 
`branch_column`. Lastly, if you want to store 
a copy of the indices, use the following call

```
   adata = tmc_obj.isolate_cells_from_branches(
    path_to_csv_file="list_of_branches.csv",
    branch_column="node",
    generate_cell_id_file=True,
   )
```

### Mean expression of a branch
Imagine we have the following tree.
![TreeWithLabels](https://github.com/JRR3/toomanycells/blob/main/tests/4plex_data_full.svg)
If you want to quantify the mean expression of the marker 
CD9 on branch 261, you can use the following call

```
   m_exp = tmc_obj.compute_cluster_mean_expression(
        node=261, genes=["CD9"])
```

and you would obtain 12.791.

![Expression](https://github.com/JRR3/toomanycells/blob/main/tests/4plex_cd9_exp_full.svg)

Looking at the above plot, this suggests that Neuro-2a cells
highly express this marker.
If instead we were interested in a different marker, like
SDC1, this would be the corresponding color map
expression across the nodes.

![Expression](https://github.com/JRR3/toomanycells/blob/main/tests/4plex_sdc1_exp_full.svg)

The above plot also illustrates that some Neuro-2a cells
are rich in SDC1.

### Median absolute deviation classification

First we introduce the concept of median absolute 
deviation. Imagine you have a list of $n$ observations
$L = [z_0,z_2,\ldots,z_{n-1}]$. Let 
$\mathcal{M}:\mathbb{R}^n \to \mathbb{R}$ be
the function that computes the median of a list.
Consider a new list $K$, where 
$K_i = \left| L_i - \mathcal{M}(L) \right|$. Then,
the median absolute deviation of $L$ is the 
median of the absolute differences between the
original value and the median. Mathematically,
$\text{MAD}(L) = \mathcal{M}(K)$. For this section
we will be indicating the expression of a gene in terms
of MADs. The reason is that we want to classify cells,
and using quantities that capture the dispersion of the
data is a convenient approach for that purpose.
An important point to mention is that
for each gene,
instead of considering the raw expression values
across all cells as the elements of the list $L$, 
we use the mean expression for each node of the tree.
In other words, for a given gene,
the element $z_k$ represents
the mean expression of that gene for node $k$. Thus,
$n$ indicates the number of nodes in the tree.

Based on the previous example, now imagine
you want to find cells whose expression
of two markers, CD9 and SDC1, is 1 MAD above the median.
First, you need a CSV file containing the following 
information. 

```
     Marker      Cell  Threshold Direction
       CD9  Neuro-2a        1.0     Above
      SDC1  Neuro-2a        1.0     Above
```

Let's call it `marker_and_cell_info.csv`.
**Note**: For this discussion the cell types indicated in 
the `Cell` column are not relevant and will not be 
used. We quantify the mean expression of those 
markers for every node of the tree and store 
that information within each node.
We can do that using the following call.

```
tmc_obj.populate_tree_with_mean_expression_for_all_markers(
    cell_marker_path="marker_and_cell_info.csv")
```

Then we compute basic statistics for each marker using the
following function

```
tmc_obj.compute_node_expression_metadata()
```

These are
the statistics associated to those markers.

```
           median        mad       min         max   min_mad      max_mad       delta
CD9      3.080538   1.918258  0.000890   22.944445 -1.605441    10.355182    0.797375
SDC1     2.989691   1.165005  0.001669    6.639456 -2.564814     3.132832    0.379843
```

Note that the maximum expression of CD9 
is about 10 MADs above the median, while that of
SDC1 is only about 3 MADs above the median.
The plot corresponding to the distribution of those
markers across all nodes can be generated through this call

```
tmc_obj.plot_marker_distributions()
```

The plots will be all contained in a dynamic html file. 
Here are some examples.

This is the distribution for CD9:

![nodalDist](https://github.com/JRR3/toomanycells/blob/main/tests/marker_nodal_distribution/cd9_nodal_distribution_homemade.png)

and with TooManyCellsInteractive

![nodalDist](https://github.com/JRR3/toomanycells/blob/main/tests/marker_nodal_distribution/cd9_nodal_distribution_tmci.png)

The distribution for SDC1 looks as follows.

![nodalDist](https://github.com/JRR3/toomanycells/blob/main/tests/marker_nodal_distribution/sdc1_nodal_distribution_homemade.png)

If we want to isolate the cells that satisfy the conditions

```
     Marker      Cell  Threshold Direction
       CD9  Neuro-2a        1.0     Above
      SDC1  Neuro-2a        1.0     Above
```

We can use the call

```
tmc_obj.select_cells_based_on_inequalities(
    cell_ann_col="cell_type")
```

where the cell annotation column in the `.obs` 
dataframe is specified through the
`cell_ann_col` keyword. This function will 
return an AnnData object with all the
cells satisfying all the constraints.

This function will also produce
multiple CSV files. One for each inequality
specified through the file of constraints.
For example,
one for all cells whose 
expression of CD9 was above 1 MAD of the median 
expression of CD9,
one for all cells whose 
expression of SDC1 was above 1 MAD of the median 
expression of SDC1, 
and one corresponding to the intersection
of all of the above. The above function will 
modify the original AnnData object by adding to
the `.obs` dataframe a column
named `Intersection`
indicating with a boolean value 
if a cell satisfies all the constraints.

Lastly, if the number of markers is less than or
equal to three, then the `.obs` dataframe will
include a column classifying the cells 
based on whether they express
highly or not each of the markers. For instance,
in this example we obtained the following outputs.

```
Class
CD9-Low-SDC1-Low      28729
CD9-High-SDC1-Low      6072
CD9-High-SDC1-High     4223
CD9-Low-SDC1-High      2058
Name: count, dtype: int64
Class
CD9-Low-SDC1-Low      0.699309
CD9-High-SDC1-Low     0.147802
CD9-High-SDC1-High    0.102794
CD9-Low-SDC1-High     0.050095
Name: proportion, dtype: float64
```

This indicates that the majority of the cells,
i.e., about `70%` of cells,
are low in CD9 and low in SDC1, and about `10%` of cells
are high in both. Note that in this particular example
when we say high it means
that the expression is above 1 MAD from the median, and
low is the complement of that.

![Expression](https://github.com/JRR3/toomanycells/blob/main/tests/4plex_data_full_marker_class.svg)


## Heterogeneity quantification
Imagine you want to compare the heterogeneity of cell 
populations belonging to different branches of the 
toomanycells tree. By branch we mean all the nodes that
derive from a particular node, including the node 
that defines the branch in question.
For example, we want to compare branch 1183 against branch 2.
![heterogeneity](https://github.com/JRR3/toomanycells/blob/main/tests/heterogeneity.svg)
One way to do this is by comparing the modularity 
distribution and the cumulative modularity for all the 
nodes that belong to each branch. 
 We can do that using the following calls. First for 
 branch 1183
```
   tmc_obj.quantify_heterogeneity(
      list_of_branches=[1183],
      use_log_y=true,
      tag="branch_A",
      show_column_totals=true,
      color="blue",
      file_format="svg")
```
<br/>
<img src="https://github.com/JRR3/toomanycells/blob/main/tests/branch_A.svg"
width="500" height="420"/>
<br/>
And then for branch 2

```
   tmc_obj.quantify_heterogeneity(
      list_of_branches=[2],
      use_log_y=true,
      tag="branch_B",
      show_column_totals=true,
      color="red",
      file_format="svg")
```
<br/>
<img src="https://github.com/JRR3/toomanycells/blob/main/tests/branch_B.svg"
width="500" height="420"/>
<br/>
Note that you can include multiple nodes in the 
list of branches.
From these figures we observe that the higher cumulative 
modularity of branch 1183 with respect to branch 2 suggests 
that the former has a higher degree of heterogeneity.
However, just relying in modularity could provide a 
misleading interpretation. For example, consider the 
following scenario where the numbers within the nodes 
indicate the modularity at that node.
<br/>
<img src="https://github.com/JRR3/toomanycells/blob/main/tests/counter_node_modularity.svg"
width="300" height="400"/>
<br/>
In this case, scenario A has a larger cumulative modularity, 
but we note that scenario B is more heterogeneous.
For that reason we recommend also computing additional
diversity measures. First, we need some notation. 
For all the branches belonging to the list of branches in the
above function 

`quantify_heterogeneity`, let $C$ be
the set of leaf nodes that belong to those branches. 
We consider each leaf node as a separate species, and we 
call the whole collection of cells an ecosystem.
For $c_i \in C$, let $|c_i|$ be the number of cells in
$c_i$ and $|C| = \sum_i |c_i|$ the total number 
of cells contained in the given branches. If we let

$$p_i = \dfrac{|c_i|}{|C|},$$

then we define the following diversity measure

$$D(q) = \left(\sum_{i=1}^{n} p_i^q \right)^{\frac{1}{1-q}}.
$$

In general, the larger the value of $D(q)$, the more diverse
is the collection of species. Note that $D(q=0)$ 
describes the total number of species. We 
call this quantity the richness of the ecosystem. 
When $q=1$, $D$ is the exponential of the Shannon entropy

$$H = -\sum_{i=1}^{n}p_i \ln(p_i).$$

When $q=2$, $D$ is 
the inverse of the Simpson's index

$$S = \sum_{i=1}^{n} (p_i)^2,$$

which represents the
probability that two cells picked at random belong
to the same species. Hence, the higher the Simpson's
index, the less diverse is the ecosystem. 
Lastly, when $q=\infty$, $D$ is the inverse of
the largest proportion $\max_i(p_i)$.

In the above example, for branch 1183 we obtain
```
               value
Richness  460.000000
Shannon     5.887544
Simpson     0.003361
MaxProp     0.010369
q = 0     460.000000
q = 1     360.518784
q = 2     297.562094
q = inf    96.442786
```
and for branch 2 we obtain
```
               value
Richness  280.000000
Shannon     5.500414
Simpson     0.004519
MaxProp     0.010750
q = 0     280.000000
q = 1     244.793371
q = 2     221.270778
q = inf    93.021531
```
After comparing the results using two different measures,
namely, modularity and diversity, we conclude that branch
1183 is more heterogeneous than branch 2.

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
By default we use the ARPACK library (written in Fortran)
to compute the truncated singular value decomposition.
The Halko-Martinsson-Tropp algorithm 
is also available. However, this one is not deterministic.
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

### Dimension-adaptive Euclidean Norm (DaEN)
If your data consists of points whose Euclidean
norm varies across multiple length scales, then 
one option is to use a similarity function that
can adapt to those changes in magnitude.
Before I explain it in detail, here is how
you can call it
```
tmc_obj.run_spectral_clustering(
    similarity_function="norm_sparse")
```

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
The shift transformation can also be applied to any of 
the subsequent similarity matrices.

### Laplacian
The similarity matrix is given by

$$
S(x,y)=\exp(-\gamma\cdot \left\lVert x-y \right\rVert _1).
$$

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
The similarity matrix is given by

$$
S(x,y)=\exp(-\gamma\cdot \left\lVert x-y\right\rVert _2^2).
$$

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
The similarity matrix is given by

$$
S(x,y)=1-\frac{
   \left\lVert x-y \right\rVert_p
   }{
      \left\lVert x \right\rVert_p +
       \left\lVert y \right\rVert_p
   },
$$

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
fluctuates as we move from node $X$ 
(lower left side of the tree), which is rich
in `Class B`, to node $Y$ (upper left side of the tree), 
which is rich in `Class
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
the distance between $C_1$ and $C_2$ as 

$$d(C_1,C_2) = Q(P).$$

We also define $d(C_i,P) = Q(P)/2$. Note that with 
those definitions we have that 

$$d(C_1,C_2)=d(C_1,P) +d(P,C_2)=Q(P)/2+Q(P)/2=Q(P),$$

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
