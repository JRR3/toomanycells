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

## Installation

Just type
```
pip install toomanycells
```
in your home environment. If you want to install an updated version, then use the following flag.
```
pip install toomanycells -U
```

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
