# toomanycells


[![image](https://img.shields.io/pypi/v/toomanycells.svg)](https://pypi.python.org/pypi/toomanycells)
[![image](https://img.shields.io/conda/vn/conda-forge/toomanycells.svg)](https://anaconda.org/conda-forge/toomanycells)


**A python package for spectral clustering based on the powerful suite of tools named [too-many-cells](https://github.com/GregorySchwartz/too-many-cells). In essence, you can use toomanycells to partition a data set in the form of a matrix of integers or floating point numbers into clusters. Initially, toomanycells will partition your data set into two subsets, trying to maximize the differences between the two. Subsequently, it will reapply that same criterion to each subset and will continue bifurcating until the [modularity](https://en.wikipedia.org/wiki/Modularity_(networks)) of the parent becomes negative, implying that the current set is fairly homogeneous, and consequently suggesting that further partitioning is not warranted. Thus, when the process finishes, you end up with a tree structure of your data set, where the leaves represent the clusters. As mentioned earlier, you can use this tool with any kind of data. However, a common application is to classify cells. You can read about this application in this [Nature paper](https://www.nature.com/articles/s41592-020-0748-5). **


-   Free software: BSD License
-   Documentation: https://JRR3.github.io/toomanycells
    

## Features

-   TODO
