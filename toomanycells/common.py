import os
import json
import numpy as np
import pandas as pd
from typing import Union
#=====================================================
class MultiIndexList(list):
    """
    This class is derived from the list class.\
        It allows the use of iterables to \
        access the list. For example: \
        L[(1,2,0)] will access item #1 \
        of the list, then item #2 of the \
        previously retrieved item, and \
        finally item #0 of that last item.\
        We use this class to store the \
        TooManyCells tree in a structure \
        composed of nested lists and dictionaries.
    """
    #=================================================
    def __getitem__(self, indices: Union[list,int]):
        """
        This implementation of the __getitem__ method \
            allows the possibility of indexing a nested \
            list with a list of integers.
        """

        if hasattr(indices, '__iter__'):
            #If the indices object is iterable
            #then traverse the list using the indices.
            obj = self
            for index in indices:
                obj = obj[index]
            return obj
        else:
            #Otherwise, just use the __getitem__ 
            #method of the parent class.
            return super().__getitem__(indices)

#=====================================================
class JEncoder(json.JSONEncoder):
    def default(self, x):
        if isinstance(x, np.bool_):
            return bool(x)
        elif isinstance(x, np.integer):
            return int(x)
        elif isinstance(x, np.floating):
            return float(x)
        elif isinstance(x, np.ndarray):
            return x.tolist()
        else:
            return super().default(x)

#=====================================
def load_metadata_for_demo()-> pd.DataFrame:
    """
    This function loads the cell indices and \
    labels for the demo file.
    """
    fname = os.path.dirname(__file__)
    fname = os.path.join(fname, "data")
    fname = os.path.join(fname, "metadata.csv")
    df = pd.read_csv(fname, index_col = 0)
    return df