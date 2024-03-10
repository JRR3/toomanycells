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
    def __getitem__(self, indices: list | int):
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
