import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from scipy.sparse import issparse
from scipy.io import mmread

def get_count_matrix_(filename):
    """ Read count matrix in sparse format

    This function also loads the associated cell/barcode information from
    the .bct file.

    Parameters
    ----------
    filename : str
       Path to input matrix in matrix market format.
    shape : tuple(int)
       (Obsolete parameter) Target shape. Was used in an earlier version, before matrix market format was supported.
    header : bool
       (Obsolete parameter) header information
    offset : int
       (Obsolete parameter) offset

    Returns
    -------
        Sparse matrix in CSR format
    """
    if filename.endswith(".mtx"):
        return mmread(filename).tocsr()
    if filename.endswith('.npz'):
        files = np.load(filename)
        return csr_matrix((files['arr_0'], files['arr_1'], files['arr_2']))
    else:
        raise ValueError('unknown file format. Counts must be in mtx for npz format')

def get_regions_from_bed_(filename):
    """
    load a bed file

    Parameter
    ---------
    filename : str
       BED file

    Returns
    -------
        Region annotation from bed file as pd.DataFrame
    """
    regions = pd.read_csv(filename, sep='\t',
                          names=['chrom', 'start', 'end'],
                          usecols=[0,1,2])
    return regions

def get_cell_annotation(filename, header_=True):
    """ Load Cell/barcode information from '.bct' file

    Parameter
    ---------
    filename : str
       Filename prefix (without the .bct file ending)

    Returns
    -------
        Cell annotation as pd.DataFrame
    """
    if header_:
        df = pd.read_csv(filename, sep='\t')
    else:
        df = pd.read_csv(filename, sep='\t', header=None)
    df['barcode'] = df[df.columns[0]]
    return df


class CountMatrix:

    @classmethod
    def from_mtx(cls, countmatrixfile, regionannotation, cellannotation):
        """ Load Countmatrix from matrix market format file.

        Parameters
        ----------
        countmatrixfile : str
            Matrix market file
        regionannotation : str
            Region anntation in bed format
        cellannotation : str
            Cell anntation in tsv format

        Returns
        -------
        CountMatrix object
        """
        cmat = get_count_matrix_(countmatrixfile)
        cannot = get_cell_annotation(cellannotation, header_=True)
       
        if cmat.shape[1] != cannot.shape[0]:
            # retry without header if dims don't match
            cannot = get_cell_annotation(cellannotation, header_=False)
        
        if 'cell' not in cannot.columns:
            cannot['cell'] = cannot[cannot.columns[0]]

        rannot = get_regions_from_bed_(regionannotation)

        return cls(cmat, rannot, cannot)

    def __init__(self, countmatrix, regionannotation, cellannotation):

        if not issparse(countmatrix):
            countmatrix = csr_matrix(countmatrix)

        self.cmat = countmatrix.tocsr()
        self.cannot = cellannotation
        self.regions = regionannotation
        assert self.cmat.shape[0] == len(self.regions)
        assert self.cmat.shape[1] == len(self.cannot)

    def remove_chroms(self, chroms):
        """Remove chromsomes."""
        idx = self.regions.chrom[~self.regions.chrom.isin(chroms)].index
        self.regions = self.regions[~self.regions.chrom.isin(chroms)]
        self.cmat = self.cmat[idx]
        return self

    @property
    def counts(self):
        """
        count matrix property
        """
        return self.cmat

    def filter(self, minreadsincell=None, maxreadsincell=None,
                            minreadsinregion=None, maxreadsinregion=None,
                            binarize=True, trimcount=None):
        """
        Applies in-place quality filtering to the count matrix.

        Parameters
        ----------
        minreadsincell : int or None
            Minimum counts in cells to remove poor quality cells with too few reads.
            Default: None
        maxreadsincell : int or None
            Maximum counts in cells to remove poor quality cells with too many reads.
            Default: None
        minreadsinregion : int or None
            Minimum counts in region to remove low coverage regions.
            Default: None
        maxreadsinregion : int or None
            Maximum counts in region to remove low coverage regions.
            Default: None
        binarize : bool
            Whether to binarize the count matrix. Default: True
        trimcounts : int or None
            Whether to trim the maximum number of reads per cell and region.
            This is a generalization to the binarize option.
            Default: None (No trimming performed)

        """

        if minreadsincell is None:
            minreadsincell = 0

        if maxreadsincell is None:
            maxreadsincell = sys.maxsize

        if minreadsinregion is None:
            minreadsinregion = 0

        if maxreadsinregion is None:
            maxreadsinregion = sys.maxsize

        cmat = self.cmat.copy()
        if binarize:
            cmat.data[self.cmat.data > 0] = 1

        if trimcount is not None and trimcount > 0:
            cmat.data[self.cmat.data > trimcount] = trimcount

        cellcounts = cmat.sum(axis=0)

        keepcells = np.where((cellcounts >= minreadsincell) &
                             (cellcounts <= maxreadsincell) &
                             (self.cannot.cell.values != 'dummy'))[1]

        cmat = cmat[:, keepcells]
        cannot = self.cannot.iloc[keepcells].copy()

        regioncounts = cmat.sum(axis=1)
        keepregions = np.where((regioncounts >= minreadsinregion) &
                               (regioncounts <= maxreadsinregion))[0]

        cmat = cmat[keepregions, :]
        regions = self.regions.iloc[keepregions].copy()
        return CountMatrix(cmat, regions, cannot)

    def __getitem__(self, ireg):
        if issparse(cmat.cmat):
            return cmat.cmat[idx]
        return csr_matrix(cmat.cmat[idx])

    def __repr__(self):
        return "{} x {} CountMatrix with {} entries".format(self.cmat.shape[0], self.cmat.shape[1], self.cmat.nnz)

    @property
    def n_cells(self):
        return self.cmat.shape[1]

    @property
    def n_regions(self):
        return self.cmat.shape[0]

    @property
    def shape(self):
        return (self.n_regions, self.n_cells)

    @property
    def __len__(self):
        return self.n_regions


