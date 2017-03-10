#!/usr/bin/env python
"""
SIMLR_runner.py

Reads processed single-cell RNA-seq data from stdin in format where rows are
genes, and columns are cells and runs SIMLR. Requires that
https://github.com/bowang87/SIMLR_PY is installed. Assumes there are header
lines.

Script based on
https://github.com/bowang87/SIMLR_PY/blob/master/tests/test_largescale.py
"""
import os
import sys
import scipy.io as sio
sys.path.insert(0,os.path.abspath('..'))
import time
import numpy as np
import SIMLR
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics.cluster import adjusted_rand_score as ari
from scipy.sparse import csr_matrix
import argparse
import errno
from collections import defaultdict

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, 
                formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--k-min', type=int, required=False,
        default=1,
        help='min cluster value to run SIMLR on')
    parser.add_argument('--k-max', type=int, required=False,
        default=10,
        help='max cluster value to run SIMLR on')
    parser.add_argument('--nn', type=int, required=False,
        default=30,
        help='number of nearest neighbors to use in SIMLR algo')
    parser.add_argument('--save-memory', action='store_const', const=True,
        default=False,
        help='turns on memory-saving mode in SIMLR')
    parser.add_argument('--output', type=str, required=True,
        help='where to write output')
    args = parser.parse_args()
    # Create output dir
    try:
        os.makedirs(args.output)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    # Load data
    print >>sys.stderr, 'Reading input data...'
    cells_by_genes = [
            map(float, line.strip().split('\t')[1:])
            for line in sys.stdin.readlines()[1:]
        ]
    cells_by_genes = csr_matrix(zip(*cells_by_genes))
    cells_by_genes.data = np.log10(
                1 + cells_by_genes.data
            )
    print >>sys.stderr, 'Cells by genes matrix has dimensions {}.'.format(
            cells_by_genes.shape
        )
    with open(os.path.join(args.output, 'nmi_ari.tsv'), 'w') as kmeans_stream:
        print >>nmi_stream, 'cluster size\tkmeans objective function value'
        for cluster_size in xrange(args.k_min, args.k_max + 1):
            # Obtain log transform of gene counts to make data Gaussian
            print >>sys.stderr, "Running PCA for cluster size {}.".format(
                                                                cluster_size
                                                            )
            if cells_by_genes.shape[1] > 500:
                cells_by_genes = SIMLR.helper.fast_pca(cells_by_genes, 500)
            else:
                cells_by_genes = cells_by_genes.todense()
            print >>sys.stderr, "Running SIMLR for cluster size {}.".format(
                                                                cluster_size
                                                            )
            simlr = SIMLR.SIMLR_LARGE(cluster_size, args.nn,
                                        1 if args.save_memory else 0)
            S, F, val, ind = simlr.fit(cells_by_genes)
            y_pred = simlr.fast_minibatch_kmeans(F, cluster_size)
            clusters = defaultdict(set)
            for i, label in enumerate(y_pred):
                clusters[label].add(i)
            centroids = {}
            for label in clusters:
                transposed_points = zip(
                        *[point for i, point in enumerate(F)
                            if i in clusters[label]]
                    )
                centroids[label] = [
                        sum(component) for component in transposed_points
                    ]
                length = len(centroids[label])
                centroids[label] = [float(component) / length
                                        for component in centroids[label]]
            kmeans_objective = 0.0
            for i, label in enumerate(y_pred):
                kmeans_objective += sum([(F[i][j] - centroids[label][j])**2
                                            for j in xrange(len(F[i]))])
            print >>kmeans_stream, '{}\t{}'.format(
                    cluster_size, kmeans_objective
                )
