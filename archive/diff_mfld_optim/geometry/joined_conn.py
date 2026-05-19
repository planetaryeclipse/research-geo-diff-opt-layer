import torch
import itertools

from typing import List, Tuple

from diff_mfld_optim.geometry.connection import Connection

InsertIdxs = List[
    int
]  # list of indices to map from joining conn to larger tangent bundle conn
JoiningConn = Tuple[
    Connection,
    Tuple[
        InsertIdxs,  # insertion from uppder index
        InsertIdxs,  # insertion from first lower index
        InsertIdxs,  # insertion from second lower index
        # NOTE: last also defines the indexing to go from a point on the larger
        # manifold to the base space for this joining connection
    ],
]


class JoinedConnection(Connection):
    # joins connections defined on vector bundles that are subsets of the
    # tangent bundle on the larger manifold onto connections on this tb

    def __init__(self, n, conns: List[JoiningConn]):
        super().__init__(n, n)
        self.conns = conns

    def _eval(self, p):
        # assumes that unless otherwise assigned then there is no curvature
        # pertaining to these components and so the connection coefficients
        # are therefore zero
        full_conn_coeffs = torch.zeros(self.n, self.n, self.n)

        # first need to evaluate the connections for all subconnections and
        # then we can remap them into the connection on this tangent bundle
        for subconn, (
            upper_insert_map,
            first_lower_insert_map,
            second_lower_insert_map,
        ) in self.conns:

            p_on_subconn_base_mfld = p[second_lower_insert_map]
            conn_coeffs_subconn = subconn(p_on_subconn_base_mfld)

            # inefficient but direct indexing with dynamic arrays seems either
            # impossible or difficult to do so at the current moment
            for k, i, j in itertools.product(
                range(len(upper_insert_map)),
                range(len(first_lower_insert_map)),
                range(len(second_lower_insert_map)),
            ):
                full_conn_coeffs[
                    upper_insert_map[k],
                    first_lower_insert_map[i],
                    second_lower_insert_map[j],
                ] = conn_coeffs_subconn[k, i, j]

            # full_conn_coeffs[
            #     upper_insert_map, first_lower_insert_map, second_lower_insert_map
            # ] = conn_coeffs_subconn

        return full_conn_coeffs
