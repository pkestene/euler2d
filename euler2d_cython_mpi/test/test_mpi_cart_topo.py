#!/usr/bin/env python
"""
mpi4py example use of cartesian topology communicator.
"""

from mpi4py import MPI

import sys
import time

import numpy as np

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

TOPO_SIZE_X=4
TOPO_SIZE_Y=3
TOPO_SIZE_Z=2

SIZE_2D = TOPO_SIZE_X*TOPO_SIZE_Y
SIZE_3D = TOPO_SIZE_X*TOPO_SIZE_Y*TOPO_SIZE_Z

# identifying neighbors
X_PLUS_1  = 0
X_MINUS_1 = 1
Y_PLUS_1  = 2
Y_MINUS_1 = 3
Z_PLUS_1  = 4
Z_MINUS_1 = 5

# allow processor reordering by the MPI cartesian communicator
MPI_REORDER_FALSE = 0
MPI_REORDER_TRUE  = 1

MPI_CART_PERIODIC_FALSE = 0
MPI_CART_PERIODIC_TRUE  = 1

# MPI topology directions
MPI_TOPO_DIR_X = 0
MPI_TOPO_DIR_Y = 1
MPI_TOPO_DIR_Z = 2

# MPI topology shift direction
MPI_SHIFT_NONE = 0
MPI_SHIFT_FORWARD = 1

N_NEIGHBORS_2D = 4
N_NEIGHBORS_3D = 6

NDIM = 2


def main():

    sys.stdout.write(
        "Hello, World! I am process %d of %d on %s.\n"
        % (rank, size, name))

    if rank == 0:
        print("Take care that MPI Cartesian Topology uses COLUMN MAJOR-FORMAT !!!");
        print("");
        print("In this test, each MPI process of the cartesian grid sends a message");
        print("containing a integer (rank of the current process) to all of its");
        print("neighbors. So you must chech that arrays \"neighbors\" and \"inbuf\"");
        print("contain the same information !");

    nbrs = [0]*2*NDIM
    nbrs2 = [0]*2*NDIM

    # 2D CARTESIAN MPI MESH
    if size == SIZE_2D:

        dims = [TOPO_SIZE_X, TOPO_SIZE_Y]
        periods = [True] * len(dims)
        
        # create the cartesian topology
        topo = MPI.COMM_WORLD.Create_cart(dims, periods=periods)

        # get rank inside the tolopogy
        rank_topo = topo.Get_rank()

        # get 2D coordinates inside topology
        coords = topo.Get_coords(rank_topo)

        # get rank of source (x-1) and destination (x+1) process
        # take care MPI uses column-major order
        nbrs[X_MINUS_1], nbrs[X_PLUS_1] = topo.Shift(MPI_TOPO_DIR_X, 1)
        
        # # get rank of source (y-1) and destination (y+1) process
        nbrs[Y_MINUS_1], nbrs[Y_PLUS_1] = topo.Shift(MPI_TOPO_DIR_Y, 1)
        
        # another way to get shifted process' rank using MPI_Cart_rank
        shiftedCoords = [ coords[0] - 1, coords[1] ]
        nbrs2[X_MINUS_1] = topo.Get_cart_rank(shiftedCoords)

        shiftedCoords = [ coords[0] + 1, coords[1] ]
        nbrs2[X_PLUS_1] = topo.Get_cart_rank(shiftedCoords)

        shiftedCoords = [ coords[0], coords[1] - 1 ]
        nbrs2[Y_MINUS_1] = topo.Get_cart_rank(shiftedCoords)

        shiftedCoords = [ coords[0], coords[1] + 1 ]
        nbrs2[Y_PLUS_1] = topo.Get_cart_rank(shiftedCoords)


        outbuf = np.array([rank], dtype=np.int)
        inbuf = np.zeros(N_NEIGHBORS_2D, dtype=np.int)

        data_recv = np.zeros( ( N_NEIGHBORS_2D, 1), dtype=np.int)
        
        # send    my rank to   each of my neighbors
        # receive my rank from each of my neighbors
        # inbuf should contain the rank of all neighbors
        # for (i=0; i<N_NEIGHBORS_2D; i++) {
        # dest = nbrs[i];
        # source = nbrs[i];
        tag = 1
        send_request = [MPI.REQUEST_NULL] * N_NEIGHBORS_2D
        recv_request = [MPI.REQUEST_NULL] * N_NEIGHBORS_2D
        for i in range(N_NEIGHBORS_2D):
            dest = nbrs[i]
            source = nbrs[i]
            send_request[i] = topo.Isend(outbuf, dest, tag)
            recv_request[i] = topo.Irecv(data_recv[i], source, tag)
        MPI.Request.Waitall(send_request)
        MPI.Request.Waitall(recv_request)
        for i in range(N_NEIGHBORS_2D):
            inbuf[i] = data_recv[i][0]
        
        print("rank= {0:2d} coords= {1} {2}  neighbors(x+,x-,y+,y-) = {3:2d} {4:2d} {5:2d} {6:2d}".format(
            rank,
            coords[0],coords[1], 
            nbrs[X_PLUS_1], 
            nbrs[X_MINUS_1], 
            nbrs[Y_PLUS_1], 
            nbrs[Y_MINUS_1]))
        print("rank= {0:2d} coords= {1} {2}  neighbors(x+,x-,y+,y-) = {3:2d} {4:2d} {5:2d} {6:2d}".format(
            rank,
            coords[0],coords[1], 
            nbrs2[X_PLUS_1], 
            nbrs2[X_MINUS_1], 
            nbrs2[Y_PLUS_1], 
            nbrs2[Y_MINUS_1]))
        print("rank= {0:2d} coords= {1} {2}  inbuf    (x+,x-,y+,y-) = {3:2d} {4:2d} {5:2d} {6:2d}".format(
            rank,
            coords[0],coords[1],
            inbuf[X_PLUS_1],
            inbuf[X_MINUS_1],
            inbuf[Y_PLUS_1],
            inbuf[Y_MINUS_1]))

        # print topology
        topo.Barrier()
        time.sleep(1)

        if rank == 0:
            print("Print topology (COLUMN MAJOR-ORDER) for {}x{} 2D grid:".format(TOPO_SIZE_X,TOPO_SIZE_Y))
            print(" rank     i     j")
        print("{0:5d} {1:5d} {2:5d} {3:10d} {4:10d} {5:10d} {6:10d}".format(rank,coords[0], coords[1], nbrs[X_PLUS_1], nbrs[X_MINUS_1], nbrs[Y_PLUS_1], nbrs[Y_MINUS_1]))


    else:
        print("Must specify {} processors. Terminating.".format(SIZE_2D))
        
    
if __name__ == '__main__':

    main()
