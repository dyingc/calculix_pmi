
CFLAGS = -Wall -O3 -fopenmp -I ../../../SPOOLES.2.2/ -I ../../../SPOOLES.2.2/MPI -DARCH="Linux" -DSPOOLES -DARPACK -DMATRIXSTORAGE -DUSE_MT=1 -DCALCULIX_MPI=1 -DDEBUG_LVL=200
FFLAGS = -Wall -O3 -fopenmp

CC=cc
CC=mpicc
FC=gfortran
FC=mpifort

.c.o :
	$(CC) $(CFLAGS) -c $<
.f.o :
	$(FC) $(FFLAGS) -c $<

include Makefile.inc

SCCXMAIN = ccx_2.14.c

OCCXF = $(SCCXF:.f=.o)
OCCXC = $(SCCXC:.c=.o)
OCCXMAIN = $(SCCXMAIN:.c=.o)

DIR=../../../SPOOLES.2.2

LIBS = \
       $(DIR)/MT/src/spoolesMT.a \
       $(DIR)/MPI/src/spoolesMPI.a \
       $(DIR)/spooles.a \
       ../../../ARPACK/libarpack_INTEL.a \
       -lpthread -lm

ccx_2.14_MPI: $(OCCXMAIN) ccx_2.14_MPI.a  $(LIBS)
	./date.pl; $(CC) $(CFLAGS) -c ccx_2.14.c; $(FC) -fopenmp -Wall -O3 -o $@ $(OCCXMAIN) ccx_2.14_MPI.a $(LIBS)

ccx_2.14_MPI.a: $(OCCXF) $(OCCXC)
	ar vr $@ $?
                                                                               
