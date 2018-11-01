CC = mpicc
OPTLEVEL = -O
DIR		= /usr/local/SPOOLES.2.2
MPE_DIR		= /home/opc/mpich_install
#MPE_INCDIR	= /home/apollo/hda8/mpich-1.2.5.2/mpe/include
MPE_INCDIR	= $(MPE_DIR)/include
INCLUDE_DIR	= -I$(MPE_INCDIR)
MPE_CFLAGS	=  -DMPI_LINUX -DUSE_STDARG -DHAVE_PROTOTYPES
CFLAGS = $(OPTLEVEL) -I $(DIR) -DARCH="Linux"


all: p_solver.o p_solver
	rm -f p_solver.o
MPI_INSTALL_DIR = $(MPE_DIR)
MPI_LIB_PATH	= -L$(MPI_INSTALL_DIR)/lib
MPI_LIBS 		= $(MPI_LIB_PATH) -lmpich
MPI_INCLUDE_DIR = -I$(MPI_INSTALL_DIR)/include
#Uncomment the below two lines so that log file can be created
MPE_LIBDIR    = $(MPE_DIR)/lib
#LOG_LIBS      = -L$(MPE_LIBDIR) -llmpe -lmpe
LOG_LIBS      = -L$(MPE_LIBDIR) -lm -lpthread
LIBS = \
	$(DIR)/MPI/src/spoolesMPI.a \
        $(DIR)/spooles.a -lm \
	-lpthread

p_solver: p_solver.o
	$(CC) p_solver.o -o $@ $(LIBS) $(LOG_LIBS)
