/* This program is free software; you can redistribute it and/or    */
/* modify it under the terms of the GNU General Public License as   */
/* published by the Free Software Foundation(version 2);      */
/*                       */
/* This program is distributed in the hope that it will be useful */ /* but WITHOUT ANY WARRANTY; without even the implied warranty of */ /* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the */ /* GNU General Public License for more details. */
/*                       */
/* You should have received a copy of the GNU General Public */ /* License along with this program; if not, write to the Free */
/* Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, */
/* USA. */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <misc.h>
#include <FrontMtx.h>
#include <SymbFac.h>
#include "/usr/local/Calculix/ccx_2.14/src/CalculiX.h"
#include "/usr/local/SPOOLES.2.2/MPI/spoolesMPI.h"
#include "/usr/local/SPOOLES.2.2/SPOOLES.h"
#include "/usr/local/SPOOLES.2.2/timings.h"

int main(int argc, char *argv[]) {
    char buffer[20];
    DenseMtx *mtxX, *mtxB, *newB;
    Chv *rootchv;
    ChvManager *chvmanager;
    SubMtxManager *mtxmanager, *solvemanager;
    FrontMtx *frontmtx;
    InpMtx *mtxA, *newA;
    double cutoff, droptol = 0.0, minops, tau = 100.;
    double cpus[20];
    double *opcounts;
    DV *cumopsDV;
    ETree *frontETree;
    FILE *inputFile, *msgFile, *densematrix, *inputmatrix;
    Graph *graph;
    int jcol, jrow, error, firsttag, ncol, lookahead = 0, msglvl = 0, nedges, nent, neqns, nmycol, nrhs, ient, iroow, nrow, pivotingflag = 0, root, seed = 7892713, symmetryflag = 0, type = 1;
    int stats[20];
    int *rowind;
    IV *newToOldIV, *oldToNewIV, *ownedColumnsIV, *ownersIV, *vtxmapIV;
    IVL *adjIVL, *symbfacIVL;
    SolveMap *solvemap;
    double value;


    int myid, nproc;
    int namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    double starttime = 0.0, endtime;
    int maxdomainsize, maxsize, maxzeros;
    /* Solving the system of equations using Spooles */
    /*----------------------------------------------------------------*/
    /*
       -----------------------------------------------------------------
       Find out the identity of this process and the number of processes
       -----------------------------------------------------------------
     * edong: re-coded into spooles_factor, inside #ifdef PMI_READY
     */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Get_processor_name(processor_name, &namelen);
    if (myid == 0) {
        printf("Solving the system of equations using SpoolesMPI\n\n");
    }
    fprintf(stdout, "Process %d of %d on %s\n", myid + 1, nproc,
            processor_name);
    /* Start a timer to determine how long the solve process takes */
    starttime = MPI_Wtime();
    /*----------------------------------------------------------------*/

    sprintf(buffer, "res.%d", myid);
    if ((msgFile = fopen(buffer, "w")) == NULL) {
        fprintf(stderr, "\n fatal error in spooles.c"
                "\n unable to open file res\n");
    }
    /*
       --------------------------------------------
       STEP 1: Read the entries from the input file
               and create the InpMtx object
       --------------------------------------------
    // edong: It's completed in spooles_factor function in spooles.c
    // edong: Now it's recoded in function mtxA_propagate
     */
    /* Read in the input matrix, A */
    sprintf(buffer, "matrix.%d.input", myid);
    inputFile = fopen(buffer, "r");
    fprintf(stdout, "edong: buffer = %s\n", buffer);
    fscanf(inputFile, "%d %d %d", &neqns, &ncol, &nent);
    printf("edong: Row = %d, Col = %d, Ent = %d\n", neqns, ncol, nent);
    nrow = neqns;
    MPI_Barrier(MPI_COMM_WORLD);
    mtxA = InpMtx_new();
    InpMtx_init(mtxA, INPMTX_BY_ROWS, type, nent, 0);
    for (ient = 0; ient < nent; ient++) {
        fscanf(inputFile, "%d %d %lf", &iroow, &jcol, &value);
    printf("edong: Cell (%d,%d) = %lf\n", iroow, jcol, value);
        InpMtx_inputRealEntry(mtxA, iroow, jcol, value);
    }
    fclose(inputFile);
    printf("edong: Message file is: %s\n", buffer);
    /* Change the storage mode to vectors */
    InpMtx_sortAndCompress(mtxA);
    InpMtx_changeStorageMode(mtxA, INPMTX_BY_VECTORS);
    if (msglvl > 1) {
        fprintf(msgFile, "\n\n input matrix");
        InpMtx_writeForHumanEye(mtxA, msgFile);
        fflush(msgFile);
    }
    /*----------------------------------------------------------------*/
    /*
       ----------------------------------------------------
       STEP 2: Read the right hand side entries from the
       input file and create the DenseMtx object for B
       ----------------------------------------------------
    // edong: It's completed in spooles_solve function in spooles.c
     */
    sprintf(buffer, "rhs.%d.input", myid);
    inputFile = fopen(buffer, "r");

    fscanf(inputFile, "%d %d", &nrow, &nrhs);
    printf("edong: Row = %d, Type = %s\n", nrow, (nrhs==1)?"real":"complex");
    mtxB = DenseMtx_new();
    DenseMtx_init(mtxB, type, 0, 0, nrow, nrhs, 1, nrow);
    DenseMtx_rowIndices(mtxB, &nrow, &rowind);
    printf("\tedong: nrow = %d, rowind = %d\n", nrow, *rowind);
    for (iroow = 0; iroow < nrow; iroow++) {
        fscanf(inputFile, "%d", rowind + iroow);
    printf("\tedong: iroow = %d, rowind = %d\n", iroow, *rowind);
        for (jcol = 0; jcol < nrhs; jcol++) {
            fscanf(inputFile, "%lf", &value);
    printf("\t\tedong: value=%lf\n", value);
            DenseMtx_setRealEntry(mtxB, iroow, jcol, value);
        }
    }
    fclose(inputFile);
    if (msglvl > 1) {
        fprintf(msgFile, "\n\n rhs matrix in original ordering");
        DenseMtx_writeForHumanEye(mtxB, msgFile);
        fflush(msgFile);
    }
    printf("edong: walked here");
    /*----------------------------------------------------------------*/
    /*
       -------------------------------------------------------
       STEP 3 : Find a low-fill ordering
       (1) Processor 0 creates the Graph object
       (2) Processor 0 orders the graph using the better of
       Nested Dissection and Multisection
       (3) Optimal front matrix paremeters are chosen depending
       on the number of processors
       (4) Broadcast ordering to the other processors
       -------------------------------------------------------
    // edong: It's completed in ssolve_creategraph function in spooles.c
     */
    if (myid == 0) {
        graph = Graph_new();
        adjIVL = InpMtx_fullAdjacency(mtxA);
        nedges = IVL_tsize(adjIVL);
        Graph_init2(graph, 0, neqns, 0, nedges, neqns, nedges, adjIVL,
                NULL, NULL);
        if (msglvl > 1) {
            fprintf(msgFile, "\n\n graph of the input matrix");
            Graph_writeForHumanEye(graph, msgFile);
            fflush(msgFile);
        }
        /* Below choose the optimized values for maxdomainsize, */
        /* maxzeros, and maxsize depending on the number of     */

        /* processors. */
        if (nproc == 2) {
            maxdomainsize = 700;
            maxzeros = 1000;
            maxsize = 96;
        } else if (nproc == 3) {
            maxdomainsize = 900;
            maxzeros = 1000;
            maxsize = 64;
        } else {
            maxdomainsize = 900;
            maxzeros = 1000;
            maxsize = 80;
        }
        /* Perform an ordering with the better of nested dissection and */
        /* multi-section.  */
        frontETree = orderViaBestOfNDandMS(graph, maxdomainsize, maxzeros,
                maxsize, seed, msglvl, msgFile);
        Graph_free(graph);
    } else {
    }
    /* The ordering is now sent to all processors with MPI_Bcast. */
    frontETree = ETree_MPI_Bcast(frontETree, root,
            msglvl, msgFile, MPI_COMM_WORLD);
    /*----------------------------------------------------------------*/
    /*
       -------------------------------------------------------
       STEP 4: Get the permutations, permute the front tree,
               permute the matrix and right hand side.
       -------------------------------------------------------
    // edong: It's completed in ssolve_permuteA & ssolve_permuteB (last statement for mtxB) function in spooles.c
    // edong: the last statement of ssolve_permuteA (SymbFac_initFromInpMtx) will be finished by its MPI version
    // edong: in STEP 7
     */
    /* Very similar to the serial code */
    oldToNewIV = ETree_oldToNewVtxPerm(frontETree);
    newToOldIV = ETree_newToOldVtxPerm(frontETree);
    ETree_permuteVertices(frontETree, oldToNewIV);
    InpMtx_permute(mtxA, IV_entries(oldToNewIV),
            IV_entries(oldToNewIV));
    InpMtx_mapToUpperTriangle(mtxA);
    InpMtx_changeCoordType(mtxA, INPMTX_BY_CHEVRONS);
    InpMtx_changeStorageMode(mtxA, INPMTX_BY_VECTORS);
    DenseMtx_permuteRows(mtxB, oldToNewIV);

    /*---------------------------------------------------------------*/
    /*
       -------------------------------------------
       STEP 5: Generate the owners map IV object
               and the map from vertices to owners
       -------------------------------------------
     */
    // edong: We can see some similar code in the first part of STEP 3 in factor_MT (NOT the serial factor) in spooles.c, 
    // edong: to get the ownersIV
    // edong: The second part, about vtxmapIV, should be used by STEP 7 which is an MPI dedicated one
    /* This is all new from the serial code:               */
    /* Obtains map from fronts to processors.  Also a map  */
    /* from vertices to processors is created that enables */
    /* the matrix A and right hand side B to be distributed*/
    /* as necessary. */
    cutoff = 1. / (2 * nproc);
    cumopsDV = DV_new();
    DV_init(cumopsDV, nproc, NULL);
    ownersIV = ETree_ddMap(frontETree,
            type, symmetryflag, cumopsDV, cutoff);
    DV_free(cumopsDV);
    vtxmapIV = IV_new();
    IV_init(vtxmapIV, neqns, NULL);
    IVgather(neqns, IV_entries(vtxmapIV),
            IV_entries(ownersIV), ETree_vtxToFront(frontETree));
    if (msglvl > 1) {
        fprintf(msgFile, "\n\n map from fronts to owning processes");
        IV_writeForHumanEye(ownersIV, msgFile);
        fprintf(msgFile, "\n\n map from vertices to owning processes");
        IV_writeForHumanEye(vtxmapIV, msgFile);
        fflush(msgFile);
    }
    /*---------------------------------------------------------------*/
    /*
       ---------------------------------------------------
       STEP 6: Redistribute the matrix and right hand side
       ---------------------------------------------------
    // edong: This should be dedicated to MPI version.
     */
    /* Now the entries of A and B are assembled and distributed */
    firsttag = 0;
    newA = InpMtx_MPI_split(mtxA, vtxmapIV, stats,
            msglvl, msgFile, firsttag, MPI_COMM_WORLD);
    firsttag++;
    InpMtx_free(mtxA);
    mtxA = newA;
    InpMtx_changeStorageMode(mtxA, INPMTX_BY_VECTORS);
    if (msglvl > 1) {
        fprintf(msgFile, "\n\n split InpMtx");
        InpMtx_writeForHumanEye(mtxA, msgFile);
        fflush(msgFile);
    }
    newB = DenseMtx_MPI_splitByRows(mtxB, vtxmapIV, stats, msglvl,
            msgFile, firsttag, MPI_COMM_WORLD);
    DenseMtx_free(mtxB);
    mtxB = newB;
    firsttag += nproc;
    if (msglvl > 1) {
        fprintf(msgFile, "\n\n split DenseMtx B");
        DenseMtx_writeForHumanEye(mtxB, msgFile);
        fflush(msgFile);
    }
    /*---------------------------------------------------------------*/
    /*
       ------------------------------------------
       STEP 7: Compute the symbolic factorization
       ------------------------------------------
    // edong: It's completed in ssolve_permuteA (last statement for mtxA) function in spooles.c
     */
    symbfacIVL = SymbFac_MPI_initFromInpMtx(frontETree, ownersIV, mtxA,
            stats, msglvl, msgFile, firsttag, MPI_COMM_WORLD);
    firsttag += frontETree->nfront;
    if (msglvl > 1) {
        fprintf(msgFile, "\n\n local symbolic factorization");
        IVL_writeForHumanEye(symbfacIVL, msgFile);
        fflush(msgFile);
    }
    /*---------------------------------------------------------------*/
    /*
       -----------------------------------
       STEP 8: initialize the front matrix
       -----------------------------------
    // edong: We can find the similar part in STEP 3 of factor
    // edong: or STEP 4 of factor_MT in spooles.c
     */
    /* Very similar to the serial code.  The arguments, myid and      */
    /* ownersIV tell the front matrix object to initialize only those */
    /* parts of the factor matrices that it owns                      */
    mtxmanager = SubMtxManager_new();
    SubMtxManager_init(mtxmanager, NO_LOCK, 0);
    frontmtx = FrontMtx_new();
    FrontMtx_init(frontmtx, frontETree, symbfacIVL, type, symmetryflag,
            FRONTMTX_DENSE_FRONTS, pivotingflag, NO_LOCK, myid,
            ownersIV, mtxmanager, msglvl, msgFile);
    /*---------------------------------------------------------------*/
    /*
       ---------------------------------
       STEP 9: Compute the factorization
       ---------------------------------
     */
    // edong: We can find the similar part in STEP 4 of factor
    // edong: or STEP 5 of factor_MT in spooles.c
    /* Similar to the serial code */
    chvmanager = ChvManager_new();
    /* For the serial code, the 0 is replaced by a 1 */
    ChvManager_init(chvmanager, NO_LOCK, 0);
    rootchv = FrontMtx_MPI_factorInpMtx(frontmtx, mtxA, tau, droptol,
            chvmanager, ownersIV, lookahead, &error, cpus,
            stats, msglvl, msgFile, firsttag,
            MPI_COMM_WORLD);
    ChvManager_free(chvmanager);
    firsttag += 3 * frontETree->nfront + 2;
    if (msglvl > 1) {
        fprintf(msgFile, "\n\n numeric factorization");
        FrontMtx_writeForHumanEye(frontmtx, msgFile);
        fflush(msgFile);
    }
    if (error >= 0) {
        fprintf(stderr,
                "\n proc %d : factorization error at front %d", myid, error);
        MPI_Finalize();
        exit(-1);
    }
    /*---------------------------------------------------------------*/
    /*
       ------------------------------------------------
       STEP 10: Post-process the factorization and split
       the factor matrices into submatrices
       ------------------------------------------------
     */
    // edong: We can find the similar part in STEP 5 of factor (call ssolve_postfactor)
    // edong: or STEP 6 of factor_MT (call ssolve_postfactor) in spooles.c
    // edong: The difference is we use the MPI version of post process
    // edong: FrontMtx_MPI_postProcess vs. FrontMtx_postProcess
    /* Very similar to the serial code */
    FrontMtx_MPI_postProcess(frontmtx, ownersIV, stats, msglvl,
            msgFile, firsttag, MPI_COMM_WORLD);
    firsttag += 5 * nproc;
    if (msglvl > 1) {
        fprintf(msgFile, "\n\n numeric factorization after post-processing");
        FrontMtx_writeForHumanEye(frontmtx, msgFile);
        fflush(msgFile);
    }
    /*---------------------------------------------------------------*/
    /*
       -----------------------------------
       STEP 11: Create the solve map object
       -----------------------------------
    // edong: In STEP 7 of factor_MT in spooles.c
    // edong: Somehow I can't find related steps in the serial version factor
     */
    solvemap = SolveMap_new(); SolveMap_ddMap(solvemap, frontmtx->symmetryflag,
            FrontMtx_upperBlockIVL(frontmtx),
            FrontMtx_lowerBlockIVL(frontmtx),
            nproc, ownersIV, FrontMtx_frontTree(frontmtx),
            seed, msglvl, msgFile);
    if (msglvl > 1) {
        SolveMap_writeForHumanEye(solvemap, msgFile);
        fflush(msgFile);
    }
    /*---------------------------------------------------------------*/
    /*
       ----------------------------------------------------
       STEP 12: Redistribute the submatrices of the factors
       ----------------------------------------------------
    // edong: dedicated for MPI version
     */
    /* Now submatrices that a processor owns are local to
       that processor */
    FrontMtx_MPI_split(frontmtx, solvemap,
            stats, msglvl, msgFile, firsttag, MPI_COMM_WORLD);
    if (msglvl > 1) {
        fprintf(msgFile, "\n\n numeric factorization after split");
        FrontMtx_writeForHumanEye(frontmtx, msgFile);
        fflush(msgFile);
    }
    /*---------------------------------------------------------------*/
    /*
       ------------------------------------------
       STEP 13: Create a solution DenseMtx object
       ------------------------------------------
    // edong: This step and the next one: STEP 9 of fsolve_MT or STEP 7 of fsolve in spooles.c
    // edong: Have defined fsolve_MPI to replace fsolve_MT or fsolve which is invoked from spooles_solve or spooles_solve_rad (both have been updated)Have defined fsolve_MPI to replace fsolve_MT or fsolve which is invoked from spooles_solve or spooles_solve_rad (both have been updated)
     */
    ownedColumnsIV = FrontMtx_ownedColumnsIV(frontmtx, myid, ownersIV,
            msglvl, msgFile);
    nmycol = IV_size(ownedColumnsIV);
    mtxX = DenseMtx_new();
    if (nmycol > 0) {
        DenseMtx_init(mtxX, type, 0, 0, nmycol, nrhs, 1, nmycol);
        DenseMtx_rowIndices(mtxX, &nrow, &rowind);
        IVcopy(nmycol, rowind, IV_entries(ownedColumnsIV));
    }
    /*---------------------------------------------------------------*/
    /*
       --------------------------------
       STEP 14: Solve the linear system
       --------------------------------
    // edong: This step and the previous one: STEP 9 of fsolve_MT or STEP 7 of fsolve in spooles.c
    // edong: the solvemanager is got from STEP 3 of factor or STEP 4 of factor_MT
    // edong: Have defined fsolve_MPI to replace fsolve_MT or fsolve which is invoked from spooles_solve or spooles_solve_rad (both have been updated)Have defined fsolve_MPI to replace fsolve_MT or fsolve which is invoked from spooles_solve or spooles_solve_rad (both have been updated)
     */
    /* Very similar to the serial code */
    solvemanager = SubMtxManager_new();
    SubMtxManager_init(solvemanager, NO_LOCK, 0);
    FrontMtx_MPI_solve(frontmtx, mtxX, mtxB, solvemanager, solvemap, cpus,
            stats, msglvl, msgFile, firsttag, MPI_COMM_WORLD);
    SubMtxManager_free(solvemanager);
    if (msglvl > 1) {
        fprintf(msgFile, "\n solution in new ordering");
        DenseMtx_writeForHumanEye(mtxX, msgFile);
    }
    /*---------------------------------------------------------------*/
    /*
       --------------------------------------------------------
       STEP 15: Permute the solution into the original ordering
                and assemble the solution onto processor 0
       --------------------------------------------------------
    // edong: STEP 8 in fsolve or STEP 10 in fsolve_MT in spooles.c
    // edong: STEP 6 of fsolve (or STEP 8 of fsolve_MT) can be found in MPI version STEP 4 (the last statement about mtxB)
    // edong: Have defined fsolve_MPI to replace fsolve_MT or fsolve which is invoked from spooles_solve or spooles_solve_rad (both have been updated)Have defined fsolve_MPI to replace fsolve_MT or fsolve which is invoked from spooles_solve or spooles_solve_rad (both have been updated)
     */
    DenseMtx_permuteRows(mtxX, newToOldIV);
    if (msglvl > 1) {
        fprintf(msgFile, "\n\n solution in old ordering");
        DenseMtx_writeForHumanEye(mtxX, msgFile);
        fflush(msgFile);
    }
    // edong: Dedicated for MPI since here
    IV_fill(vtxmapIV, 0);
    firsttag++;
    mtxX = DenseMtx_MPI_splitByRows(mtxX, vtxmapIV, stats, msglvl,
            msgFile, firsttag, MPI_COMM_WORLD);
    /* End the timer */
    endtime = MPI_Wtime();
    /* Determine how long the solve operation took */
    fprintf(stdout, "Total time for %s: %f\n", processor_name,
            endtime - starttime);
    /* Now gather the solution the processor 0 */
    if (myid == 0) {
        printf("%d\n", nrow);
        sprintf(buffer, "x.result");
        inputFile = fopen(buffer, "w");
        for (jrow = 0; jrow < ncol; jrow++) {
            fprintf(inputFile, "%1.5e\n", DenseMtx_entries(mtxX)[jrow]);
        }
        fclose(inputFile);
    }
    /*----------------------------------------------------------------*/
    /* End the MPI environment */
    MPI_Finalize();
    /* Free up memory */
    InpMtx_free(mtxA);
    DenseMtx_free(mtxB);
    FrontMtx_free(frontmtx);
    DenseMtx_free(mtxX);

    IV_free(newToOldIV);
    IV_free(oldToNewIV);
    ETree_free(frontETree);
    IVL_free(symbfacIVL);
    SubMtxManager_free(mtxmanager);
    return (0);
}
