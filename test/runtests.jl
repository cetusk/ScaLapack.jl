using ScaLapack
using MPI

MPI.Init()

# Base.disable_threaded_libs()

# problem size
na_rows=2;na_cols=3;
nb_rows=na_cols;nb_cols=1;
blocksize=2

A = randn(na_rows, na_cols)
B = randn(nb_rows, nb_cols)

print("matrix A: $A\n")
print("matrix B: $B\n")
print("matrix C: $C\n")

# initialize grid
id,nprocs = ScaLapack.BLACS.pinfo()

print("id: $id, nprocs: $nprocs\n")

# ic = ScaLapack.sl_init(trunc(Integer, sqrt(nprocs)), div(nprocs, trunc(Integer, sqrt(nprocs))))
ictxt = ScaLapack.sl_init(Int64(sqrt(nprocs)), Int64(div(nprocs, sqrt(nprocs))))

@sync for p in MPI.workers()
    # initialize grid
    @spawnat p begin

        # who am I?
        nprow, npcol, myrow, mycol = ScaLapack.BLACS.gridinfo(ictxt)
        my_na_rows = ScaLapack.numroc(na_rows, blocksize, myrow, 0, nprow)
        my_na_cols = ScaLapack.numroc(na_cols, blocksize, mycol, 0, npcol)
        my_nb_rows = ScaLapack.numroc(nb_rows, blocksize, myrow, 0, nprow)
        my_nb_cols = ScaLapack.numroc(nb_cols, blocksize, mycol, 0, npcol)
        print("myrow: $myrow, mycol: $mycol,
            blocksize: $blocksize,
            pna_rows: $pna_rows, pna_cols: $pna_cols,
            pnb_rows: $pnb_rows, pnb_cols: $pnb_cols\n")

        

        lld=9
        if nprow >= 0 && npcol >= 0

            print("check: A\n")

            # Get DArray info
            descA = ScaLapack.descinit(pna_rows, pna_cols, blocksize, blocksize, 0, 0, ictxt, lld)
            descB = ScaLapack.descinit(pnb_rows, pnb_cols, blocksize, blocksize, 0, 0, ictxt, lld)
            descC = ScaLapack.descinit(pna_rows, pnb_cols, blocksize, blocksize, 0, 0, ictxt, lld)


            print("check: B\n")


            # allocate local array
            # A = float32(randn(Int(np), Int(nq)))
            # A = complex(randn(Int(np), Int(nq)), randn(Int(np), Int(nq)))
            # A = complex64(complex(randn(Int(np), Int(nq)), randn(Int(np), Int(nq))))

            # calculate DGEMM
            tpA='N';tpB='N';
            m=pna_rows;n=pnb_cols;k=pna_cols;
            ia=1;ja=1;ib=1;jb=1;ic=1;jc=1;

            idAx=ia+m-1;idAy=ja+k-1;
            if tpA == 'T'
                idAx=ia+k-1;idAy=ja+m-1;
            end
            idBx=ib+k-1;idBy=jb+n-1;
            if tpB == 'T'
                idBx=ib+n-1;idBy=jb+k-1;
            end
            idCx=ic+m-1;idCy=jc+n-1


            alpha=1.0;beta=1.0;
            CC = ScaLapack.pdgemm!('N','N',
                                        m,n,k,alpha,
                                        A,idAx,idAy,descA,
                                        B,idBx,idBy,descB,
                                        beta,
                                        C,idCx,idCy,descC)

            print("check: C")


            # show result
            if myrow == 0 && mycol == 0
                println(CC)
            end

            # clean up
            tmp = ScaLapack.BLACS.gridexit(ictxt)

end
ScaLapack.BLACS.exit()

MPI.Finalize()