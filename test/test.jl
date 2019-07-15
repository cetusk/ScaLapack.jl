using MPI
using ScaLapack

const DEBUG = true

# problem size
const nrows = 8
const ncols = 5
const nrows_block = 2
const ncols_block = 2
const nprocrows = 2
const nproccols = 2

# finalizer
function mpi_finalizer(comm)
    MPI.Finalize()
end

function check_scalapack(A, B)

    # initialize BLACS
    rank, nprocs = ScaLapack.BLACS.pinfo()
    # initialize BLACS context
    # input_blacs_contxt = 0
    # blacs_indicator = 0
    # blacs_layout = 'R'
    # blacs_contxt = ScaLapack.BLACS.get(input_blacs_contxt, blacs_indicator)
    # blacs_contxt = ScaLapack.BLACS.gridinit(blacs_contxt, blacs_layout, nprocrows, nproccols)

    # initialize ScaLapack context
    sl_contxt = ScaLapack.sl_init(nprocrows, nproccols)
    my_nprocrows, my_nproccols, my_rowgrid, my_colgrid = ScaLapack.BLACS.gridinfo(sl_contxt)
    mxllda = ScaLapack.numroc(nrows, nrows_block, my_nprocrows, 0, nprocrows)
    mxlocc = ScaLapack.numroc(ncols, ncols_block, my_nproccols, 0, nproccols)
    if DEBUG
        if rank == 0
            print("\nScaLapack::\n")
            print("\ninitialize ScaLapack context:\n")
        end
        ScaLapack.BLACS.barrier(sl_contxt,'A')
        print("\nrank[$rank]; sl_contxt: $sl_contxt, my_size: ( $mxllda x $mxlocc )\n")
    end
    ScaLapack.BLACS.barrier(sl_contxt,'A')

    if my_nprocrows >= 0 && my_nproccols >= 0

        # get array descriptor
        zero=0
        desca = ScaLapack.descinit(nrows, ncols, nrows_block, ncols_block, zero, zero, sl_contxt, mxllda)
        descb = ScaLapack.descinit(nrows, ncols, nrows_block, ncols_block, zero, zero, sl_contxt, mxllda)
        descc = ScaLapack.descinit(nrows, nrows, nrows_block, ncols_block, zero, zero, sl_contxt, mxllda)
        if DEBUG
            if rank == 0
                print("\ninitialize array descriptor:\n")
            end
            ScaLapack.BLACS.barrier(sl_contxt,'A')
            print("\nrank[$rank];\ndesc-A = $desca\ndesc-B = $descb\ndesc-C = $descc\n")
        end
        ScaLapack.BLACS.barrier(sl_contxt,'A')

        # generate process matrix
        my_A = zeros(Float64,mxllda,mxlocc)
        my_B = zeros(Float64,mxllda,mxlocc)
        for ia::Integer=1:nrows
            for ja::Integer=1:ncols
                ScaLapack.pXelset!(my_A,ia,ja,desca,A[ia,ja])
            end
        end
        for ib::Integer=1:nrows
            for jb::Integer=1:ncols
                ScaLapack.pXelset!(my_B,ib,jb,descb,B[ib,jb])
            end
        end
        ScaLapack.BLACS.barrier(sl_contxt,'A')

        if DEBUG
            if rank == 0
                print("\ndestributed matrix local(A) and local(B):\n")
            end
            ScaLapack.BLACS.barrier(sl_contxt,'A')
            print("\nrank[$rank];\nlocal(A) = $my_A\nlocal(B) = $my_B\n")
        end
        ScaLapack.BLACS.barrier(sl_contxt,'A')

        # perform A*B
        # op(X) = X or X' or X*'
        # sub(A) = A[ia:ia+m-1,ja:ja+k-1]
        # sub(B) = B[ib:ib+k-1,jb:jb+n-1]
        # sub(C) = C[ic:ic+m-1,jc:jc+n-1]
        #        = α*op(sub(A))*op(sub(B))+β*sub(C)
        # op(sub(A)) denotes A[ia:ia+m-1,ja:ja+k-1]   if transa = 'n',
        #                    A[ia:ia+k-1,ja:ja+m-1]'  if transa = 't',
        #                    A[ia:ia+k-1,ja:ja+m-1]*' if transa = 'c',
        # op(sub(B)) denotes B[ib:ib+k-1,jb:jb+n-1]   if transb = 'n',
        #                    B[ib:ib+n-1,jb:jb+k-1]'  if transb = 't',
        #                    B[ib:ib+n-1,jb:jb+k-1]*' if transb = 'c',
        α=1.0;β=0.0;one=1;
        my_C = zeros(Float64,mxllda,mxllda)

        transa='n';transb='t';
        ScaLapack.pXgemm!(transa,transb,
                          nrows, nrows, ncols,
                          α,
                          my_A, one, one, desca,
                          my_B, one, one, descb,
                          β,
                          my_C, one, one, descc)

        if DEBUG
            if rank == 0
                print("\nlocal multiplied matix local(C):\n")
            end
            ScaLapack.BLACS.barrier(sl_contxt,'A')
            print("\nrank[$rank];\nlocal(C) = $my_C\n")
        end
        ScaLapack.BLACS.barrier(sl_contxt,'A')

        C = Matrix{Float64}(undef,nrows,nrows)
        for ic::Integer=1:nrows
            for jc::Integer=1:nrows
                C[ic, jc] = ScaLapack.pXelget('A', 'I', my_C, ic, jc, descc)
            end
        end

        if DEBUG
            if rank == 0
                print("\nglobal matix C that multiplied by A and B:\n")
            end
            ScaLapack.BLACS.barrier(sl_contxt,'A')
            print("\nrank[$rank];\nC = $C\n")
        end
        ScaLapack.BLACS.barrier(sl_contxt,'A')

    end

    ScaLapack.BLACS.barrier(sl_contxt,'A')
    tmp = ScaLapack.BLACS.gridexit(sl_contxt)

end

function main()
    # MPI initilaize
    MPI.Init()
    comm = MPI.COMM_WORLD
    finalizer(mpi_finalizer, comm)
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)
    MPI.Barrier(comm);

    # generate matrix
    A = Matrix{Float64}(undef, nrows, ncols)
    B = Matrix{Float64}(undef, nrows, ncols)
    if rank == 0
        for ia::Integer = 1 : nrows
            for ja::Integer = 1 : ncols
                A[ia, ja] = convert(Float64, ia+ja)
                B[ia, ja] = convert(Float64, ia*ja)
            end
        end
    end
    MPI.Barrier(comm)
    MPI.Bcast!(A, 0, comm)
    MPI.Bcast!(B, 0, comm)
    MPI.Barrier(comm)

    if DEBUG
        if rank == 0
            print("\ninput matrix:\n")
        end
        MPI.Barrier(comm)
        print("\nrank[$rank];\nA = \n$A\nB = \n$B\n")
    end
    MPI.Barrier(comm)

    # test
    # check_scalapack(A, B)

    params = ScaLapack.ScaLapackParams(nrows_block, ncols_block, nprocrows, nproccols)
    slm_A = ScaLapack.ScaLapackMatrix(params, A)
    slm_B = ScaLapack.ScaLapackMatrix(params, B)
    slm_C = slm_A * slm_B
    # C = ScaLapack.multiple(A, B, nrows_block, ncols_block, nprocrows, nproccols)
    if DEBUG
        if rank == 0
            print("\noutput matrix:\n")
        end
        MPI.Barrier(comm)
        print("\nrank[$rank];\nC = \n$(slm_C.X)\n")
    end

    MPI.Barrier(comm)

    # clean up
    ScaLapack.BLACS.exit()
    

end

main()

