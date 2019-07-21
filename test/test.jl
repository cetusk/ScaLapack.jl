using MPI
using ScaLapack
using ScaLapack: BLACS, ScaLapackLite

const DEBUG = false

# problem size
const nrows = 8
const ncols = 8
const nrows_block = 2
const ncols_block = 2
const nprocrows = 2
const nproccols = 2

# finalizer
function mpi_finalizer(comm)
    MPI.Finalize()
end

function mutiple_test(root, comm)

    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)
    MPI.Barrier(comm);

    # generate matrix
    if rank == 0
        A = Matrix{Float64}(undef, nrows, ncols)
        B = Matrix{Float64}(undef, nrows, ncols)
        for ia::Integer = 1 : nrows
            for ja::Integer = 1 : ncols
                A[ia, ja] = convert(Float64, ia+ja)
                B[ia, ja] = convert(Float64, ia*ja)
            end
        end
    else
        A = Matrix{Float64}(undef, 0, 0)
        B = Matrix{Float64}(undef, 0, 0)
    end
    MPI.Barrier(comm)

    if DEBUG
        if rank == root
            print("\ninput matrix:\n")
        end
        MPI.Barrier(comm)
        print("\nrank[$rank];\nA = \n$A\nB = \n$B\n")
    end
    MPI.Barrier(comm)

    # prepare ScaLapackLite params
    params = ScaLapackLite.ScaLapackLiteParams(nrows_block, ncols_block, nprocrows, nproccols, root)
    slm_A = ScaLapackLite.ScaLapackLiteMatrix(params, A)
    slm_B = ScaLapackLite.ScaLapackLiteMatrix(params, B)

    # perform C = A*B
    slm_C = slm_A * slm_B'
    if DEBUG
        if rank == root
            print("\noutput matrix:\n")
        end
        MPI.Barrier(comm)
        print("\nrank[$rank];\nC = \n$(slm_C.X)\n")
    end

    MPI.Barrier(comm)
    # clean up
    BLACS.exit()

end

function hessenberg_test(root, comm)
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)
    MPI.Barrier(comm);

    A = Matrix{Float64}(undef, nrows, ncols)
    for ia::Integer = 1 : nrows
        for ja::Integer = 1 : ncols
            A[ia, ja] = convert(Float64, ia+ja)
        end
    end

    params = ScaLapackLite.ScaLapackLiteParams(nrows_block, ncols_block, nprocrows, nproccols, root)
    slm_A = ScaLapackLite.ScaLapackLiteMatrix(params, A)

    myA = ScaLapackLite.hessenberg(slm_A)

    print("[$rank] local(A): $myA\n")

end

function main()
    # MPI initilaize
    MPI.Init()
    comm = MPI.COMM_WORLD
    finalizer(mpi_finalizer, comm)
    # test
    # mutiple_test(0, comm)
    hessenberg_test(0, comm)
end

main()
