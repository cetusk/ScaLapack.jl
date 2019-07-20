
# abstract ScaLapackLite type
abstract type AbstractScaLapackLite end

# ScaLapackLite parameters
mutable struct ScaLapackLiteParams <: AbstractScaLapackLite
    mblocks::ScaInt; nblocks::ScaInt;
    mprocs::ScaInt; nprocs::ScaInt;
    rootproc::ScaInt
    # Constructor
    ScaLapackLiteParams() = new(1, 1, 1, 1, 0)
    ScaLapackLiteParams(mblocks, nblocks, mprocs, nprocs, rootproc) = new(mblocks, nblocks, mprocs, nprocs, rootproc)
end
function Base.:(==)(A::ScaLapackLiteParams, B::ScaLapackLiteParams)
    is_equal_blocks = (A.mblocks == B.mblocks) && (A.nblocks == B.nblocks)
    if !is_equal_blocks
        error("different block sizes: A and B")
    end
    is_equal_procs = (A.mprocs == B.mprocs) && (A.nprocs == B.nprocs)
    if !is_equal_procs
        error("different process grids: A and B")
    end
    is_equal_rootproc = (A.rootproc == B.rootproc)
    return is_equal_blocks && is_equal_procs && is_equal_rootproc
end

# ScaLapackLiteMatrix type
mutable struct ScaLapackLiteMatrix <: AbstractScaLapackLite
    params::ScaLapackLiteParams
    X::Matrix{Float64}
    ScaLapackLiteMatrix() = new(ScaLapackLiteParams())
    ScaLapackLiteMatrix(params) = new(params)
    ScaLapackLiteMatrix(params, X) = new(params, X)
end

# perform A'
function Base.:(adjoint)(A::ScaLapackLiteMatrix)
    return ScaLapackLiteMatrix(A.params, A.X') 
end

# perform A*B
function Base.:(*)(A::ScaLapackLiteMatrix, B::ScaLapackLiteMatrix)
    # check A and B have same params
    if A.params == B.params
        C = multiple(A.X, B.X, A.params)
        return ScaLapackLiteMatrix(A.params, C)
    else
        throw(DimensionMismatch("ScaLapackLiteMatrix A and B have different ScaLapaclLiteParams"))
    end
end
for (elty) in (:Float32, :Float64, :ComplexF32, :ComplexF64)
    @eval begin
        function multiple(A::Matrix{$elty}, B::Matrix{$elty}, params::ScaLapackLiteParams)

            # MPI parameters
            mblocks = params.mblocks
            nblocks = params.nblocks
            mprocs = params.mprocs
            nprocs = params.nprocs
            rootproc = params.rootproc
            comm = MPI.COMM_WORLD
            rank = MPI.Comm_rank(comm)
            zero = 0; one = 1;

            # copy matrix from master to slave
            A = MPI.bcast(A, rootproc, comm)
            B = MPI.bcast(B, rootproc, comm)
            MPI.Barrier(comm)

            # check both dimensions; cols(A) = rows(B)
            # (mA, nA), (mB, nB) = (mA, nA), (nA, nB)
            if rank == rootproc
                check_multiple(A, B)
            end

            # create context
            ctxt = ScaLapack.sl_init(mprocs, nprocs)
            ctxt0 = ScaLapack.sl_init(mprocs, nprocs)

            # matrix parameters
            mA, nA = size(A)
            mB, nB = size(B)
            mproc, nproc, myrow, mycol = BLACS.gridinfo(ctxt)
            mxlocr_A = ScaLapack.numroc(mA, mblocks, mproc, 0, mprocs)
            mxlocr_B = ScaLapack.numroc(mB, mblocks, mproc, 0, mprocs)
            mxlocc_A = ScaLapack.numroc(nA, nblocks, nproc, 0, nprocs)
            mxlocc_B = ScaLapack.numroc(nB, nblocks, nproc, 0, nprocs)
            mxllda_A = max(1, mxlocr_A)
            mxllda_B = max(1, mxlocr_B)

            if mproc >= 0 && nproc >= 0
                # get array descriptor
                # (mA, nA) x (mB, nB) = (mA, nA) x (nA, nB) = (mA, nB)
                # (mA, nB) --> (mxlocr_A, mxlocc_B) --> mxllda_A
                desc_A = ScaLapack.descinit(mA, nA, mA, nA, zero, zero, ctxt0, mA)
                desc_B = ScaLapack.descinit(mB, nB, mB, nB, zero, zero, ctxt0, mB)
                desc_C = ScaLapack.descinit(mA, nB, mA, nB, zero, zero, ctxt0, mA)
                desc_myA = ScaLapack.descinit(mA, nA, mblocks, nblocks, zero, zero, ctxt, mxllda_A)
                desc_myB = ScaLapack.descinit(mB, nB, mblocks, nblocks, zero, zero, ctxt, mxllda_B)
                desc_myC = ScaLapack.descinit(mA, nB, mblocks, nblocks, zero, zero, ctxt, mxllda_A)

                # generate process matrix
                myA = zeros($elty, mxlocr_A, mxlocc_A)
                myB = zeros($elty, mxlocr_B, mxlocc_B)
                ScaLapack.pXgemr2d!(mA, nA,
                                    A, one, one, desc_A,
                                    myA, one, one, desc_myA,
                                    ctxt)
                ScaLapack.pXgemr2d!(mB, nB,
                                    B, one, one, desc_B,
                                    myB, one, one, desc_myB,
                                    ctxt)

                # perform A*B
                α = 1.0; β = 0.0;
                myC = zeros($elty, mxlocr_A, mxlocc_B)
                transa = 'n'; transb = 'n';
                ScaLapack.pXgemm!(transa, transb,
                                  mA, nB, nA,
                                  α,
                                  myA, one, one, desc_myA,
                                  myB, one, one, desc_myB,
                                  β,
                                  myC, one, one, desc_myC)

                C = Matrix{$elty}(undef, mA, nB)
                ScaLapack.pXgemr2d!(mA, nB,
                                    myC, one, one, desc_myC,
                                    C, one, one, desc_C,
                                    ctxt)

            end
            
            BLACS.barrier(ctxt, 'A')
            BLACS.gridexit(ctxt)

            # free
            if rank != rootproc
                A = Matrix{$elty}(undef, 0, 0)
                B = Matrix{$elty}(undef, 0, 0)
                C = Matrix{$elty}(undef, 0, 0)
            end

            return C

        end     # function

    end         # eval
end             # for

#--- ERROR dumpings ---#
function check_multiple(A::Matrix{Float64}, B::Matrix{Float64})
    mA, nA = size(A)
    mB, nB = size(B)
    if nA != mB
        throw(DimensionMismatch("matrix A has dimensions ($mA, $nA), matrix B has dimensions ($mB, $nB)"))
    end
end