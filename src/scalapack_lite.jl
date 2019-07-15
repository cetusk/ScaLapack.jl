
# abstract ScaLapackLite type
abstract type AbstractScaLapackLite end

# ScaLapackLite parameters
mutable struct ScaLapackLiteParams <: AbstractScaLapackLite
    mb::ScaInt; nb::ScaInt;
    mprocs::ScaInt; nprocs::ScaInt;
    # Constructor
    ScaLapackLiteParams() = new(1, 1, 1, 1)
    ScaLapackLiteParams(mb, nb, mprocs, nprocs) = new(mb, nb, mprocs, nprocs)
    # functions
    function set_procparams(nrows_block::ScaInt, ncols_block::ScaInt, nprocrows::ScaInt, nproccols::ScaInt)
        mb = nrows_block; nb = ncols_block;
        mprocs = nprocrows; nprocs = nproccols;
    end
end
function Base.:(==)(A::ScaLapackLiteParams, B::ScaLapackLiteParams)
    is_equal_blocks = (A.mb == B.mb) && (A.nb == B.nb)
    if !is_equal_blocks
        error("different block sizes: A and B")
    end
    is_equal_procs = (A.mprocs == B.mprocs) && (A.nprocs == B.nprocs)
    if !is_equal_procs
        error("different process grids: A and B")
    end
    return is_equal_blocks && is_equal_procs
end

# ScaLapackMatrix type
mutable struct ScaLapackLiteMatrix <: AbstractScaLapackLite
    params::ScaLapackLiteParams
    X::Matrix{Float64}
    ScaLapackLiteMatrix() = new(ScaLapackLiteParams())
    ScaLapackLiteMatrix(params) = new(params)
    ScaLapackLiteMatrix(params, X) = new(params, X)
end

# perform A*B
function Base.:(*)(A::ScaLapackLiteMatrix, B::ScaLapackLiteMatrix)
    if A.params == B.params
        C = multiple(A.X, B.X, A.params.mb, A.params.nb, A.params.mprocs, A.params.nprocs)
        return ScaLapackLiteMatrix(A.params, C)
    end
end
function multiple(A::Matrix{Float64}, B::Matrix{Float64}, mb::ScaInt, nb::ScaInt, mprocs::ScaInt, nprocs::ScaInt)

    # create context
    ctxt = ScaLapack.sl_init(mprocs, nprocs)

    # params
    m, n = size(A)
    mproc, nproc, myrow, mycol = BLACS.gridinfo(ctxt)
    mxllda = ScaLapack.numroc(m, mb, mproc, 0, mprocs)
    mxlocc = ScaLapack.numroc(n, nb, nproc, 0, nprocs)

    if mproc >= 0 && nproc >= 0
        # get array descriptor
        zero=0
        desca = ScaLapack.descinit(m, n, mb, nb, zero, zero, ctxt, mxllda)
        descb = ScaLapack.descinit(m, n, mb, nb, zero, zero, ctxt, mxllda)
        descc = ScaLapack.descinit(m, m, mb, nb, zero, zero, ctxt, mxllda)
        
        # generate process matrix
        my_A = zeros(Float64,mxllda,mxlocc)
        my_B = zeros(Float64,mxllda,mxlocc)
        for ia::ScaInt=1:m
            for ja::ScaInt=1:n
                ScaLapack.pXelset!(my_A,ia,ja,desca,A[ia,ja])
            end
        end
        for ib::ScaInt=1:m
            for jb::ScaInt=1:n
                ScaLapack.pXelset!(my_B,ib,jb,descb,B[ib,jb])
            end
        end
        
        # perform A*B
        α = 1.0; β = 0.0; one = 1;
        my_C = zeros(Float64, mxllda, mxllda)
        transa = 'n'; transb = 't';
        ScaLapack.pXgemm!(transa,transb,
                          m, m, n,
                          α,
                          my_A, one, one, desca,
                          my_B, one, one, descb,
                          β,
                          my_C, one, one, descc)

        C = Matrix{Float64}(undef,m,m)
        for ic::ScaInt=1:m
            for jc::ScaInt=1:m
                C[ic, jc] = ScaLapack.pXelget('A', 'I', my_C, ic, jc, descc)
            end
        end

    end
    BLACS.barrier(ctxt,'A')
    BLACS.gridexit(ctxt)

    return C

end
