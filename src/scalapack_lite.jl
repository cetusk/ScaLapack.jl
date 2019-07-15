
# depend on envionment
# const ScaInt = Int32
const ScaInt = Int64

# enicode encoded array to input ccall
f_pchar(scope::Char) = transcode(UInt8, string(scope))

abstract type ScaLapackLite end

mutable struct ScaLapackParams <: ScaLapackLite
    mb::ScaInt; nb::ScaInt;
    mprocs::ScaInt; nprocs::ScaInt;
    # Constructor
    ScaLapackParams() = new(2, 2, 1, 1)
    ScaLapackParams(mb, nb, mprocs, nprocs) = new(mb, nb, mprocs, nprocs)
    # functions
    function set_procparams(nrows_block::ScaInt, ncols_block::ScaInt, nprocrows::ScaInt, nproccols::ScaInt)
        mb = nrows_block; nb = ncols_block;
        mprocs = nprocrows; nprocs = nproccols;
    end
end

function Base.:(==)(A::ScaLapackParams, B::ScaLapackParams)
    is_equal_blocks = (A.mb == B.mb) && (A.nb == B.nb)
    is_equal_procs = (A.mprocs == B.mprocs) && (A.nprocs == B.nprocs)
    return is_equal_blocks && is_equal_procs
end

mutable struct ScaLapackMatrix <: ScaLapackLite
    params::ScaLapackParams
    X::Matrix{Float64}
    ScaLapackMatrix() = new(ScaLapackParams())
    ScaLapackMatrix(params) = new(params)
    ScaLapackMatrix(params, X) = new(params, X)
end

# perform A*B
function Base.:(*)(A::ScaLapackMatrix, B::ScaLapackMatrix)
    if A.params == B.params
        C = multiple(A.X, B.X, A.params.mb, A.params.nb, A.params.mprocs, A.params.nprocs)
        return ScaLapackMatrix(A.params, C)
    end
end

function multiple(A::Matrix{Float64}, B::Matrix{Float64}, mb::ScaInt, nb::ScaInt, mprocs::ScaInt, nprocs::ScaInt)

    # params
    m, n = size(A)

    ctxt = sl_init(mprocs, nprocs)
    mproc, nproc, myrow, mycol = BLACS.gridinfo(ctxt)
    mxllda = numroc(m, mb, mproc, 0, mprocs)
    mxlocc = numroc(n, nb, nproc, 0, nprocs)

    if mproc >= 0 && nproc >= 0
        # get array descriptor
        zero=0
        desca = descinit(m, n, mb, nb, zero, zero, ctxt, mxllda)
        descb = descinit(m, n, mb, nb, zero, zero, ctxt, mxllda)
        descc = descinit(m, m, mb, nb, zero, zero, ctxt, mxllda)
        
        # generate process matrix
        my_A = zeros(Float64,mxllda,mxlocc)
        my_B = zeros(Float64,mxllda,mxlocc)
        for ia::ScaInt=1:m
            for ja::ScaInt=1:n
                pXelset!(my_A,ia,ja,desca,A[ia,ja])
            end
        end
        for ib::ScaInt=1:m
            for jb::ScaInt=1:n
                pXelset!(my_B,ib,jb,descb,B[ib,jb])
            end
        end
        
        # perform A*B
        α = 1.0; β = 0.0; one = 1;
        my_C = zeros(Float64, mxllda, mxllda)
        transa = 'n'; transb = 't';
        pXgemm!(transa,transb,
                m, m, n,
                α,
                my_A, one, one, desca,
                my_B, one, one, descb,
                β,
                my_C, one, one, descc)

        C = Matrix{Float64}(undef,m,m)
        for ic::ScaInt=1:m
            for jc::ScaInt=1:m
                C[ic, jc] = pXelget('A', 'I', my_C, ic, jc, descc)
            end
        end

    end
    BLACS.barrier(ctxt,'A')
    ScaLapack.BLACS.gridexit(ctxt)

    return C

end
