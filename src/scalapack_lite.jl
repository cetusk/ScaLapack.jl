
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
    is_equal_procs = (A.mprocs == B.mprocs) && (A.nprocs == B.nprocs)
    is_equal_rootproc = (A.rootproc == B.rootproc)
    return is_equal_blocks && is_equal_procs && is_equal_rootproc
end

# ScalapackLiteVector type
mutable struct ScaLapackLiteVector <: AbstractScaLapackLite
    params::ScaLapackLiteParams
    x::Vector{T} where {T} 
    ScaLapackLiteVector() = new(ScaLapackLiteVector())
    ScaLapackLiteVector(params) = new(params)
    v(params, x) = new(params, x)
end

# ScaLapackLiteMatrix type
# plannings:
#   - add member "localpart" that distributed matrix in each process
mutable struct ScaLapackLiteMatrix <: AbstractScaLapackLite
    params::ScaLapackLiteParams
    X::Matrix{T} where {T} 
    ScaLapackLiteMatrix() = new(ScaLapackLiteParams())
    ScaLapackLiteMatrix(params) = new(params)
    ScaLapackLiteMatrix(params, X) = new(params, X)
end

#--- operator overloadings ---#

# perform A'
function Base.:(adjoint)(A::ScaLapackLiteMatrix)
    if length(A.X) > 0
        elty = typeof(A.X[1,1])
        if elty <: Complex
            return ScaLapackLiteMatrix(A.params, conj(transpose(A.X)))
        else
            return ScaLapackLiteMatrix(A.params, transpose(A.X))
        end
    else
        return ScaLapackLiteMatrix(A.params, transpose(A.X))
    end
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

#--- generate zero and identity matrix ---#
for (elty) in (:Float32, :Float64, :ComplexF32, :ComplexF64)
    @eval begin
        function zero!(A::Matrix{$elty}, m::ScaInt, n::ScaInt, desc::Vector{ScaInt})
            uplo = 'A'
            α = convert($elty, 0); β = convert($elty, 1);
            ia = one; ja = one;
            ScaLapack.pXlaset!(uplo, m, n,
                               α, β,
                               A, ia, ja, desc)
        end
    end
end
for (elty) in (:Float32, :Float64, :ComplexF32, :ComplexF64)
    @eval begin
        function identity!(I::Matrix{$elty}, m::ScaInt, n::ScaInt, desc::Vector{ScaInt})
            uplo = 'A'
            α = convert($elty, 0); β = convert($elty, 1);
            ia = one; ja = one;
            ScaLapack.pXlaset!(uplo, m, n,
                               α, β,
                               I, ia, ja, desc)
        end
    end
end

#--- multiple ---#
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

            # copy matrix from master to slave
            A = MPI.bcast(A, rootproc, comm)
            B = MPI.bcast(B, rootproc, comm)
            MPI.Barrier(comm)

            # check both dimensions; cols(A) = rows(B)
            # (mA, nA), (mB, nB) = (mA, nA), (nA, nB)
            if rank == rootproc
                check_multiple(A, B)
            end

            # type of the matrix A
            elty = typeof(A[1,1])

            # create context
            ctxt = ScaLapack.sl_init(mprocs, nprocs)
            ctxt0 = ScaLapack.sl_init(mprocs, nprocs)

            # matrix parameters
            mA, nA = size(A)
            mB, nB = size(B)
            mproc, nproc, myrow, mycol = BLACS.gridinfo(ctxt)
            mxlocr_A = ScaLapack.numroc(mA, mblocks, mproc, zero, mprocs)
            mxlocr_B = ScaLapack.numroc(mB, mblocks, mproc, zero, mprocs)
            mxlocc_A = ScaLapack.numroc(nA, nblocks, nproc, zero, nprocs)
            mxlocc_B = ScaLapack.numroc(nB, nblocks, nproc, zero, nprocs)
            mxllda_A = max(one, mxlocr_A)
            mxllda_B = max(one, mxlocr_B)

            if mproc >= zero && nproc >= zero

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
                α = convert(elty, one); β = convert(elty, zero);
                myC = zeros($elty, mxlocr_A, mxlocc_B)
                transa = 'n'; transb = 'n';
                ScaLapack.pXgemm!(transa, transb,
                                  mA, nB, nA,
                                  α,
                                  myA, one, one, desc_myA,
                                  myB, one, one, desc_myB,
                                  β,
                                  myC, one, one, desc_myC)

                # merge local matrix to global
                C = Matrix{$elty}(undef, mA, nB)
                ScaLapack.pXgemr2d!(mA, nB,
                                    myC, one, one, desc_myC,
                                    C, one, one, desc_C,
                                    ctxt0)

            end
            
            BLACS.barrier(ctxt, 'A')
            BLACS.barrier(ctxt0, 'A')
            BLACS.gridexit(ctxt)
            BLACS.gridexit(ctxt0)

            # free
            if rank != rootproc
                A = Matrix{$elty}(undef, zero, zero)
                B = Matrix{$elty}(undef, zero, zero)
                C = Matrix{$elty}(undef, zero, zero)
            end

            return C

        end     # function

    end         # eval
end             # for

#--- find hessenberg matrix ---#
function hessenberg(sllm_A::ScaLapackLiteMatrix, reduced::Bool = true, get_Q::Bool = true)

    # assume: ScaLapackLiteParams must be correctly given for all processes
    params = sllm_A.params
    
    # MPI parameters
    mblocks = params.mblocks
    nblocks = params.nblocks
    mprocs = params.mprocs
    nprocs = params.nprocs
    rootproc = params.rootproc
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    # copy matrix from master to slave
    sllm_A = MPI.bcast(sllm_A, rootproc, comm)

    # decompose
    A = sllm_A.X

    # type of the matrix A
    elty = typeof(A[1,1])

    # check dimensions
    if rank == rootproc
        check_hessenberg(A)
        if mblocks != nblocks
            throw(DimensionMismatch("process grid must be contained from squared blocks"))
        end
    end

    # create context
    ctxt = ScaLapack.sl_init(mprocs, nprocs)
    ctxt0 = ScaLapack.sl_init(mprocs, nprocs)

    # matrix parameters
    m, n = convert.(ScaInt, size(A))
    mproc, nproc, myrow, mycol = BLACS.gridinfo(ctxt)
    mxlocr = ScaLapack.numroc(m, mblocks, mproc, zero, mprocs)
    mxlocc = ScaLapack.numroc(n, nblocks, nproc, zero, nprocs)
    mxllda = max(one, mxlocr)

    if mproc >= zero && nproc >= zero

        two = convert(ScaInt, 2)
        three = convert(ScaInt, 3)

        # Q matrix is enabled if get_Q = true 
        Q = Matrix{elty}(undef, zero, zero)

        # get array descriptor
        desc_A = ScaLapack.descinit(m, n, m, n, zero, zero, ctxt0, m)
        desc_myA = ScaLapack.descinit(m, n, mblocks, nblocks, zero, zero, ctxt, mxllda)

        # prepare local matrix
        myA = zeros(elty, mxllda, mxlocc)
        ScaLapack.pXgemr2d!(m, n,
                            A, one, one, desc_A,
                            myA, one, one, desc_myA,
                            ctxt)

        # MPI params
        numblocks = mblocks
        rsrc_a = zero
        csrc_a = zero

        # find hessenberg matrix
        ilo = one; ihi = m;
        ia = one; ja = one;
        τ = zeros(elty, ScaLapack.numroc(ja+n-two, numblocks, mycol, csrc_a, nprocs))
        ScaLapack.pXgehrd!(m, ilo, ihi,
                           myA, ia, ja, desc_myA,
                           τ)

        if get_Q

            # temporary variables
            desc_Q = ScaLapack.descinit(m, n, m, n, zero, zero, ctxt0, m)
            desc_myQ = ScaLapack.descinit(m, n, mblocks, nblocks, zero, zero, ctxt, mxllda)
            Q = zeros(elty, m, n)
            myQ = zeros(elty, mxllda, mxlocc)
            identity!(myQ, m, n, desc_myQ)

            # generate orthogonal Q matrix; Q = Q * I
            # the orthogonal matrix Q is generated from the non-reduced Hessenberg matrix myA
            # Q = ( I-tau*v(1)*v(1)' )*( I-tau*v(2)*v(2)' )...
            # the v(j) is contained in the lower off-diag excepted the sub-siag of myA
            side = 'L'; trans = 'N';
            ScaLapack.pXYYmhr!(side, trans,
                               m, n, ilo, ihi,
                               myA, ia, ja, desc_myA,
                               τ,
                               myQ, one, one, desc_myQ)
            # merge local matrix to global
            ScaLapack.pXgemr2d!(m, n,
                                myQ, one, one, desc_myQ,
                                Q, one, one, desc_Q,
                                ctxt0)
        end

        if reduced
            # remove non-reduced part of the upper Hessenberg matrix
            uplo = 'L'
            α = convert(elty, zero); β = convert(elty, zero);
            ia = three; ja = one;
            ScaLapack.pXlaset!(uplo, m-two, n-two,
                               α, β,
                               myA, ia, ja, desc_myA)
        end

        # merge local matrix to global
        ScaLapack.pXgemr2d!(m, n,
                            myA, one, one, desc_myA,
                            A, one, one, desc_A,
                            ctxt0)

    end
    
    BLACS.barrier(ctxt, 'A')
    BLACS.barrier(ctxt0, 'A')
    BLACS.gridexit(ctxt)
    BLACS.gridexit(ctxt0)

    # free
    if rank != rootproc
        A = Matrix{elty}(undef, zero, zero)
        if get_Q; Q = Matrix{elty}(undef, zero, zero); end;
    end

    if get_Q; return (ScaLapackLiteMatrix(params, A), ScaLapackLiteMatrix(params, Q));
    else; return ScaLapackLiteMatrix(params, A); end;

end     # function

#--- Schur decomposition ---#
function schur(sllm_A::ScaLapackLiteMatrix, force_complex::Bool = false, get_T::Bool = true)

    # assume: ScaLapackLiteParams must be correctly given for all processes
    params = sllm_A.params

    # MPI parameters
    mblocks = params.mblocks
    nblocks = params.nblocks
    mprocs = params.mprocs
    nprocs = params.nprocs
    rootproc = params.rootproc
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    # perform find reduced upper hessenberg matrix
    # the arguments: general matrix A, reducing flag, return Q matrix
    (sllm_H, sllm_Q) = ScaLapackLite.hessenberg(sllm_A, true, true)

    # copy matrix from master to slave
    sllm_H = MPI.bcast(sllm_H, rootproc, comm)
    sllm_Q = MPI.bcast(sllm_Q, rootproc, comm)

    # type of the matrix H
    elty = typeof(sllm_H.X[1,1])
    elty_s = elty_w = elty_e = elty
    if elty == ComplexF32; elty_s = Float32;
    elseif elty == ComplexF64; elty_s = Float64; end
    if elty == Float32; elty_w = ComplexF32;
    elseif elty == Float64; elty_w = ComplexF64; end
    if force_complex; elty_e = elty_w; end

    # check dimensions
    if rank == rootproc
        check_schur(sllm_H.X)
        if mblocks != nblocks
            throw(DimensionMismatch("process grid must be contained from squared blocks"))
        end
    end

    # convert Float to ComplexF
    if force_complex
        sllm_H.X = convert(Matrix{elty_e}, sllm_H.X)
        sllm_Q.X = convert(Matrix{elty_e}, sllm_Q.X)
    end

    # decompose
    H = sllm_H.X; Q = sllm_Q.X;

    # create context
    ctxt = ScaLapack.sl_init(mprocs, nprocs)
    ctxt0 = ScaLapack.sl_init(mprocs, nprocs)

    # matrix parameters
    m, n = convert.(ScaInt, size(H))
    mproc, nproc, myrow, mycol = BLACS.gridinfo(ctxt)
    mxlocr = ScaLapack.numroc(m, mblocks, mproc, zero, mprocs)
    mxlocc = ScaLapack.numroc(n, nblocks, nproc, zero, nprocs)
    mxllda = max(one, mxlocr)

    if mproc >= zero && nproc >= zero

        two = convert(ScaInt, 2)
        three = convert(ScaInt, 3)

        # get array descriptor
        desc_H = ScaLapack.descinit(m, n, m, n, zero, zero, ctxt0, m)
        desc_Q = ScaLapack.descinit(m, n, m, n, zero, zero, ctxt0, m)
        desc_myH = ScaLapack.descinit(m, n, mblocks, nblocks, zero, zero, ctxt, mxllda)
        desc_myQ = ScaLapack.descinit(m, n, mblocks, nblocks, zero, zero, ctxt, mxllda)

        # generate process matrix
        myH = zeros(elty_e, mxllda, mxlocc)
        ScaLapack.pXgemr2d!(m, n,
                            H, one, one, desc_H,
                            myH, one, one, desc_myH,
                            ctxt)
        myQ = zeros(elty_e, mxllda, mxlocc)
        ScaLapack.pXgemr2d!(m, n,
                            Q, one, one, desc_Q,
                            myQ, one, one, desc_myQ,
                            ctxt)

        # perform Schur decomposition: H = S * T * S^H
        # its eigenvalues ( = diag(T) ) are stored into the w
        # myQ will be updated to the local part of Q * S
        #       because of the: H = Q^H * A * Q = S * T * S^H
        #                         <=> A = (Q*S) * T (Q*S)^H
        # if wantt = true: myA will be overwitten to T
        wantt = get_T; wantz = true;
        ilo = one; ihi = m;
        w = zeros(elty_w, m)
        # for real input
        if elty == Float32 || elty == Float64
            wr = zeros(elty, m); wi = zeros(elty, m)
            ScaLapack.pXlahqr!(wantt, wantz, m,
                               ilo, ihi, myH, desc_myH,
                               wr, wi,
                               ilo, ihi, myQ, desc_myQ)
            w = wr + im*wi
        # for complex input
        elseif elty == ComplexF32 || elty == ComplexF64
            ScaLapack.pXlahqr!(wantt, wantz, m,
                               ilo, ihi, myH, desc_myH,
                               w,
                               ilo, ihi, myQ, desc_myQ)
        end

        # merge local Schur vectors matrix to global
        ScaLapack.pXgemr2d!(m, n,
                            myQ, one, one, desc_myQ,
                            Q, one, one, desc_Q,
                            ctxt0)
        # merge local upper triangular matrix T
        if get_T
            ScaLapack.pXgemr2d!(m, n,
                                myH, one, one, desc_myH,
                                H, one, one, desc_H,
                                ctxt0)
        end

    end

    BLACS.barrier(ctxt, 'A')
    BLACS.barrier(ctxt0, 'A')
    BLACS.gridexit(ctxt)
    BLACS.gridexit(ctxt0)

    # free
    if rank != rootproc
        w = Vector{elty_w}(undef, zero)
        Q = Matrix{elty_e}(undef, zero, zero)
        H = Matrix{elty_e}(undef, zero, zero)
    end

    if get_T; return (w, ScaLapackLiteMatrix(params, Q), ScaLapackLiteMatrix(params, H));
    else; return (w, ScaLapackLiteMatrix(params, Q)); end;

end

#--- find eigenvalues ---#
function eigen(sllm_A::ScaLapackLiteMatrix, eidx::Vector{ScaInt} = [], is_leftside::Bool = false)

    # assume: ScaLapackLiteParams must be correctly given for all processes
    params = sllm_A.params

    # MPI parameters
    mblocks = params.mblocks
    nblocks = params.nblocks
    mprocs = params.mprocs
    nprocs = params.nprocs
    rootproc = params.rootproc
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    # perform Schur decomposition
    # the arguments: general matrix A, the flag to force convert to complex, return T matrix
    (eigvals, sllm_Z, sllm_T) = schur(sllm_A, true, true)
    sllm_Z = MPI.bcast(sllm_Z, rootproc, comm)
    sllm_T = MPI.bcast(sllm_T, rootproc, comm)

    # type of the matrix T
    elty = typeof(sllm_T.X[1,1])
    elty_s = elty; elty_ev = elty;
    if elty == ComplexF32; elty_s = Float32;
    elseif elty == ComplexF64; elty_s = Float64; end
    if elty == Float32; elty_ev = ComplexF32;
    elseif elty == Float64; elty_ev = ComplexF64; end

    # check dimensions
    if rank == rootproc
        check_eigen(sllm_T.X)
        if mblocks != nblocks
            throw(DimensionMismatch("process grid must be contained from squared blocks"))
        end
    end

    # convert Float to ComplexF
    if elty != elty_ev
        sllm_T.X = convert(Matrix{elty_ev}, sllm_T.X)
        sllm_Z.X = convert(Matrix{elty_ev}, sllm_Z.X)
    end

    # decompose and copy Z to initial input V for pXtrevc
    T = sllm_T.X; Z = sllm_Z.X;

    # create context
    ctxt = ScaLapack.sl_init(mprocs, nprocs)
    ctxt0 = ScaLapack.sl_init(mprocs, nprocs)

    # matrix parameters
    m, n = convert.(ScaInt, size(T))
    mproc, nproc, myrow, mycol = BLACS.gridinfo(ctxt)
    mxlocr = ScaLapack.numroc(m, mblocks, mproc, zero, mprocs)
    mxlocc = ScaLapack.numroc(n, nblocks, nproc, zero, nprocs)
    mxllda = max(one, mxlocr)

    # eigenvector parameters
    mm = m; neigs = convert(ScaInt, length(eidx));
    if neigs > zero
        # specified eigen vectors
        select = Vector{Bool}([ false for j = one:m ])
        select[eidx] .= true
    else
        if neigs < zero
            throw(DimensionMismatch("illegal value; num of specified eigen index = $neigs"))
        else
            # all eigen vectors ( neigs = 0 )
            eidx = Vector{ScaInt}([ j for j = one:m ])
            select = Vector{Bool}([ true for j = one:m ])
        end
    end

    if mproc >= zero && nproc >= zero

        # get array descriptor
        desc_T = ScaLapack.descinit(m, n, m, n, zero, zero, ctxt0, m)
        desc_Z = ScaLapack.descinit(m, n, m, n, zero, zero, ctxt0, m)
        desc_myT = ScaLapack.descinit(m, n, mblocks, nblocks, zero, zero, ctxt, mxllda)
        desc_myZ = ScaLapack.descinit(m, n, mblocks, nblocks, zero, zero, ctxt, mxllda)

        # generate process matrix of T and Z
        myT = zeros(elty_ev, mxllda, mxlocc)
        myZ = zeros(elty_ev, mxllda, mxlocc)
        ScaLapack.pXgemr2d!(m, n,
                            T, one, one, desc_T,
                            myT, one, one, desc_myT,
                            ctxt)
        ScaLapack.pXgemr2d!(m, n,
                            Z, one, one, desc_Z,
                            myZ, one, one, desc_myZ,
                            ctxt)

        # find eigenvectors
        o = zeros(ScaInt, zero)
        O = zeros(elty_ev, zero, zero)
        if is_leftside; side = 'L'; else; side = 'R'; end;
        if neigs == zero; howmny = 'B'; else; howmny = 'S'; end
        # for each side
        nV = zero
        if is_leftside
            nV = ScaLapack.pXtrevc!(side, howmny, select,
                                    m, myT, desc_myT,
                                    myZ, desc_myZ, O, o)
        else
            nV = ScaLapack.pXtrevc!(side, howmny, select,
                                    m, myT, desc_myT,
                                    O, o, myZ, desc_myZ)
        end

        # merge local eigen vectors matrix to global
        ScaLapack.pXgemr2d!(m, n,
                            myZ, one, one, desc_myZ,
                            Z, one, one, desc_Z,
                            ctxt0)


    end

    BLACS.barrier(ctxt, 'A')
    BLACS.barrier(ctxt0, 'A')
    BLACS.gridexit(ctxt)
    BLACS.gridexit(ctxt0)

    # free
    if rank != rootproc
        eigvals = Vector{elty_ev}(undef, zero)
        T = Matrix{elty_ev}(undef, zero, zero)
        Z = Matrix{elty_ev}(undef, zero, zero)
    else
        # store actual size of result
        eigvals = eigvals[eidx]
        Z = Z[:, eidx]
    end

    return (eigvals, Z)

end


#--- ERROR dumpings ---#
function check_multiple(A::Matrix{<:Number}, B::Matrix{<:Number})
    mA, nA = size(A)
    mB, nB = size(B)
    if nA != mB
        throw(DimensionMismatch("matrix A has dimensions ($mA, $nA), matrix B has dimensions ($mB, $nB)"))
    end
end
function check_hessenberg(A::Matrix{<:Number})
    mA, nA = size(A)
    if mA != nA
        throw(DimensionMismatch("matrix A has no-squared dimensions ($mA, $nA)"))
    end
end
function check_schur(A::Matrix{<:Number})
    mA, nA = size(A)
    if mA != nA
        throw(DimensionMismatch("matrix A has no-squared dimensions ($mA, $nA)"))
    end
end
function check_eigen(A::Matrix{<:Number})
    mA, nA = size(A)
    if mA != nA
        throw(DimensionMismatch("matrix A has no-squared dimensions ($mA, $nA)"))
    end
end