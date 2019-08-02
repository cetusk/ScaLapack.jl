using MPI
using ScaLapack
using ScaLapack: BLACS, ScaLapackLite

# for check correction
using LinearAlgebra

# for DEBUG
using Printf
const DEBUG = true
function prtf(x)
    if x < 0; s = @sprintf("%.7f", x);
    else s = @sprintf(" %.7f", x); end;
end

# problem size
const nrows = 20
const ncols = 20
const nrows_block = 6
const ncols_block = 6
const nprocrows = 2
const nproccols = 2

# test type
const testtype = ComplexF64

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
        A = Matrix{testtype}(undef, nrows, ncols)
        B = Matrix{testtype}(undef, nrows, ncols)
        for ia::Integer = 1 : nrows
            for ja::Integer = 1 : ncols
                A[ia, ja] = convert(Float64, ia+ja)
                B[ia, ja] = convert(Float64, ia*ja)
            end
        end
    else
        A = Matrix{testtype}(undef, 0, 0)
        B = Matrix{testtype}(undef, 0, 0)
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

    A = Matrix{testtype}(undef, nrows, ncols)
    for ia::Integer = 1 : nrows
        for ja::Integer = 1 : ncols
            A[ia, ja] = convert(testtype, rand())
        end
    end
    params = ScaLapackLite.ScaLapackLiteParams(nrows_block, ncols_block, nprocrows, nproccols, root)
    slm_A = ScaLapackLite.ScaLapackLiteMatrix(params, A)

    slm_hA = ScaLapackLite.hessenberg(slm_A, true)
    print("[$rank] A: $(slm_hA.X)\n")
end

function schur_test(root, comm)
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)
    MPI.Barrier(comm);

    A = Matrix{testtype}(undef, nrows, ncols)
    for ia::Integer = 1 : nrows
        for ja::Integer = 1 : ncols
            A[ia, ja] = convert(testtype, rand())
        end
    end
    params = ScaLapackLite.ScaLapackLiteParams(nrows_block, ncols_block, nprocrows, nproccols, root)
    slm_A = ScaLapackLite.ScaLapackLiteMatrix(params, A)

    if rank == root
        # reference: original matrix A
        A_ref = deepcopy(A)
        # reference: LAPACK routine
        (eig_ref,eigvec_ref)=LinearAlgebra.eigen(A)
    end

    # perform Schur decomposition and en-return T
    (eig, slm_Z, slm_T) = ScaLapackLite.schur(slm_A, true)

    # CHECK: recompose A from Schur vectors Z; A = Z * T * Z^H
    slm_A_ = slm_Z * slm_T * slm_Z'
    if rank == root
        frobenius_norm = norm(slm_A_.X-A_ref, 2)
    end

    if rank == root
        eidx=sortperm(real(eig))
        eidx_ref=sortperm(real(eig_ref))
        diff = [ abs(eig[eidx[j]]-eig_ref[eidx_ref[j]]) for j=1:nrows ]

        print("\n--- eigenvalues ---\n")
        wlines = "\nindex / eig(ScaLapackLite) / eig(LAPACK) / | eig(ScaLapackLite)-eig(LAPACK) |\n"
        for idx = 1:nrows
            wlines *= @sprintf("  %s / %s + i %s / %s + i %s / %s\n",
                          lpad(idx, 3, "0"),
                          prtf(real(eig[eidx[idx]])), prtf(imag(eig[eidx[idx]])),
                          prtf(real(eig_ref[eidx_ref[idx]])), prtf(imag(eig_ref[eidx_ref[idx]])),
                          prtf(diff[idx]))
        end
        wlines *= "\n"; print(wlines);

        print("\n--- Schur vectors / eigen vectors ---\n")

        wline = @sprintf("\nFrobenius norm: || Z * T * Z^H - A ||_{2} = %s\n", prtf(frobenius_norm))
        print(wline)

        for jdx = 1:ncols
            print("\n[$jdx-th vec]:\n")
            wlines = "\nindex / Schur vector (ScaLapackLite) / eig vector (LAPACK)\n"
            for idx = 1:nrows
                wlines *= @sprintf("  %s / %s + i %s / %s + i %s\n",
                            lpad(idx, 3, "0"),
                            prtf(real(slm_Z.X[idx, eidx[jdx]])),
                            prtf(imag(slm_Z.X[idx, eidx[jdx]])),
                            prtf(real(eigvec_ref[idx,eidx_ref[jdx]])),
                            prtf(imag(eigvec_ref[idx,eidx_ref[jdx]])))
            end
            wlines *= "\n"; print(wlines);
        end

    end

end

function eig_test(root, comm)
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)
    MPI.Barrier(comm);

    A = Matrix{testtype}(undef, nrows, ncols)
    for ia::Integer = 1 : nrows
        for ja::Integer = 1 : ncols
            # A[ia, ja] = convert(testtype, rand())
            A[ia, ja] = convert(testtype, rand()+im*rand())
        end
    end
    A_copied = deepcopy(A)
    params = ScaLapackLite.ScaLapackLiteParams(nrows_block, ncols_block, nprocrows, nproccols, root)
    slm_A = ScaLapackLite.ScaLapackLiteMatrix(params, A)

    if rank == root
        # reference: LAPACK routine
        (eigvals_ref,eigvecs_ref)=LinearAlgebra.eigen(A)
    end

    # find eigenvalues and eigenvectors
    eigidx = Vector{Int64}([])
    (eigvals, eigvecs) = ScaLapackLite.eigen(slm_A, eigidx, false)

    if rank == root
        eidx = sortperm(real(eigvals))
        eidx_ref = sortperm(real(eigvals_ref))

        print("\n--- eigenvalues ---\n")
        wlines = "\nindex / eig(ScaLapackLite) / eig(LAPACK)\n"
        for idx = 1:nrows
            wlines *= @sprintf("  %s / %s + i %s / %s + i %s\n",
                          lpad(idx, 3, "0"),
                          prtf(real(eigvals[eidx[idx]])), prtf(imag(eigvals[eidx[idx]])),
                          prtf(real(eigvals_ref[eidx_ref[idx]])), prtf(imag(eigvals_ref[eidx_ref[idx]])))
        end
        wlines *= "\n"; print(wlines);

        print("\n--- eigv(ScaLapackLite) / eigv(LAPACK) ---\n")
        for jdx = 1:ncols

            # calc error
            x = Vector{testtype}(eigvecs[:,eidx[jdx]])
            x_ref = Vector{testtype}(eigvecs_ref[:,eidx_ref[jdx]])
            ej = eigvals[eidx[jdx]]
            ej_ref = eigvals_ref[eidx_ref[jdx]]
            error = norm(A_copied*x - ej*x, 2)
            error_ref = norm(A_copied*x_ref - ej_ref*x_ref, 2)
            wlines = @sprintf("\nlog10(|| Ax - λx ||)[ScaLapackLite] = %s\nlog10(|| Ax - λx ||)[LAPACK]        = %s\n",
                               prtf(log10(error)), prtf(log10(error_ref)))

            print("\n[$jdx-th vec]:\n")
            print(wlines)
            wlines = "\nindex / eigv(ScaLapackLite) / eigv(LAPACK)\n"
            for idx = 1:nrows
                wlines *= @sprintf("  %s / %s + i %s / %s + i %s\n",
                            lpad(idx, 3, "0"),
                            prtf(real(eigvecs[idx, eidx[jdx]])), prtf(imag(eigvecs[idx, eidx[jdx]])),
                            prtf(real(eigvecs_ref[idx, eidx_ref[jdx]])), prtf(imag(eigvecs_ref[idx, eidx_ref[jdx]])))
            end
            wlines *= "\n"; print(wlines);
        end

    end

end

function main()
    # MPI initilaize
    MPI.Init()
    comm = MPI.COMM_WORLD
    finalizer(mpi_finalizer, comm)
    # test
    # mutiple_test(0, comm)
    # hessenberg_test(0, comm)
    # schur_test(0, comm)
    eig_test(0, comm)
end

main()
