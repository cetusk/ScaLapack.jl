using MPI
using ScaLapack
using ScaLapack: BLACS, ScaLapackLite

# for check correction
using LinearAlgebra

# for DEBUG
using Printf
const DEBUG = false
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
            A[ia, ja] = convert(Float64, rand())
        end
    end
    params = ScaLapackLite.ScaLapackLiteParams(nrows_block, ncols_block, nprocrows, nproccols, root)
    slm_A = ScaLapackLite.ScaLapackLiteMatrix(params, A)

    if rank == root
        # reference: LAPACK routine
        (eig_ref,eigvec_ref)=LinearAlgebra.eigen(A)
    end

    # slm_hA = ScaLapackLite.hessenberg(slm_A, true)
    # print("[$rank] A: $(slm_hA.X)\n")

    (eig,schurvec) = ScaLapackLite.eigs(slm_A)

    if rank == root
        eidx=sortperm(real(eig))
        eidx_ref=sortperm(real(eig_ref))
        diff = [ abs(eig[eidx[j]]-eig_ref[eidx_ref[j]]) for j=1:nrows ]

        print("\n--- eigenvalues ---\n")
        wlines = "\nindex / eig(ScaLapackLite) / eig(LAPACK) / | eig(ScaLapackLite)-eig(LAPACK) |\n"
        for idx = 1:nrows
            wlines*= @sprintf("  %s / %s + i %s / %s + i %s / %s\n",
                          lpad(idx, 3, "0"),
                          prtf(real(eig[eidx[idx]])), prtf(imag(eig[eidx[idx]])),
                          prtf(real(eig_ref[eidx_ref[idx]])), prtf(imag(eig_ref[eidx_ref[idx]])),
                          prtf(diff[idx]))
        end
        wlines*="\n"; print(wlines);

        print("\n--- Schur / eigen vectors ---\n")
        for jdx = 1:ncols
            print("\n[$jdx-th vec]:\n")
            wlines = "\nindex / Schur vector (ScaLapackLite) / eig vector (LAPACK)\n"
            for idx = 1:nrows
                wlines*= @sprintf("  %s / %s / %s + i %s\n",
                            lpad(idx, 3, "0"),
                            prtf(schurvec[idx, eidx[jdx]]),
                            prtf(real(eigvec_ref[idx,eidx_ref[jdx]])),
                            prtf(imag(eigvec_ref[idx,eidx_ref[jdx]])))
            end
            wlines*="\n"; print(wlines);
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
    hessenberg_test(0, comm)
end

main()
