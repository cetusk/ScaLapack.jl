using ScaLapack
using MPI

MPI.Init()

# Base.disable_threaded_libs()

# problem size
na_rows=2;na_cols=3;
nb_rows=na_cols;nb_cols=1;
blocksize=2

A=randn(na_rows,na_cols)
B=randn(nb_rows,nb_cols)

print("matrix A: $A\n")
print("matrix B: $B\n")

# initialize grid
mGrid,nGrid=size(A.chunks)
id,nprocs=ScaLapack.BLACS.pinfo()
ictxt=ScaLapack.BLACS.gridinit(BLACS.get(0,0),'c',mGrid,nGrid)

print("id: $id, nprocs: $nprocs\n")

# who am I?
nrow,ncol,myrow,mycol=ScaLapack.BLACS.gridinfo(ictxt)
p_nrow=ScaLapack.numroc(na_rows,blocksize,myrow,0,nrow)
p_ncol=ScaLapack.numroc(na_cols,blocksize,mycol,0,ncol)
print("myrow: $myrow, mycol: $mycol,
        blocksize: $blocksize,
        p_nrow: $p_nrow, p_ncol: $p_ncol\n")

#     if nrow >= 0 && ncol >= 0

#         print("check: A\n")

#         # Get DArray info
#         descA=ScaLapack.descinit(p_nrow,p_ncol,blocksize,blocksize,0,0,ictxt,p_ncol)

#         print("check: B\n")

#         eigval=Vector{Float64}(undef,blocksize)
#         eigvec=Matrix{Float64}(undef,blocksize,blocksize)
#         # eigval,eigvec=ScaLapack.pdstedc!('I',blocksize,
#         # eigvec,
#         #                                  Vector{Float64}(undef,blocksize),
#         #                                  Matrix{Float64}(undef,blocksize,blocksize),
#         #                                  1,1,descA)

#         print("check: C\n")

#         # show result
#         if myrow == 0 && mycol == 0
#             println(eigval)
#             println(eigvec)
#         end

#         # clean up
#         ScaLapack.BLACS.gridexit(ictxt)


MPI.Finalize()