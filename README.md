# ScaLapack.jl
This file was created refering this repogitory: https://github.com/JuliaParallel/ScaLAPACK.jl. And supports Julia v.1.1.0.

# Preparing for use ScaLapack.jl @ Mac OS X
```
brew install openmpi
brew install scalapack --with-shared-libs
```

# Add package ScaLapack
```
(v1.1) pkg> add https://github.com/Cetus-K/ScaLapack.jl.git
```

# Execution: example
```
$ mpirun -np 4 --hostfile ../hosts /path/to/bin/of/julia /path/to/source.jl
```

# Usage: example for matrix multiplication
*_1. declare using ScaLapack.jl_*
```
using MPI
using ScaLapack
```
*_2. define parallelization parameters_*
```
const nrows_block = 2
const ncols_block = 2
const nprocrows = 2
const nproccols = 2
```
*_3. define matrices A and B_*
```
# generate matrix ( T/nrows/ncols = user defined )
A = Matrix{T}(undef, nrows, ncols)
B = Matrix{T}(undef, nrows, ncols)
if rank == 0
    for ia::Integer = 1 : nrows
        for ja::Integer = 1 : ncols
            A[ia, ja] = convert(T, ia+ja)
            B[ia, ja] = convert(T, ia*ja)
        end
    end
end
MPI.Barrier(comm)
MPI.Bcast!(A, 0, comm)
MPI.Bcast!(B, 0, comm)
MPI.Barrier(comm)
```
*_4. perform C = AB_*
```
params = ScaLapack.ScaLapackParams(nrows_block, ncols_block, nprocrows, nproccols)
slm_A = ScaLapack.ScaLapackMatrix(params, A)
slm_B = ScaLapack.ScaLapackMatrix(params, B)
slm_C = slm_A * slm_B
```
*_5. extract result_*
```
C = slm_C.X
```