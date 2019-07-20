# ScaLapack.jl
This file was created refering this repogitory: https://github.com/JuliaParallel/ScaLAPACK.jl. And supports Julia v.1.1.0.

# Preparing for use MPI @ Mac OS X
```
$ brew install cmake
$ xcode-select --install
$ brew install gcc
```

# Preparing for use ScaLapack.jl @ Mac OS X
```
$ brew install openmpi
$ brew install scalapack
```

# Add package MPI
```
(v1.1) pkg> build MPI
(v1.1) pkg> add MPI
```

# Add package ScaLapack
```
(v1.1) pkg> add https://github.com/Cetus-K/ScaLapack.jl.git
```

# Execution: example
```
$ mpirun -np 4 --hostfile /path/to/hostfile /path/to/bin/of/julia /path/to/source.jl
```

# Usage: example for matrix multiplication
###_1. declare using ScaLapack.jl_###
```
using MPI
using ScaLapack
using ScaLapack: BLACS, ScaLapackLite
```
###_2. define parallelization parameters_###
```
const nrows_block = 2
const ncols_block = 2
const nprocrows = 2
const nproccols = 2
```
###_3. define matrices A and B_###
```
# T/nrows/ncols = user defined
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
###_4. prepare ScaLapackLiteParams and create ScaLapackLiteMatrix_###
```
params = ScaLapackLite.ScaLapackLiteParams(nrows_block, ncols_block, nprocrows, nproccols)
slm_A = ScaLapackLite.ScaLapackLiteMatrix(params, A)
slm_B = ScaLapackLite.ScaLapackLiteMatrix(params, B)
```
###_5. perform C = AB and extract result_###
```
slm_C = slm_A * slm_B
C = slm_C.X
```
