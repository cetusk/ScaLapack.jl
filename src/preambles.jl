
# dependencies
using MPI
using Compat
using LinearAlgebra
using LinearAlgebra: BlasFloat, BlasReal
using DistributedArrays, Distributed
using DistributedArrays: DArray, defaultdist

# enicode encoded array to input ccall
f_pchar(scope::Char) = transcode(UInt8, string(scope))

# path to scalapack dynamic library
const libscalapack = "/usr/local/lib/libscalapack.dylib"
# depend on envionment
# const ScaInt = Int32
const ScaInt = Int64

struct ScaLapackException <: Exception
    info::Integer
end

if myid() > 1
    MPI.Initialized() || MPI.Init()
end