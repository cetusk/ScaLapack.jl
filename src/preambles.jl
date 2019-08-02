
# dependencies
using MPI
using Compat
using Printf
using LinearAlgebra: BlasFloat, BlasReal
using DistributedArrays, Distributed
using DistributedArrays: DArray, defaultdist

# enicode encoded array to input ccall
f_pchar(scope::Char) = transcode(UInt8, string(scope))
f_pstring(scope::String) = transcode(UInt8, string(scope))

# path to scalapack dynamic library
const libscalapack = "/home/k0324/k032403/.julia/packages/ScaLapack/lib/libscalapack.so"
# const libscalapack = "/usr/local/Cellar/scalapack/2.0.2_15/lib/libscalapack.dylib"
# depend on envionment
# const ScaInt = Int32
const ScaInt = Int64
const zero = convert(ScaInt, 0)
const one = convert(ScaInt, 1)

# array descriptor index
struct descriptor_index
    # params
    dtype::ScaInt
    ctxt::ScaInt
    m::ScaInt; n::ScaInt;
    mb::ScaInt; nb::ScaInt;
    rsrc::ScaInt; csrc::ScaInt;
    lld::ScaInt;
    # constructor
    descriptor_index() = new(1, 2, 3, 4, 5, 6, 7, 8, 9)
end
const desc_idx = descriptor_index()
