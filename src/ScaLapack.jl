# using MPI

module ScaLapack

using Compat
using LinearAlgebra: BlasFloat, BlasReal

using MPI
using DistributedArrays, Distributed
import DistributedArrays: DArray, defaultdist

# this should only be a temporary solution until procs returns a type that encodes more information about the processes
DArray(init, dims, manager::MPIManager, args...) = DArray(init, dims, collect(values(manager.mpi2j))[sortperm(collect(keys(manager.mpi2j)))], args...)
function defaultdist(sz::Integer, nc::Integer)
    if sz >= nc
        d, r = divrem(sz, nc)
        if r == 0
            return vcat(1:d:sz+1)
        end
        return vcat(vcat(1:d+1:sz+1), [sz+1])
    else
        return vcat(vcat(1:(sz+1)), zeros(Int, nc-sz))
    end
end


if myid() > 1
    MPI.Initialized() || MPI.Init()
end

struct ScaLapackException <: Exception
    info::Integer
end

const libscalapack = "/usr/local/lib/libscalapack.dylib"

include("blacs.jl")
include("scalapackWrappers.jl")

end # module
