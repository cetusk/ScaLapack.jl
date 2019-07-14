module BLACS
const BlaInt = Int64
f_pchar(scope::Char) = transcode(UInt8, string(scope))

import ..libscalapack

# https://software.intel.com/en-us/mkl-developer-reference-c-blacs-pinfo
# https://searchcode.com/file/21350024/blacs/BLACS-openmpi/SRC/MPI/blacs_pinfo_.c
# input : nothing
# output: rank, num of processes
function pinfo()
    mypnum, nprocs = zeros(BlaInt,1), zeros(BlaInt,1)
    ccall((:blacs_pinfo_, libscalapack), Nothing,
        (Ptr{BlaInt}, Ptr{BlaInt}),
        mypnum, nprocs)
    return mypnum[1], nprocs[1]
end

# https://software.intel.com/en-us/mkl-developer-reference-c-blacs-get
# https://searchcode.com/file/21350069/blacs/BLACS-openmpi/SRC/MPI/blacs_get_.c
# input : BLACS context ( MPI communicator )
#         BLACS indicator for BLACS contxt which should be returned
# output: BLACS context
function get(icontxt::Integer, what::Integer)
    val = zeros(BlaInt,1)
    ccall((:blacs_get_, libscalapack), Nothing,
        (Ptr{BlaInt}, Ptr{BlaInt}, Ptr{BlaInt}),
        Ref(icontxt), Ref(what), val)
    return val[1]
end

# https://software.intel.com/en-us/mkl-developer-reference-c-blacs-gridinit
# https://searchcode.com/file/21350091/blacs/BLACS-openmpi/SRC/MPI/blacs_gridinit_.c
# input : BLACS context
#         BLACS layout; how to map processes to BLACS grid
#         num of rows and cols in the process grid
# output: BLACS context
function gridinit(icontxt::Integer, layout::Char, nprow::Integer, npcol::Integer)
    ocontxt = Array{BlaInt,1}([icontxt])
    ccall((:blacs_gridinit_, libscalapack), Nothing,
        (Ptr{BlaInt}, Ptr{Char}, Ptr{BlaInt}, Ptr{BlaInt}),
        ocontxt, f_pchar(layout), Ref(nprow), Ref(npcol))
    return ocontxt[1]
end

# https://software.intel.com/en-us/mkl-developer-reference-c-blacs-gridinfo
# https://searchcode.com/file/21350044/blacs/BLACS-openmpi/SRC/MPI/blacs_gridinfo_.c
# input : BLACS context
# output: num of process rows and cols in the current process id
#         row and col coordinate in the current process id
function gridinfo(icontxt::Integer)
    nprow = zeros(BlaInt,1)
    npcol = zeros(BlaInt,1)
    myrow = zeros(BlaInt,1)
    mycol = zeros(BlaInt,1)
    ccall((:blacs_gridinfo_, libscalapack), Nothing,
        (Ptr{BlaInt}, Ptr{BlaInt}, Ptr{BlaInt}, Ptr{BlaInt}, Ptr{BlaInt}),
        Ref(icontxt), nprow, npcol, myrow, mycol)
    return nprow[1], npcol[1], myrow[1], mycol[1]
end

# https://software.intel.com/en-us/mkl-developer-reference-c-blacs-pnum
# https://searchcode.com/file/21350042/blacs/BLACS-openmpi/SRC/MPI/blacs_pnum_.c
# input : BLACS context
#         row and col coordinate in a process id
# output: process number
function pnum(icontxt::Integer, prow::Integer, pcol::Integer)
    return ccall((:blacs_pnum_, libscalapack), BlaInt,
        (Ptr{BlaInt}, Ptr{BlaInt}, Ptr{BlaInt}),
        Ref(icontxt), Ref(prow), Ref(pcol))
end

# https://software.intel.com/en-us/mkl-developer-reference-c-blacs-pcoord
# https://searchcode.com/file/21350056/blacs/BLACS-openmpi/SRC/MPI/blacs_pcoord_.c
# input : BLACS context, process number
# output: row and col coordinate in the pnum process id
function pcoord(icontxt::Integer, pnum::Integer)
    prow = zeros(BlaInt,1)
    pcol = zeros(BlaInt,1)
    ccall((:blacs_pcoord_, libscalapack), Nothing,
        (Ptr{BlaInt}, Ptr{BlaInt}, Ptr{BlaInt}, Ptr{BlaInt}),
        Ref(icontxt), Ref(pnum), prow, pcol)
    return prow[1], pcol[1]
end

# https://software.intel.com/en-us/mkl-developer-reference-c-blacs-barrier
# https://searchcode.com/file/21350053/blacs/BLACS-openmpi/SRC/MPI/blacs_barrier_.c
# input : BLACS context
#         BLACS parameter for row/col/all grid to participate barrier
# output: nothing
function barrier(icontxt::Integer, scope::Char)
    ccall((:blacs_barrier_, libscalapack), Nothing,
        (Ptr{BlaInt}, Ptr{Char}),
        Ref(icontxt), f_pchar(scope))
end

# https://software.intel.com/en-us/mkl-developer-reference-c-blacs-gridexit
# https://searchcode.com/file/21350036/blacs/BLACS-openmpi/SRC/MPI/blacs_gridexit_.c
# input : BLACS context
# output: nothing
gridexit(icontxt::Integer) = ccall((:blacs_gridexit_, libscalapack), Nothing, (Ptr{BlaInt},), Ref(icontxt))

# https://software.intel.com/en-us/mkl-developer-reference-c-blacs-exit
# https://searchcode.com/file/21349846/blacs/BLACS-openmpi/SRC/MPI/blacs_exit_.c
# input : BLACS flag whether continue message passing or not after done
# output: nothing
exit(continue_::Integer = 0) = ccall((:blacs_exit_, libscalapack), Nothing, (Ptr{BlaInt},), Ref(continue_))

end
