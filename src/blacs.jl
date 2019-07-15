
# https://software.intel.com/en-us/mkl-developer-reference-c-blacs-pinfo
# https://searchcode.com/file/21350024/blacs/BLACS-openmpi/SRC/MPI/blacs_pinfo_.c
# input : nothing
# output: rank, num of processes
function pinfo()
    mypnum, nprocs = zeros(ScaInt,1), zeros(ScaInt,1)
    ccall((:blacs_pinfo_, libscalapack), Nothing,
        (Ptr{ScaInt}, Ptr{ScaInt}),
        mypnum, nprocs)
    return mypnum[1], nprocs[1]
end

# https://software.intel.com/en-us/mkl-developer-reference-c-blacs-get
# https://searchcode.com/file/21350069/blacs/BLACS-openmpi/SRC/MPI/blacs_get_.c
# input : BLACS context ( MPI communicator )
#         BLACS indicator for BLACS contxt which should be returned
# output: BLACS context
function get(icontxt::ScaInt, what::ScaInt)
    val = zeros(ScaInt,1)
    ccall((:blacs_get_, libscalapack), Nothing,
        (Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}),
        Ref(icontxt), Ref(what), val)
    return val[1]
end

# https://software.intel.com/en-us/mkl-developer-reference-c-blacs-gridinit
# https://searchcode.com/file/21350091/blacs/BLACS-openmpi/SRC/MPI/blacs_gridinit_.c
# input : BLACS context
#         BLACS layout; how to map processes to BLACS grid
#         num of rows and cols in the process grid
# output: BLACS context
function gridinit(icontxt::ScaInt, layout::Char, nprow::ScaInt, npcol::ScaInt)
    ocontxt = Array{ScaInt,1}([icontxt])
    ccall((:blacs_gridinit_, libscalapack), Nothing,
        (Ptr{ScaInt}, Ptr{Char}, Ptr{ScaInt}, Ptr{ScaInt}),
        ocontxt, f_pchar(layout), Ref(nprow), Ref(npcol))
    return ocontxt[1]
end

# https://software.intel.com/en-us/mkl-developer-reference-c-blacs-gridinfo
# https://searchcode.com/file/21350044/blacs/BLACS-openmpi/SRC/MPI/blacs_gridinfo_.c
# input : BLACS context
# output: num of process rows and cols in the current process id
#         row and col coordinate in the current process id
function gridinfo(icontxt::ScaInt)
    nprow = zeros(ScaInt,1)
    npcol = zeros(ScaInt,1)
    myrow = zeros(ScaInt,1)
    mycol = zeros(ScaInt,1)
    ccall((:blacs_gridinfo_, libscalapack), Nothing,
        (Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}),
        Ref(icontxt), nprow, npcol, myrow, mycol)
    return nprow[1], npcol[1], myrow[1], mycol[1]
end

# https://software.intel.com/en-us/mkl-developer-reference-c-blacs-pnum
# https://searchcode.com/file/21350042/blacs/BLACS-openmpi/SRC/MPI/blacs_pnum_.c
# input : BLACS context
#         row and col coordinate in a process id
# output: process number
function pnum(icontxt::ScaInt, prow::ScaInt, pcol::ScaInt)
    return ccall((:blacs_pnum_, libscalapack), ScaInt,
        (Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}),
        Ref(icontxt), Ref(prow), Ref(pcol))
end

# https://software.intel.com/en-us/mkl-developer-reference-c-blacs-pcoord
# https://searchcode.com/file/21350056/blacs/BLACS-openmpi/SRC/MPI/blacs_pcoord_.c
# input : BLACS context, process number
# output: row and col coordinate in the pnum process id
function pcoord(icontxt::ScaInt, pnum::ScaInt)
    prow = zeros(ScaInt,1)
    pcol = zeros(ScaInt,1)
    ccall((:blacs_pcoord_, libscalapack), Nothing,
        (Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}),
        Ref(icontxt), Ref(pnum), prow, pcol)
    return prow[1], pcol[1]
end

# https://software.intel.com/en-us/mkl-developer-reference-c-blacs-barrier
# https://searchcode.com/file/21350053/blacs/BLACS-openmpi/SRC/MPI/blacs_barrier_.c
# input : BLACS context
#         BLACS parameter for row/col/all grid to participate barrier
# output: nothing
function barrier(icontxt::ScaInt, scope::Char)
    ccall((:blacs_barrier_, libscalapack), Nothing,
        (Ptr{ScaInt}, Ptr{Char}),
        Ref(icontxt), f_pchar(scope))
end

# https://software.intel.com/en-us/mkl-developer-reference-c-blacs-gridexit
# https://searchcode.com/file/21350036/blacs/BLACS-openmpi/SRC/MPI/blacs_gridexit_.c
# input : BLACS context
# output: nothing
gridexit(icontxt::ScaInt) = ccall((:blacs_gridexit_, libscalapack), Nothing, (Ptr{ScaInt},), Ref(icontxt))

# https://software.intel.com/en-us/mkl-developer-reference-c-blacs-exit
# https://searchcode.com/file/21349846/blacs/BLACS-openmpi/SRC/MPI/blacs_exit_.c
# input : BLACS flag whether continue message passing or not after done
# output: nothing
exit(continue_::ScaInt = 0) = ccall((:blacs_exit_, libscalapack), Nothing, (Ptr{ScaInt},), Ref(continue_))
