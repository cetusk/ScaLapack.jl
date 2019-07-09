module BLACS

const BlaInt = Int64

import ..libscalapack

function get(icontxt::Integer, what::Integer)
    val = zeros(BlaInt,1)
    ccall((:blacs_get_, libscalapack), Nothing,
        (Ptr{BlaInt}, Ptr{BlaInt}, Ptr{BlaInt}),
        Ref(icontxt), Ref(what), val)
    return val[1]
end

function gridinit(icontxt::Integer, order::Char, nprow::Integer, npcol::Integer)
    icontxta = zeros(BlaInt,icontxt)
    ccall((:blacs_gridinit_, libscalapack), Nothing,
        (Ptr{BlaInt}, Ptr{Char}, Ptr{BlaInt}, Ptr{BlaInt}),
        icontxta, Ref(order), Ref(nprow), Ref(npcol))
    icontxta[1]
end

function pinfo()
    mypnum, nprocs = zeros(BlaInt,1), zeros(BlaInt,1)
    ccall((:blacs_pinfo_, libscalapack), Nothing,
        (Ptr{BlaInt}, Ptr{BlaInt}),
        mypnum, nprocs)
    return mypnum[1], nprocs[1]
end

function gridinfo(ictxt::Integer)
    nprow = zeros(BlaInt,1)
    npcol = zeros(BlaInt,1)
    myprow = zeros(BlaInt,1)
    mypcol = zeros(BlaInt,1)
    ccall((:blacs_gridinfo_, libscalapack), Nothing,
        (Ptr{BlaInt}, Ptr{BlaInt}, Ptr{BlaInt}, Ptr{BlaInt}, Ptr{BlaInt}),
        Ref(ictxt), nprow, npcol, myprow, mypcol)
    return nprow[1], npcol[1], myprow[1], mypcol[1]
end

gridexit(ictxt::Integer) = ccall((:blacs_gridexit_, libscalapack), Nothing, (Ptr{BlaInt},), Ref(ictxt))

exit(cont = 0) = ccall((:blacs_exit_, libscalapack), Nothing, (Ptr{BlaInt},), Ref(cont))

end
