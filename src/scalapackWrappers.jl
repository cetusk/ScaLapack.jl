
const ScaInt = Int64
f_pchar(scope::Char) = transcode(UInt8, string(scope))

# input : num of rows and cols in the process grid
# output: BLACS context
function sl_init(nprow::Integer, npcol::Integer)
    ctxt = zeros(ScaInt,1)
    ccall((:sl_init_, libscalapack), Nothing,
        (Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}),
        ctxt, Ref(nprow), Ref(npcol))
    return ctxt[1]
end

# input : num of rows/cols, num of rows/cols in its block
#         num of process rows and cols in the current process id
#         head grid address of its process grid
#         num of process grid
# output: num of process rows and cols in the current process id
function numroc(n::Integer, nb::Integer, iproc::Integer, isrcproc::Integer, nprocs::Integer)
    return ccall((:numroc_, libscalapack), ScaInt,
                (Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}),
                Ref(n), Ref(nb), Ref(iproc), Ref(isrcproc), Ref(nprocs))
end

# input : num of rows/cols, num of rows/cols in its block
#         head grid address of its process grid
#         BLACS context, maximum local leading dimension ( mxlld )
# output: array descriptor
function descinit(m::Integer, n::Integer, mb::Integer, nb::Integer, irsrc::Integer, icsrc::Integer, ictxt::Integer, lld::Integer)

    # extract values
    nprow, npcol, myrow, mycol = BLACS.gridinfo(ictxt)
    locrm = numroc(m, mb, myrow, irsrc, nprow)

    # checks
    m >= 0 || throw(ArgumentError("first dimension must be non-negative"))
    n >= 0 || throw(ArgumentError("second dimension must be non-negative"))
    mb > 0 || throw(ArgumentError("first dimension blocking factor must be positive"))
    nb > 0 || throw(ArgumentError("second dimension blocking factor must be positive"))
    0 <= irsrc < nprow || throw(ArgumentError("process row must be positive and less that grid size"))
    0 <= icsrc < npcol || throw(ArgumentError("process column must be positive and less that grid size"))
    # lld >= locrm || throw(ArgumentError("leading dimension of local array is too small"))

    # allocation
    desc = zeros(ScaInt,9)
    info = zeros(ScaInt,1)

    # ccall
    ccall((:descinit_, libscalapack), Nothing,
        (Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt},
         Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt},
         Ptr{ScaInt}, Ptr{ScaInt}),
        desc, Ref(m), Ref(n), Ref(mb),
        Ref(nb), Ref(irsrc), Ref(icsrc), Ref(ictxt),
        Ref(lld), info)

    info[1] == 0 || error("input argument $(info[1]) has illegal value")

    return desc
end

# input : row/col index in the global array A indicating the first row of sub(A)
#         ( will be updated ! ) array descriptor for the distributed matrix A
#         scalar alpha which will be substituted into the A
# output: nothing
for (fname, elty) in ((:pselset_, :Float32),
                      (:pdelset_, :Float64),
                      (:pcelset_, :ComplexF32),
                      (:pzelset_, :ComplexF64))
    @eval begin
        function pXelset!(A::Matrix{$elty}, ia::Integer, ja::Integer, desca::Vector{ScaInt}, α::$elty)
            ccall(($(string(fname)), libscalapack), Nothing,
                (Ptr{$elty}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{$elty}),
                A, Ref(ia), Ref(ja), desca, Ref(α))
        end
    end
end

# input : BLACS scope in which alpha is returned
#         topology to be used if broadcast is needed
#         distributed matrix A
#         row/col index in the global array A
#         array descriptor for the distributed matrix A
# output: scalar alpha which will be returned from the A
for (fname, elty) in ((:pselget_, :Float32),
                      (:pdelget_, :Float64),
                      (:pcelget_, :ComplexF32),
                      (:pzelget_, :ComplexF64))
    @eval begin
        function pXelget(scope::Char, top::Char, A::Matrix{$elty}, ia::Integer, ja::Integer, desca::Vector{ScaInt})
            α = zeros($elty,1)
            ccall(($(string(fname)), libscalapack), Nothing,
                  (Ptr{Char}, Ptr{Char}, Ptr{$elty}, Ptr{$elty}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}),
                  f_pchar(scope), f_pchar(top), α, A, Ref(ia), Ref(ja), desca)
            return α[1]
        end
    end
end

# input : BLACS scope to operate; op(X) = X or X' or X*'
#         grid indices of the inputted global matrices
#         coefficients
#         local matrices, grid indices of the local matrices
#         array descriptors for the distributed matrices
# output: nothing
# === detail ===
# op(X) = X or X' or X*'
# sub(A) = A[ia:ia+m-1,ja:ja+k-1]
# sub(B) = B[ib:ib+k-1,jb:jb+n-1]
# sub(C) = C[ic:ic+m-1,jc:jc+n-1]
#        = α*op(sub(A))*op(sub(B))+β*sub(C)
# op(sub(A)) denotes A[ia:ia+m-1,ja:ja+k-1]   if transa = 'n',
#                    A[ia:ia+k-1,ja:ja+m-1]'  if transa = 't',
#                    A[ia:ia+k-1,ja:ja+m-1]*' if transa = 'c',
# op(sub(B)) denotes B[ib:ib+k-1,jb:jb+n-1]   if transb = 'n',
#                    B[ib:ib+n-1,jb:jb+k-1]'  if transb = 't',
#                    B[ib:ib+n-1,jb:jb+k-1]*' if transb = 'c',
for (fname, elty) in ((:psgemm_, :Float32),
                      (:pdgemm_, :Float64),
                      (:pcgemm_, :ComplexF32),
                      (:pzgemm_, :ComplexF64))
    @eval begin
        function pXgemm!(transa::Char, transb::Char,
                         m::Integer, n::Integer, k::Integer,
                         α::$elty,
                         A::Matrix{$elty}, ia::Integer, ja::Integer, desca::Vector{ScaInt},
                         B::Matrix{$elty}, ib::Integer, jb::Integer, descb::Vector{ScaInt},
                         β::$elty,
                         C::Matrix{$elty}, ic::Integer, jc::Integer, descc::Vector{ScaInt})

            ccall(($(string(fname)), libscalapack), Nothing,
                (Ptr{Char}, Ptr{Char},
                 Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt},
                 Ptr{$elty},
                 Ptr{$elty}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt},
                 Ptr{$elty}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt},
                 Ptr{$elty},
                 Ptr{$elty}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}),
                 f_pchar(transa), f_pchar(transb),
                 Ref(m), Ref(n), Ref(k),
                 Ref(α),
                 A, Ref(ia), Ref(ja), desca, 
                 B, Ref(ib), Ref(jb), descb,
                 Ref(β),
                 C, Ref(ic), Ref(jc), descc)
        end
    end
end
