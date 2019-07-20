
# input : num of rows and cols in the process grid
# output: BLACS context
function sl_init(nprow::ScaInt, npcol::ScaInt)
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
function numroc(n::ScaInt, nb::ScaInt, iproc::ScaInt, isrcproc::ScaInt, nprocs::ScaInt)
    return ccall((:numroc_, libscalapack), ScaInt,
                (Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}),
                Ref(n), Ref(nb), Ref(iproc), Ref(isrcproc), Ref(nprocs))
end

# input : num of rows/cols, num of rows/cols in its block
#         head grid address of its process grid
#         BLACS context, maximum local leading dimension ( mxlld )
# output: array descriptor
function descinit(m::ScaInt, n::ScaInt, mb::ScaInt, nb::ScaInt, irsrc::ScaInt, icsrc::ScaInt, ictxt::ScaInt, lld::ScaInt)
    desc = zeros(ScaInt,9)
    info = zeros(ScaInt,1)
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
        function pXelset!(A::Matrix{$elty}, ia::ScaInt, ja::ScaInt, desca::Vector{ScaInt}, α::$elty)
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
        function pXelget(scope::Char, top::Char, A::Matrix{$elty}, ia::ScaInt, ja::ScaInt, desca::Vector{ScaInt})
            α = zeros($elty,1)
            ccall(($(string(fname)), libscalapack), Nothing,
                  (Ptr{Char}, Ptr{Char}, Ptr{$elty}, Ptr{$elty}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}),
                  f_pchar(scope), f_pchar(top), α, A, Ref(ia), Ref(ja), desca)
            return α[1]
        end
    end
end

# input : num of rows/cols
#         local matrix A
#         head grid address of its process grid
#         array descriptor
#         ( will be updated ! ) matrix B
#         BLACS context
# output: Nothing
for (fname, elty) in ((:psgemr2d_, :Float32),
                      (:pdgemr2d_, :Float64),
                      (:pcgemr2d_, :ComplexF32),
                      (:pzgemr2d_, :ComplexF64))
    @eval begin
        function pXgemr2d!(m::ScaInt, n::ScaInt,
                           A::Matrix{$elty}, ia::ScaInt, ja::ScaInt, desca::Vector{ScaInt},
                           B::Matrix{$elty}, ib::ScaInt, jb::ScaInt, descb::Vector{ScaInt},
                           ictxt::ScaInt)
            ccall(($(string(fname)), libscalapack), Nothing,
                 (Ptr{ScaInt}, Ptr{ScaInt}, Ptr{$elty}, Ptr{ScaInt},
                  Ptr{ScaInt}, Ptr{ScaInt}, Ptr{$elty}, Ptr{ScaInt},
                  Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}),
                  Ref(m), Ref(n),
                  A, Ref(ia), Ref(ja), desca,
                  B, Ref(ib), Ref(jb), descb,
                  Ref(ictxt))
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
                         m::ScaInt, n::ScaInt, k::ScaInt,
                         α::$elty,
                         A::Matrix{$elty}, ia::ScaInt, ja::ScaInt, desca::Vector{ScaInt},
                         B::Matrix{$elty}, ib::ScaInt, jb::ScaInt, descb::Vector{ScaInt},
                         β::$elty,
                         C::Matrix{$elty}, ic::ScaInt, jc::ScaInt, descc::Vector{ScaInt})

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

