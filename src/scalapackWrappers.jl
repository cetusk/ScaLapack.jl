# typealias ScaInt Int32 # Fixme! Have to find a way of detecting if this is always the case
# const ScaInt = Int32
const ScaInt = Int64

#############
# Auxiliary #
#############

# Initialize
function sl_init(nprow::Integer, npcol::Integer)
    ictxt = ScaInt[0]
    ccall((:sl_init_, libscalapack), Nothing,
        (Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}),
        ictxt, Ref(nprow), Ref(npcol))
    return ictxt[1]
end

# Calculate size of local array
function numroc(n::Integer, nb::Integer, iproc::Integer, isrcproc::Integer, nprocs::Integer)
    ccall((:numroc_, libscalapack), ScaInt,
        (Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}),
        Ref(n), Ref(nb), Ref(iproc), Ref(isrcproc), Ref(nprocs))
end

# Array descriptor
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
    0 <= irsrc < nprow || throw(ArgumentError("process column must be positive and less that grid size"))
    # lld >= locrm || throw(ArgumentError("leading dimension of local array is too small"))

    # allocation
    desc = ScaInt[ 0 for j=1:9 ]
    info = ScaInt[0]

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

# Redistribute arrays
for (fname, elty) in ((:psgemr2d_, :Float32),
                      (:pdgemr2d_, :Float64),
                      (:pcgemr2d_, :ComplexF32),
                      (:pzgemr2d_, :ComplexF64))

    @eval begin
        function pxgemr2d!(m::Integer, n::Integer, A::Matrix{$elty}, ia::Integer, ja::Integer, desca::Vector{ScaInt}, B::Matrix{$elty}, ib::Integer, jb::Integer, descb::Vector{ScaInt}, ictxt::Integer)

            ccall((eval(string($fname)), libscalapack), Nothing,
                (Ptr{ScaInt}, Ptr{ScaInt}, Ptr{$elty}, Ptr{ScaInt},
                 Ptr{ScaInt}, Ptr{ScaInt}, Ptr{$elty}, Ptr{ScaInt},
                 Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}),
                 Ref(m), Ref(n), A, Ref(ia),
                 Ref(ja), desca, B, Ref(ib),
                 Ref(jb), descb, Ref(ictxt))
        end
    end
end

##################
# Linear Algebra #
##################

# Matmul
for (fname, elty) in ((:psgemm_, :Float32),
                      (:pdgemm_, :Float64),
                      (:pcgemm_, :ComplexF32),
                      (:pzgemm_, :ComplexF64))
    @eval begin
        function pdgemm!(transa::Char, transb::Char, m::Integer, n::Integer, k::Integer, α::$elty, A::Matrix{$elty}, ia::Integer, ja::Integer, desca::Vector{ScaInt}, B::Matrix{$elty}, ib::Integer, jb::Integer, descb::Vector{ScaInt}, β::$elty, C::Matrix{$elty}, ic::Integer, jc::Integer, descc::Vector{ScaInt})

            ccall(($(string(fname)), libscalapack), Nothing,
                (Ptr{Char}, Ptr{Char}, Ptr{ScaInt}, Ptr{ScaInt},
                 Ptr{ScaInt}, Ptr{$elty}, Ptr{$elty}, Ptr{ScaInt},
                 Ptr{ScaInt}, Ptr{ScaInt}, Ptr{$elty}, Ptr{ScaInt},
                 Ptr{ScaInt}, Ptr{ScaInt}, Ptr{$elty}, Ptr{$elty},
                 Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}),
                 Ref(transa), Ref(transb), Ref(m), Ref(n),
                 Ref(k), Ref(α), A, Ref(ia),
                 Ref(ja), desca, B, Ref(ib),
                 Ref(jb), descb, Ref(β), C,
                 Ref(ic), Ref(jc), descc)
        end
    end
end

# Eigensolves
for (fname, elty) in ((:psstedc_, :Float32),
                      (:pdstedc_, :Float64))
    @eval begin
        function pxstedc!(compz::Char, n::Integer, d::Vector{$elty}, e::Vector{$elty}, Q::Matrix{$elty}, iq::Integer, jq::Integer, descq::Vector{ScaInt})


            work    = $elty[0]
            lwork   = convert(ScaInt, -1)
            iwork   = ScaInt[0]
            liwork  = convert(ScaInt, -1)
            info    = ScaInt[0]

            for i = 1:2
                ccall(($(string(fname)), libscalapack), Nothing,
                    (Ptr{Char}, Ptr{Char}, Ptr{$elty}, Ptr{$elty},
                     Ptr{$elty}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt},
                     Ptr{$elty}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt},
                     Ptr{$ScaInt}),
                     Ref(compz), Ref(n), d, e,
                    Q, Ref(iq), Ref(jq), descq,
                    work, Ref(lwork), iwork, Ref(liwork),
                    info)

                if i == 1
                    lwork = convert(ScaInt, work[1])
                    work = $elty[ 0 for j=1:lwork ]
                    liwork = convert(ScaInt, iwork[1])
                    iwork = ScaInt[ 0 for j=1:liwork ]
                end
            end

            return d, Q
        end
    end
end

# SVD solver
for (fname, elty) in ((:psgesvd_, :Float32),
                      (:pdgesvd_, :Float64))
# for (fname, elty) in ((:PSGESVD_, :Float32),
#                       (:PDGESVD_, :Float64))
  
    @eval begin
        function pxgesvd!(jobu::Char, jobvt::Char, m::Integer, n::Integer, A::Matrix{$elty}, ia::Integer, ja::Integer, desca::Vector{ScaInt}, s::Vector{$elty}, U::Matrix{$elty}, iu::Integer, ju::Integer, descu::Vector{ScaInt}, Vt::Matrix{$elty}, ivt::Integer, jvt::Integer, descvt::Vector{ScaInt})
            # extract values

            # allocate
            info = ScaInt[0]
            work = $elty[0]
            lwork = -1

            # ccall
            for i = 1:2
                ccall(($(string(fname)), libscalapack), Nothing,
                    (Ptr{Char}, Ptr{Char}, Ptr{ScaInt}, Ptr{ScaInt},
                     Ptr{$elty}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt},
                     Ptr{$elty}, Ptr{$elty}, Ptr{ScaInt}, Ptr{ScaInt},
                     Ptr{ScaInt}, Ptr{$elty}, Ptr{ScaInt}, Ptr{ScaInt},
                     Ptr{ScaInt}, Ptr{$elty}, Ptr{ScaInt}, Ptr{ScaInt}),
                    Ref(jobu), Ref(jobvt), Ref(m), Ref(n),
                    A, Ref(ia), Ref(ja), desca,
                    s, U, Ref(iu), Ref(ju),
                    descu, Vt, Ref(ivt), Ref(jvt),
                    descvt, work, Ref(lwork), info)
                if i == 1
                    lwork = convert(ScaInt, work[1])
                    work = $elty[ 0 for j=1:lwork ]
                end
            end

            if 0 < info[1] <= min(m,n)
                throw(ScaLapackException(info[1]))
            end

            return U, s, Vt
        end
    end
end
for (fname, elty, relty) in ((:pcgesvd_, :ComplexF32, :Float32),
                             (:pzgesvd_, :ComplexF64, :Float64))
    @eval begin
        function pxgesvd!(jobu::Char, jobvt::Char, m::Integer, n::Integer, A::Matrix{$elty}, ia::Integer, ja::Integer, desca::Vector{ScaInt}, s::Vector{$relty}, U::Matrix{$elty}, iu::Integer, ju::Integer, descu::Vector{ScaInt}, Vt::Matrix{$elty}, ivt::Integer, jvt::Integer, descvt::Vector{ScaInt})
            # extract values

            # allocate
            work = $elty[0]
            lwork = -1
            rwork = $relty[ 0 for j=1:(1 + 4*max(m, n)) ]
            info = ScaInt[0]

            # ccall
            for i = 1:2
                ccall(($(string(fname)), libscalapack), Nothing,
                    (Ptr{Char}, Ptr{Char}, Ptr{ScaInt}, Ptr{ScaInt},
                     Ptr{$elty}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt},
                     Ptr{$relty}, Ptr{$elty}, Ptr{ScaInt}, Ptr{ScaInt},
                     Ptr{ScaInt}, Ptr{$elty}, Ptr{ScaInt}, Ptr{ScaInt},
                     Ptr{ScaInt}, Ptr{$elty}, Ptr{ScaInt}, Ptr{$relty},
                     Ptr{ScaInt}),
                    Ref(jobu), Ref(jobvt), Ref(m), Ref(n),
                    A, Ref(ia), Ref(ja), desca,
                    s, U, Ref(iu), Ref(ju),
                    descu, Vt, Ref(ivt), Ref(jvt),
                    descvt, work, Ref(lwork), rwork,
                    info)
                if i == 1
                    lwork = convert(ScaInt, work[1])
                    work = $elty[ 0 for j=1:lwork ]
                end
            end

            info[1] > 0 && throw(ScaLapackException(info[1]))

            return U, s, Vt
        end
    end
end

