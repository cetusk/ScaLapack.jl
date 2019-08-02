
# ScaLapack.jl
module ScaLapack
    # preambles
    include("preambles.jl")
    # scalapack wrapper
    include("scalapack_wrappers.jl")

    # blacs wrapper
    module BLACS
        import ..libscalapack, ..ScaInt, ..f_pchar
        include("blacs.jl")
    end

    # pblas wrapper: planning add thus module ...
    # module PBLAS
    #     import ..libscalapack, ..ScaInt, ..f_pstring
    #     using ..BLACS
    #     include("pblas.jl")
    # end
    
    # Lite ver. of the ScaLapack
    module ScaLapackLite
        import ..libscalapack, ..ScaInt, ..f_pchar
        import ..zero, ..one, ..desc_idx
        using ..MPI, ..ScaLapack, ..BLACS
        # using ..PBLAS
        include("scalapack_lite.jl")
    end

end