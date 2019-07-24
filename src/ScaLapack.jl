
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

    # Lite ver. of the ScaLapack
    module ScaLapackLite
        import ..libscalapack, ..ScaInt, ..f_pchar
        using ..LinearAlgebra, ..MPI, ..ScaLapack, ..BLACS
        include("scalapack_lite.jl")
    end

end