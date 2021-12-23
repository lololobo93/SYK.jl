include("./tensor.jl")
using LinearAlgebra
using CUDA
#module majorana
# using tensor
#export majorana

function majorana(N::Int)
    #MAJORANA creates a representation of N majorana fermions
    #   Creates N Majorana fermions in terms of spin chain variables (Pauli
    #   sigma matrices). The only catch is that they satisfy the Dirac algebra
    #   {Xi_i, Xi_j} = delta_{ij}. This is basically Jordan-Wignering.
    #
    #   N.B. the Hilbert space is 2^(N/2) dimensional
    #
    #   Outputs:
    #   Xi is an 2^(N/2) x 2^(N/2) x N array.
    #   You access the ith fermion as Xi(:,:,i)
    #
    #   Inputs:
    #   N is the number of majorana fermions. (Twice the number of qubits)
    #       thus, N should be even

    if N % 2 != 0
        error("N must be even")
    end

    #SU(2) Matrices
    X = [0 1; 1 0]

    Y = [0 -1im; 1im 0]

    Z = [1 0; 0 -1]

    # output of majoranas here
    dim = Int(2^(N/2))
    Xi = zeros(Complex, (dim, dim, N))

    # growing chain of Xs, start with 0
    Xs = Matrix{Complex}(I, dim, dim)

    for i in 1:N
        # the Y or Z at the end
        if i % 2 == 1
            xi = tensor(Z, (i+1)/2 , N/2)
        else
            xi = tensor(Y, i/2 , N/2)
        end

        @inbounds Xi[:,:,i] = Xs*xi

        # build an increment chains of X's
        if i % 2 == 0
            Xs = Xs*tensor(X, i/2, N/2)
        end
    end
    # right now, it's normalized {Xi_i, Xi_j} = 2*delta_{ij}
    # but Kitaev's normalization doens't have the 2
    Xi = Xi / sqrt(2)

    return Xi
end

function majoranaCu(N::Int)
    #MAJORANA creates a representation of N majorana fermions
    #   Creates N Majorana fermions in terms of spin chain variables (Pauli
    #   sigma matrices). The only catch is that they satisfy the Dirac algebra
    #   {Xi_i, Xi_j} = delta_{ij}. This is basically Jordan-Wignering.
    #
    #   N.B. the Hilbert space is 2^(N/2) dimensional
    #
    #   Outputs:
    #   Xi is an 2^(N/2) x 2^(N/2) x N array.
    #   You access the ith fermion as Xi(:,:,i)
    #
    #   Inputs:
    #   N is the number of majorana fermions. (Twice the number of qubits)
    #       thus, N should be even

    if N % 2 != 0
        error("N must be even")
    end

    #SU(2) Matrices
    X = cu([0 1; 1 0])

    Y = cu([0 -1im; 1im 0])

    Z = cu([1 0; 0 -1])

    # output of majoranas here
    dim = Int(2^(N/2))
    Xi = CUDA.zeros(ComplexF32, (dim, dim, N))

    # growing chain of Xs, start with 0
    Xs = cu(Matrix{ComplexF32}(I, dim, dim))

    for i in 1:N
        # the Y or Z at the end
        if i % 2 == 1
            xi = cu(tensorCu(Z, (i+1)/2 , N/2))
        else
            xi = cu(tensorCu(Y, i/2 , N/2))
        end

        @inbounds Xi[:,:,i] = Xs*xi

        # build an increment chains of X's
        if i % 2 == 0
            Xs = Xs*cu(tensorCu(X, i/2, N/2))
        end
    end
    # right now, it's normalized {Xi_i, Xi_j} = 2*delta_{ij}
    # but Kitaev's normalization doens't have the 2
    Xi = Xi / sqrt(2)

    return Xi
end
#end
