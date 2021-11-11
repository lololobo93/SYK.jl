using Statistics
using DelimitedFiles
include("./majorana.jl")
include("./kitaev_H.jl")
#using majorana
#using kitaev_H

function remove_matrix_elements_randomly(H, dim, num_remove)
    for x = 1:num_remove
        i, j = rand(1:dim, 2)
        H[i, j] = 0
    end
end

function get_spectral_form_factor(energies, beta)
    # Z is partition function
    Z = sum([exp(-beta * E) for E in energies])
    return real(conj(Z) * Z)
end

# This code generates, diagonalizes and stores SYK eigenvalues
# size parameters
N = 10  # set number of Majorana fermions
numH = 5120  # set number of samples
Htype = "SYK"  # type of the Hamiltonian used
matrix_size = Int(2^(N/2))

# SYK parameter
J = 1

# get Majorana fermions
Xi = majorana(N)

eig_saves = zeros(Float64, matrix_size, numH)
for i in 1:numH
    H = kitaev_H(J, Xi, N) # creates the Hamiltonian
    # remove_matrix_elements_randomly(H, matrix_size, 20)  # make it sparse
    VV = eigvals(H)  # diagonalizes the Hamiltonian
    eig_saves[:, i] .= VV
end
iofile = string("eig_saves/EIG_SYK_N", N, "_it", numH, ".dat")
iobuf = open(iofile, "w")
writedlm(iobuf,
            eig_saves)
close(iobuf)
