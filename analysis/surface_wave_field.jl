using MAT
using GLMakie

dir = "../data"
filename = "ETAT_R2_allexp.mat"
filepath = joinpath(dir, filename)
vars = matread(filepath)
