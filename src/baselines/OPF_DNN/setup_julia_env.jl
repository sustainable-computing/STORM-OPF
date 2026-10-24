\
# setup_julia_env.jl
# Creates a Julia project at ./jl and installs pinned package versions.
import Pkg

# Ensure we're inside the conda env so PyCall points at its Python
if !haskey(ENV, "CONDA_PREFIX")
    error("Please 'conda activate pyjl-evhc' first so CONDA_PREFIX is set.")
end

# Point PyCall at this conda env's Python
pyexe = joinpath(ENV["CONDA_PREFIX"], Sys.iswindows() ? "python.exe" : joinpath("bin", "python"))
ENV["PYTHON"] = pyexe

# Activate/create local Julia project at ./jl
Pkg.activate("jl"; shared=false)

# Add packages with exact versions where requested
Pkg.add([
    Pkg.PackageSpec(name="PowerModels",     version="0.15.1"),
    Pkg.PackageSpec(name="JuMP",            version="0.21.8"),
    Pkg.PackageSpec(name="Ipopt",           version="0.6.5"),
    Pkg.PackageSpec(name="Distributions"),
    Pkg.PackageSpec(name="JSON"),
    Pkg.PackageSpec(name="ProgressMeter",   version="1.7.1"),
    Pkg.PackageSpec(name="ArgParse",        version="1.1.4"),
    Pkg.PackageSpec(name="PyCall",          version="1.92.3"),
    #Pkg.PackageSpec(name="PyPlot"),
])

Pkg.precompile()

# Sanity checks / printouts
using PyCall
println("PyCall is using Python at: ", PyCall.python)

# Verify core packages load
using PowerModels, JuMP, Ipopt, Distributions, JSON, ProgressMeter, ArgParse
println("Julia package setup looks good.")
