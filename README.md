# Julia-Interactive-Ising-MMC

An interactive classical spin ising MCMC simulation.

The simulation can be run by executing main.jl

# Installation

Install Julia on your local machine.

This program relies on the QT6 branch of the [QML.jl package](https://github.com/barche/QML.jl)
To add this use the Julia package manager (Pkg) and type "add QML#qt6".

For Apple arm machines, there is a bug in a dependency of the QML package, CxxWrap. This is fixed in the main branch. To add this, "add CxxWrap#main".

Next, be sure that all packages that are used and included in the "Sim.jl" file are in your environment.

Then, make sure that Julia is using multiple threads. Minimum of 4 is recommended. For pure execution in the REPL start julia in the following way: "julia --threads n" where n is the amount of threads you want Julia to use. Then, from the REPL, either cd to the folder with all Julia files, or get the full path to the main file and include it. E.g.: "include("path/to/main.jl").

For use in VSCode, open VSCode in the folder with all simulation files. Then, in the Julia extension (needs to be installed) settings, change the option "julia.NumThreads" to number of cores that you want Julia to use. Then, with the main.jl file open, execute the file using the triangular button in the top right corner.

For faster startup time put [Sysimage](https://www.icloud.com/iclouddrive/017m3Oe8w0xq1AFyS_evZzc-w#JuliaSysimage) into environment folder.

# Usage

If all packages are installed correctly, then executing the "Main.jl" file starts up the simulation interface. 

To use custom weighting functions, define a const global variable named weighted = true and define a new instance of the type WeightFunc in the "CustomWF.jl) file and then alias it as "weightFunc" in the main file. The correct way do define such a struct is documented in "WeightFuncs.jl".
