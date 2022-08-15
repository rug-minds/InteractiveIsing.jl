# InteractiveIsing.jl
WIP

An interactive Spin Ising simulation using Julia and QML

Using the user interface requires multithreading. Either set up multithreading directly in VSCODE, Pluto, etc. or when running through the REPL, make sure the environment variable `JULIA_NUM_THREADS`
is set.

This can be done by running 
```
touch ~/.zshrc; open ~/.zshrc
```
and then writing
```
export JULIA_NUM_THREADS=X
```
where X is the number of threads you want to use (should be at least 4) and saving. The terminal will now start with the environment variable set correctly.

