using InteractiveIsing
using FileIO, JLD2

const simulation = IsingSim(loadGraph())

const g = simulation(true);