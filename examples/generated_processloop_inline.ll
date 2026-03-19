; Function Signature: generated_processloop(InteractiveIsing.Processes.InlineProcess{InteractiveIsing.Processes.TaskData{InteractiveIsing.Processes.CompositeAlgorithm{Tuple{InteractiveIsing.Processes.IdentifiableAlgo{InteractiveIsing.Metropolis, Base.UUID(value=UInt128(0x2e51cc537efa45449e8e2b1f0e2f4880)), InteractiveIsing.Processes.VarAliases{NamedTuple(), NamedTuple()}(), nothing, :Metropolis_1}}, InteractiveIsing.Processes.RepeatOne{1}, Tuple{}, Tuple{}, nothing}, Tuple{InteractiveIsing.Processes.NamedInput{:Metropolis_1, NamedTuple{(:state,), Tuple{InteractiveIsing.IsingGraph{Float32, InteractiveIsing.UndirectedAdjacency{Float32, Int32, SparseArrays.SparseMatrixCSC{Float32, Int32}, InteractiveIsing.OffsetArray{Int64, 1, Array{Int64, 1}}, Nothing}, Tuple{InteractiveIsing.IsingLayerData{InteractiveIsing.Discrete(), (-1f0, 1f0), 2, (100, 100), Base.UnitRange{Int64}(start=1, stop=10000), InteractiveIsing.SquareTopology{InteractiveIsing.Periodic, 2, Float64}}}, Array{Float32, 1}, 1, Base.UnitRange{Int64}}}}}}, Tuple{}, InteractiveIsing.Processes.ProcessContext{NamedTuple{(:Metropolis_1, :globals), Tuple{InteractiveIsing.Processes.SubContext{:Metropolis_1, NamedTuple{(), Tuple{}}, Tuple{}, Tuple{}}, NamedTuple{(:lifetime, :algo), Tuple{InteractiveIsing.Processes.Repeat, InteractiveIsing.Processes.CompositeAlgorithm{Tuple{InteractiveIsing.Processes.IdentifiableAlgo{InteractiveIsing.Metropolis, Base.UUID(value=UInt128(0x2e51cc537efa45449e8e2b1f0e2f4880)), InteractiveIsing.Processes.VarAliases{NamedTuple(), NamedTuple()}(), nothing, :var""}}, InteractiveIsing.Processes.RepeatOne{1}, Tuple{}, Tuple{}, nothing}}}}}, InteractiveIsing.Processes.NameSpaceRegistry{Tuple{InteractiveIsing.Processes.RegistryTypeEntry{InteractiveIsing.Metropolis, Tuple{InteractiveIsing.Processes.IdentifiableAlgo{InteractiveIsing.Metropolis, Base.UUID(value=UInt128(0x2e51cc537efa45449e8e2b1f0e2f4880)), InteractiveIsing.Processes.VarAliases{NamedTuple(), NamedTuple()}(), nothing, :Metropolis_1}}}}}}, InteractiveIsing.Processes.Repeat}, InteractiveIsing.Processes.ProcessContext{NamedTuple{(:Metropolis_1, :globals), Tuple{InteractiveIsing.Processes.SubContext{:Metropolis_1, NamedTuple{(:state, :hamiltonian, :proposer, :rng, :proposal, :ΔE, :T), Tuple{InteractiveIsing.IsingGraph{Float32, InteractiveIsing.UndirectedAdjacency{Float32, Int32, SparseArrays.SparseMatrixCSC{Float32, Int32}, InteractiveIsing.OffsetArray{Int64, 1, Array{Int64, 1}}, Nothing}, Tuple{InteractiveIsing.IsingLayerData{InteractiveIsing.Discrete(), (-1f0, 1f0), 2, (100, 100), Base.UnitRange{Int64}(start=1, stop=10000), InteractiveIsing.SquareTopology{InteractiveIsing.Periodic, 2, Float64}}}, Array{Float32, 1}, 1, Base.UnitRange{Int64}}, InteractiveIsing.HamiltonianTerms{Tuple{InteractiveIsing.PolynomialHamiltonian{2, InteractiveIsing.UniformArray{Float32, 0}, InteractiveIsing.OffsetArray{Int64, 1, Array{Int64, 1}}}, InteractiveIsing.Bilinear{InteractiveIsing.UndirectedAdjacency{Float32, Int32, SparseArrays.SparseMatrixCSC{Float32, Int32}, InteractiveIsing.OffsetArray{Int64, 1, Array{Int64, 1}}, Nothing}}, InteractiveIsing.MagField{InteractiveIsing.ConstFill{0f0, Float32, 2}}}}, InteractiveIsing.IsingGraphProposer{Base.UnitRange{Int64}, Tuple{InteractiveIsing.IsingLayer{InteractiveIsing.Discrete(), 2, InteractiveIsing.IsingLayerData{InteractiveIsing.Discrete(), (-1f0, 1f0), 2, (100, 100), Base.UnitRange{Int64}(start=1, stop=10000), InteractiveIsing.SquareTopology{InteractiveIsing.Periodic, 2, Float64}}, InteractiveIsing.IsingGraph{Float32, InteractiveIsing.UndirectedAdjacency{Float32, Int32, SparseArrays.SparseMatrixCSC{Float32, Int32}, InteractiveIsing.OffsetArray{Int64, 1, Array{Int64, 1}}, Nothing}, Tuple{InteractiveIsing.IsingLayerData{InteractiveIsing.Discrete(), (-1f0, 1f0), 2, (100, 100), Base.UnitRange{Int64}(start=1, stop=10000), InteractiveIsing.SquareTopology{InteractiveIsing.Periodic, 2, Float64}}}, Array{Float32, 1}, 1, Base.UnitRange{Int64}}}}, InteractiveIsing.IsingGraph{Float32, InteractiveIsing.UndirectedAdjacency{Float32, Int32, SparseArrays.SparseMatrixCSC{Float32, Int32}, InteractiveIsing.OffsetArray{Int64, 1, Array{Int64, 1}}, Nothing}, Tuple{InteractiveIsing.IsingLayerData{InteractiveIsing.Discrete(), (-1f0, 1f0), 2, (100, 100), Base.UnitRange{Int64}(start=1, stop=10000), InteractiveIsing.SquareTopology{InteractiveIsing.Periodic, 2, Float64}}}, Array{Float32, 1}, 1, Base.UnitRange{Int64}}}, Random.MersenneTwister, InteractiveIsing.FlipProposal{Float32}, Float32, Float32}}, Tuple{}, Tuple{}}, NamedTuple{(:lifetime, :algo), Tuple{InteractiveIsing.Processes.Repeat, InteractiveIsing.Processes.CompositeAlgorithm{Tuple{InteractiveIsing.Processes.IdentifiableAlgo{InteractiveIsing.Metropolis, Base.UUID(value=UInt128(0x2e51cc537efa45449e8e2b1f0e2f4880)), InteractiveIsing.Processes.VarAliases{NamedTuple(), NamedTuple()}(), nothing, :var""}}, InteractiveIsing.Processes.RepeatOne{1}, Tuple{}, Tuple{}, nothing}}}}}, InteractiveIsing.Processes.NameSpaceRegistry{Tuple{InteractiveIsing.Processes.RegistryTypeEntry{InteractiveIsing.Metropolis, Tuple{InteractiveIsing.Processes.IdentifiableAlgo{InteractiveIsing.Metropolis, Base.UUID(value=UInt128(0x2e51cc537efa45449e8e2b1f0e2f4880)), InteractiveIsing.Processes.VarAliases{NamedTuple(), NamedTuple()}(), nothing, :Metropolis_1}}}}}}, InteractiveIsing.Processes.Repeat, :sync}, InteractiveIsing.Processes.CompositeAlgorithm{Tuple{InteractiveIsing.Processes.IdentifiableAlgo{InteractiveIsing.Metropolis, Base.UUID(value=UInt128(0x2e51cc537efa45449e8e2b1f0e2f4880)), InteractiveIsing.Processes.VarAliases{NamedTuple(), NamedTuple()}(), nothing, :Metropolis_1}}, InteractiveIsing.Processes.RepeatOne{1}, Tuple{}, Tuple{}, nothing}, InteractiveIsing.Processes.ProcessContext{NamedTuple{(:Metropolis_1, :globals), Tuple{InteractiveIsing.Processes.SubContext{:Metropolis_1, NamedTuple{(:state, :hamiltonian, :proposer, :rng, :proposal, :ΔE, :T), Tuple{InteractiveIsing.IsingGraph{Float32, InteractiveIsing.UndirectedAdjacency{Float32, Int32, SparseArrays.SparseMatrixCSC{Float32, Int32}, InteractiveIsing.OffsetArray{Int64, 1, Array{Int64, 1}}, Nothing}, Tuple{InteractiveIsing.IsingLayerData{InteractiveIsing.Discrete(), (-1f0, 1f0), 2, (100, 100), Base.UnitRange{Int64}(start=1, stop=10000), InteractiveIsing.SquareTopology{InteractiveIsing.Periodic, 2, Float64}}}, Array{Float32, 1}, 1, Base.UnitRange{Int64}}, InteractiveIsing.HamiltonianTerms{Tuple{InteractiveIsing.PolynomialHamiltonian{2, InteractiveIsing.UniformArray{Float32, 0}, InteractiveIsing.OffsetArray{Int64, 1, Array{Int64, 1}}}, InteractiveIsing.Bilinear{InteractiveIsing.UndirectedAdjacency{Float32, Int32, SparseArrays.SparseMatrixCSC{Float32, Int32}, InteractiveIsing.OffsetArray{Int64, 1, Array{Int64, 1}}, Nothing}}, InteractiveIsing.MagField{InteractiveIsing.ConstFill{0f0, Float32, 2}}}}, InteractiveIsing.IsingGraphProposer{Base.UnitRange{Int64}, Tuple{InteractiveIsing.IsingLayer{InteractiveIsing.Discrete(), 2, InteractiveIsing.IsingLayerData{InteractiveIsing.Discrete(), (-1f0, 1f0), 2, (100, 100), Base.UnitRange{Int64}(start=1, stop=10000), InteractiveIsing.SquareTopology{InteractiveIsing.Periodic, 2, Float64}}, InteractiveIsing.IsingGraph{Float32, InteractiveIsing.UndirectedAdjacency{Float32, Int32, SparseArrays.SparseMatrixCSC{Float32, Int32}, InteractiveIsing.OffsetArray{Int64, 1, Array{Int64, 1}}, Nothing}, Tuple{InteractiveIsing.IsingLayerData{InteractiveIsing.Discrete(), (-1f0, 1f0), 2, (100, 100), Base.UnitRange{Int64}(start=1, stop=10000), InteractiveIsing.SquareTopology{InteractiveIsing.Periodic, 2, Float64}}}, Array{Float32, 1}, 1, Base.UnitRange{Int64}}}}, InteractiveIsing.IsingGraph{Float32, InteractiveIsing.UndirectedAdjacency{Float32, Int32, SparseArrays.SparseMatrixCSC{Float32, Int32}, InteractiveIsing.OffsetArray{Int64, 1, Array{Int64, 1}}, Nothing}, Tuple{InteractiveIsing.IsingLayerData{InteractiveIsing.Discrete(), (-1f0, 1f0), 2, (100, 100), Base.UnitRange{Int64}(start=1, stop=10000), InteractiveIsing.SquareTopology{InteractiveIsing.Periodic, 2, Float64}}}, Array{Float32, 1}, 1, Base.UnitRange{Int64}}}, Random.MersenneTwister, InteractiveIsing.FlipProposal{Float32}, Float32, Float32}}, Tuple{}, Tuple{}}, NamedTuple{(:lifetime, :algo, :process), Tuple{InteractiveIsing.Processes.Repeat, InteractiveIsing.Processes.CompositeAlgorithm{Tuple{InteractiveIsing.Processes.IdentifiableAlgo{InteractiveIsing.Metropolis, Base.UUID(value=UInt128(0x2e51cc537efa45449e8e2b1f0e2f4880)), InteractiveIsing.Processes.VarAliases{NamedTuple(), NamedTuple()}(), nothing, :var""}}, InteractiveIsing.Processes.RepeatOne{1}, Tuple{}, Tuple{}, nothing}, InteractiveIsing.Processes.InlineProcess{InteractiveIsing.Processes.TaskData{InteractiveIsing.Processes.CompositeAlgorithm{Tuple{InteractiveIsing.Processes.IdentifiableAlgo{InteractiveIsing.Metropolis, Base.UUID(value=UInt128(0x2e51cc537efa45449e8e2b1f0e2f4880)), InteractiveIsing.Processes.VarAliases{NamedTuple(), NamedTuple()}(), nothing, :Metropolis_1}}, InteractiveIsing.Processes.RepeatOne{1}, Tuple{}, Tuple{}, nothing}, Tuple{InteractiveIsing.Processes.NamedInput{:Metropolis_1, NamedTuple{(:state,), Tuple{InteractiveIsing.IsingGraph{Float32, InteractiveIsing.UndirectedAdjacency{Float32, Int32, SparseArrays.SparseMatrixCSC{Float32, Int32}, InteractiveIsing.OffsetArray{Int64, 1, Array{Int64, 1}}, Nothing}, Tuple{InteractiveIsing.IsingLayerData{InteractiveIsing.Discrete(), (-1f0, 1f0), 2, (100, 100), Base.UnitRange{Int64}(start=1, stop=10000), InteractiveIsing.SquareTopology{InteractiveIsing.Periodic, 2, Float64}}}, Array{Float32, 1}, 1, Base.UnitRange{Int64}}}}}}, Tuple{}, InteractiveIsing.Processes.ProcessContext{NamedTuple{(:Metropolis_1, :globals), Tuple{InteractiveIsing.Processes.SubContext{:Metropolis_1, NamedTuple{(), Tuple{}}, Tuple{}, Tuple{}}, NamedTuple{(:lifetime, :algo), Tuple{InteractiveIsing.Processes.Repeat, InteractiveIsing.Processes.CompositeAlgorithm{Tuple{InteractiveIsing.Processes.IdentifiableAlgo{InteractiveIsing.Metropolis, Base.UUID(value=UInt128(0x2e51cc537efa45449e8e2b1f0e2f4880)), InteractiveIsing.Processes.VarAliases{NamedTuple(), NamedTuple()}(), nothing, :var""}}, InteractiveIsing.Processes.RepeatOne{1}, Tuple{}, Tuple{}, nothing}}}}}, InteractiveIsing.Processes.NameSpaceRegistry{Tuple{InteractiveIsing.Processes.RegistryTypeEntry{InteractiveIsing.Metropolis, Tuple{InteractiveIsing.Processes.IdentifiableAlgo{InteractiveIsing.Metropolis, Base.UUID(value=UInt128(0x2e51cc537efa45449e8e2b1f0e2f4880)), InteractiveIsing.Processes.VarAliases{NamedTuple(), NamedTuple()}(), nothing, :Metropolis_1}}}}}}, InteractiveIsing.Processes.Repeat}, InteractiveIsing.Processes.ProcessContext{NamedTuple{(:Metropolis_1, :globals), Tuple{InteractiveIsing.Processes.SubContext{:Metropolis_1, NamedTuple{(:state, :hamiltonian, :proposer, :rng, :proposal, :ΔE, :T), Tuple{InteractiveIsing.IsingGraph{Float32, InteractiveIsing.UndirectedAdjacency{Float32, Int32, SparseArrays.SparseMatrixCSC{Float32, Int32}, InteractiveIsing.OffsetArray{Int64, 1, Array{Int64, 1}}, Nothing}, Tuple{InteractiveIsing.IsingLayerData{InteractiveIsing.Discrete(), (-1f0, 1f0), 2, (100, 100), Base.UnitRange{Int64}(start=1, stop=10000), InteractiveIsing.SquareTopology{InteractiveIsing.Periodic, 2, Float64}}}, Array{Float32, 1}, 1, Base.UnitRange{Int64}}, InteractiveIsing.HamiltonianTerms{Tuple{InteractiveIsing.PolynomialHamiltonian{2, InteractiveIsing.UniformArray{Float32, 0}, InteractiveIsing.OffsetArray{Int64, 1, Array{Int64, 1}}}, InteractiveIsing.Bilinear{InteractiveIsing.UndirectedAdjacency{Float32, Int32, SparseArrays.SparseMatrixCSC{Float32, Int32}, InteractiveIsing.OffsetArray{Int64, 1, Array{Int64, 1}}, Nothing}}, InteractiveIsing.MagField{InteractiveIsing.ConstFill{0f0, Float32, 2}}}}, InteractiveIsing.IsingGraphProposer{Base.UnitRange{Int64}, Tuple{InteractiveIsing.IsingLayer{InteractiveIsing.Discrete(), 2, InteractiveIsing.IsingLayerData{InteractiveIsing.Discrete(), (-1f0, 1f0), 2, (100, 100), Base.UnitRange{Int64}(start=1, stop=10000), InteractiveIsing.SquareTopology{InteractiveIsing.Periodic, 2, Float64}}, InteractiveIsing.IsingGraph{Float32, InteractiveIsing.UndirectedAdjacency{Float32, Int32, SparseArrays.SparseMatrixCSC{Float32, Int32}, InteractiveIsing.OffsetArray{Int64, 1, Array{Int64, 1}}, Nothing}, Tuple{InteractiveIsing.IsingLayerData{InteractiveIsing.Discrete(), (-1f0, 1f0), 2, (100, 100), Base.UnitRange{Int64}(start=1, stop=10000), InteractiveIsing.SquareTopology{InteractiveIsing.Periodic, 2, Float64}}}, Array{Float32, 1}, 1, Base.UnitRange{Int64}}}}, InteractiveIsing.IsingGraph{Float32, InteractiveIsing.UndirectedAdjacency{Float32, Int32, SparseArrays.SparseMatrixCSC{Float32, Int32}, InteractiveIsing.OffsetArray{Int64, 1, Array{Int64, 1}}, Nothing}, Tuple{InteractiveIsing.IsingLayerData{InteractiveIsing.Discrete(), (-1f0, 1f0), 2, (100, 100), Base.UnitRange{Int64}(start=1, stop=10000), InteractiveIsing.SquareTopology{InteractiveIsing.Periodic, 2, Float64}}}, Array{Float32, 1}, 1, Base.UnitRange{Int64}}}, Random.MersenneTwister, InteractiveIsing.FlipProposal{Float32}, Float32, Float32}}, Tuple{}, Tuple{}}, NamedTuple{(:lifetime, :algo), Tuple{InteractiveIsing.Processes.Repeat, InteractiveIsing.Processes.CompositeAlgorithm{Tuple{InteractiveIsing.Processes.IdentifiableAlgo{InteractiveIsing.Metropolis, Base.UUID(value=UInt128(0x2e51cc537efa45449e8e2b1f0e2f4880)), InteractiveIsing.Processes.VarAliases{NamedTuple(), NamedTuple()}(), nothing, :var""}}, InteractiveIsing.Processes.RepeatOne{1}, Tuple{}, Tuple{}, nothing}}}}}, InteractiveIsing.Processes.NameSpaceRegistry{Tuple{InteractiveIsing.Processes.RegistryTypeEntry{InteractiveIsing.Metropolis, Tuple{InteractiveIsing.Processes.IdentifiableAlgo{InteractiveIsing.Metropolis, Base.UUID(value=UInt128(0x2e51cc537efa45449e8e2b1f0e2f4880)), InteractiveIsing.Processes.VarAliases{NamedTuple(), NamedTuple()}(), nothing, :Metropolis_1}}}}}}, InteractiveIsing.Processes.Repeat, :sync}}}}}, InteractiveIsing.Processes.NameSpaceRegistry{Tuple{InteractiveIsing.Processes.RegistryTypeEntry{InteractiveIsing.Metropolis, Tuple{InteractiveIsing.Processes.IdentifiableAlgo{InteractiveIsing.Metropolis, Base.UUID(value=UInt128(0x2e51cc537efa45449e8e2b1f0e2f4880)), InteractiveIsing.Processes.VarAliases{NamedTuple(), NamedTuple()}(), nothing, :Metropolis_1}}}}}}, InteractiveIsing.Processes.Repeat)
define void @julia_generated_processloop_9755(ptr noalias nocapture noundef nonnull sret({ { { { ptr, [1 x { { { ptr }, ptr }, [1 x { { i64, i64, ptr, ptr, ptr }, ptr }], [1 x [1 x [2 x i64]]] }], { [2 x i64], [1 x { { ptr, ptr, ptr, ptr, ptr, { [2 x i64], ptr, [1 x [2 x double]] } }, ptr, i64 }], ptr }, ptr, { i64, float, float, i64, i8 }, float, float } }, { [1 x i64], { ptr }, ptr } }, [1 x [1 x { ptr, ptr }]] }) align 8 dereferenceable(296) %sret_return, ptr noalias nocapture noundef nonnull align 8 dereferenceable(160) %return_roots, ptr noundef nonnull align 16 dereferenceable(416) %"process::InlineProcess", ptr nocapture noundef nonnull readonly align 8 dereferenceable(8) %"algo::CompositeAlgorithm", ptr nocapture noundef nonnull readonly align 8 dereferenceable(296) %"context::ProcessContext", ptr nocapture readonly %.roots.context, ptr nocapture noundef nonnull readonly align 8 dereferenceable(8) %"r::Repeat") #0 {
top:
  %jlcallframe1 = alloca [4 x ptr], align 8
  %gcframe2 = alloca [22 x ptr], align 16
  call void @llvm.memset.p0.i64(ptr align 16 %gcframe2, i8 0, i64 176, i1 true)
  %0 = getelementptr inbounds ptr, ptr %gcframe2, i64 6
  %1 = getelementptr inbounds ptr, ptr %gcframe2, i64 2
  %2 = alloca [37 x i64], align 8
  %.sroa.0201.sroa.0 = alloca [27 x i64], align 8
  %.sroa.0201.sroa.11 = alloca [7 x i8], align 1
  %.sroa.8206 = alloca [3 x i64], align 8
  %.sroa.9207 = alloca [2 x i64], align 8
  %"new::SamplerRangeNDL" = alloca [2 x i64], align 8
  %"new::FlipProposal.sroa.11" = alloca [7 x i8], align 1
  %"new::#calculate##0#calculate##1" = alloca [5 x i64], align 8
  %"new::Tuple45" = alloca [1 x i64], align 8
  %"new::Tuple76" = alloca [1 x i64], align 8
  %.sroa.6344 = alloca [7 x i8], align 1
  %.sroa.10 = alloca [7 x i8], align 1
  %"new::NamedTuple.sroa.0.sroa.0" = alloca [27 x i64], align 8
  %.sroa.0.sroa.0 = alloca [27 x i64], align 8
  %.sroa.0.sroa.11 = alloca [7 x i8], align 1
  %.sroa.8 = alloca [3 x i64], align 8
  %"new::Tuple170" = alloca [1 x i64], align 8
  %pgcstack = call ptr inttoptr (i64 4329881372 to ptr)(i64 4329881408) #26
  store i64 80, ptr %gcframe2, align 8
  %task.gcstack = load ptr, ptr %pgcstack, align 8
  %frame.prev = getelementptr inbounds ptr, ptr %gcframe2, i64 1
  store ptr %task.gcstack, ptr %frame.prev, align 8
  store ptr %gcframe2, ptr %pgcstack, align 8
  %3 = load ptr, ptr %.roots.context, align 8
  %4 = getelementptr inbounds i8, ptr %.roots.context, i64 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr inbounds i8, ptr %.roots.context, i64 16
  %7 = load ptr, ptr %6, align 8
  %8 = getelementptr inbounds i8, ptr %.roots.context, i64 24
  %9 = load ptr, ptr %8, align 8
  %10 = getelementptr inbounds i8, ptr %.roots.context, i64 32
  %11 = load ptr, ptr %10, align 8
  %12 = getelementptr inbounds i8, ptr %.roots.context, i64 40
  %13 = load ptr, ptr %12, align 8
  %14 = getelementptr inbounds i8, ptr %.roots.context, i64 48
  %15 = load <2 x ptr>, ptr %14, align 8
  %16 = getelementptr inbounds i8, ptr %.roots.context, i64 64
  %17 = load <2 x ptr>, ptr %16, align 8
  %18 = getelementptr inbounds i8, ptr %.roots.context, i64 80
  %19 = load <2 x ptr>, ptr %18, align 8
  %20 = getelementptr inbounds i8, ptr %.roots.context, i64 96
  %21 = load <2 x ptr>, ptr %20, align 8
  %22 = getelementptr inbounds i8, ptr %.roots.context, i64 112
  %23 = load ptr, ptr %22, align 8
  %24 = getelementptr inbounds i8, ptr %.roots.context, i64 120
  %25 = load <2 x ptr>, ptr %24, align 8
  %26 = getelementptr inbounds i8, ptr %.roots.context, i64 136
  %27 = load <2 x ptr>, ptr %26, align 8
  %28 = getelementptr inbounds i8, ptr %.roots.context, i64 152
  %29 = load ptr, ptr %28, align 8
  %gc_slot_addr_19 = getelementptr inbounds ptr, ptr %gcframe2, i64 21
  %30 = extractelement <2 x ptr> %27, i64 0
  store ptr %30, ptr %gc_slot_addr_19, align 8
  %gc_slot_addr_18 = getelementptr inbounds ptr, ptr %gcframe2, i64 20
  %31 = extractelement <2 x ptr> %27, i64 1
  store ptr %31, ptr %gc_slot_addr_18, align 8
  %gc_slot_addr_17 = getelementptr inbounds ptr, ptr %gcframe2, i64 19
  %32 = extractelement <2 x ptr> %25, i64 0
  store ptr %32, ptr %gc_slot_addr_17, align 8
  %gc_slot_addr_16 = getelementptr inbounds ptr, ptr %gcframe2, i64 18
  %33 = extractelement <2 x ptr> %25, i64 1
  store ptr %33, ptr %gc_slot_addr_16, align 8
  %gc_slot_addr_15 = getelementptr inbounds ptr, ptr %gcframe2, i64 17
  %34 = extractelement <2 x ptr> %21, i64 0
  store ptr %34, ptr %gc_slot_addr_15, align 8
  %gc_slot_addr_14 = getelementptr inbounds ptr, ptr %gcframe2, i64 16
  %35 = extractelement <2 x ptr> %21, i64 1
  store ptr %35, ptr %gc_slot_addr_14, align 8
  %gc_slot_addr_13 = getelementptr inbounds ptr, ptr %gcframe2, i64 15
  %36 = extractelement <2 x ptr> %19, i64 0
  store ptr %36, ptr %gc_slot_addr_13, align 8
  %gc_slot_addr_12 = getelementptr inbounds ptr, ptr %gcframe2, i64 14
  %37 = extractelement <2 x ptr> %19, i64 1
  store ptr %37, ptr %gc_slot_addr_12, align 8
  %gc_slot_addr_11 = getelementptr inbounds ptr, ptr %gcframe2, i64 13
  %38 = extractelement <2 x ptr> %17, i64 0
  store ptr %38, ptr %gc_slot_addr_11, align 8
  %gc_slot_addr_10 = getelementptr inbounds ptr, ptr %gcframe2, i64 12
  %39 = extractelement <2 x ptr> %17, i64 1
  store ptr %39, ptr %gc_slot_addr_10, align 8
  %gc_slot_addr_9 = getelementptr inbounds ptr, ptr %gcframe2, i64 11
  %40 = extractelement <2 x ptr> %15, i64 0
  store ptr %40, ptr %gc_slot_addr_9, align 8
  %gc_slot_addr_8 = getelementptr inbounds ptr, ptr %gcframe2, i64 10
  %41 = extractelement <2 x ptr> %15, i64 1
  store ptr %41, ptr %gc_slot_addr_8, align 8
  %42 = call i64 @jlplt_ijl_hrtime_9759_got.jit()
  %"process::InlineProcess.starttime_ptr" = getelementptr inbounds i8, ptr %"process::InlineProcess", i64 376
  %"process::InlineProcess.starttime.tindex_ptr" = getelementptr inbounds i8, ptr %"process::InlineProcess", i64 384
  store i8 2, ptr %"process::InlineProcess.starttime.tindex_ptr", align 1
  store i64 %42, ptr %"process::InlineProcess.starttime_ptr", align 8
  %"process::InlineProcess.loopidx_ptr" = getelementptr inbounds i8, ptr %"process::InlineProcess", i64 360
  %"process::InlineProcess.loopidx" = load i64, ptr %"process::InlineProcess.loopidx_ptr", align 8
  %"r::Repeat.unbox" = load i64, ptr %"r::Repeat", align 8
  %.not = icmp sgt i64 %"r::Repeat.unbox", -1
  br i1 %.not, label %L12, label %L9

L9:                                               ; preds = %top
  %43 = load ptr, ptr getelementptr inbounds (i8, ptr @jl_small_typeof, i64 320), align 8
  store ptr null, ptr %gc_slot_addr_19, align 8
  store ptr null, ptr %gc_slot_addr_18, align 8
  store ptr null, ptr %gc_slot_addr_17, align 8
  store ptr null, ptr %gc_slot_addr_16, align 8
  store ptr null, ptr %gc_slot_addr_15, align 8
  store ptr null, ptr %gc_slot_addr_14, align 8
  store ptr null, ptr %gc_slot_addr_13, align 8
  store ptr null, ptr %gc_slot_addr_12, align 8
  store ptr null, ptr %gc_slot_addr_11, align 8
  store ptr null, ptr %gc_slot_addr_10, align 8
  store ptr null, ptr %gc_slot_addr_9, align 8
  store ptr null, ptr %gc_slot_addr_8, align 8
  call void @j_throw_inexacterror_9760(ptr nonnull @"jl_sym#convert#9761.jit", ptr nonnull %43, i64 signext %"r::Repeat.unbox") #10
  unreachable

L12:                                              ; preds = %top
  %.not208 = icmp ugt i64 %"process::InlineProcess.loopidx", %"r::Repeat.unbox"
  %44 = add i64 %"process::InlineProcess.loopidx", -1
  %value_phi = select i1 %.not208, i64 %44, i64 %"r::Repeat.unbox"
  %.not209.not = icmp ult i64 %value_phi, %"process::InlineProcess.loopidx"
  br i1 %.not209.not, label %L31.L559_crit_edge, label %L31.L35_crit_edge

L31.L559_crit_edge:                               ; preds = %L12
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(216) %.sroa.0.sroa.0, ptr noundef nonnull align 8 dereferenceable(216) %"context::ProcessContext", i64 216, i1 false)
  %".sroa.0.sroa.6.0.context::ProcessContext.sroa_idx" = getelementptr inbounds i8, ptr %"context::ProcessContext", i64 216
  %.sroa.0.sroa.6.0.copyload = load i64, ptr %".sroa.0.sroa.6.0.context::ProcessContext.sroa_idx", align 8
  %".sroa.0.sroa.7.0.context::ProcessContext.sroa_idx" = getelementptr inbounds i8, ptr %"context::ProcessContext", i64 224
  %.sroa.0.sroa.7.0.copyload = load float, ptr %".sroa.0.sroa.7.0.context::ProcessContext.sroa_idx", align 8
  %".sroa.0.sroa.8.0.context::ProcessContext.sroa_idx" = getelementptr inbounds i8, ptr %"context::ProcessContext", i64 228
  %.sroa.0.sroa.8.0.copyload = load float, ptr %".sroa.0.sroa.8.0.context::ProcessContext.sroa_idx", align 4
  %".sroa.0.sroa.9.0.context::ProcessContext.sroa_idx" = getelementptr inbounds i8, ptr %"context::ProcessContext", i64 232
  %.sroa.0.sroa.9.0.copyload = load i64, ptr %".sroa.0.sroa.9.0.context::ProcessContext.sroa_idx", align 8
  %".sroa.0.sroa.10.0.context::ProcessContext.sroa_idx" = getelementptr inbounds i8, ptr %"context::ProcessContext", i64 240
  %.sroa.0.sroa.10.0.copyload = load i8, ptr %".sroa.0.sroa.10.0.context::ProcessContext.sroa_idx", align 8
  %".sroa.0.sroa.11.0.context::ProcessContext.sroa_idx" = getelementptr inbounds i8, ptr %"context::ProcessContext", i64 241
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %.sroa.0.sroa.11, ptr noundef nonnull align 1 dereferenceable(7) %".sroa.0.sroa.11.0.context::ProcessContext.sroa_idx", i64 7, i1 false)
  %".sroa.6.0.context::ProcessContext.sroa_idx" = getelementptr inbounds i8, ptr %"context::ProcessContext", i64 248
  %.sroa.6.0.copyload197 = load float, ptr %".sroa.6.0.context::ProcessContext.sroa_idx", align 8
  %".sroa.7.0.context::ProcessContext.sroa_idx" = getelementptr inbounds i8, ptr %"context::ProcessContext", i64 252
  %.sroa.7.0.copyload198 = load float, ptr %".sroa.7.0.context::ProcessContext.sroa_idx", align 4
  %".sroa.8.0.context::ProcessContext.sroa_idx" = getelementptr inbounds i8, ptr %"context::ProcessContext", i64 256
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %.sroa.8, ptr noundef nonnull align 8 dereferenceable(24) %".sroa.8.0.context::ProcessContext.sroa_idx", i64 24, i1 false)
  br label %L559

L31.L35_crit_edge:                                ; preds = %L12
  call void @llvm.lifetime.start.p0(i64 216, ptr nonnull %.sroa.0201.sroa.0)
  call void @llvm.lifetime.start.p0(i64 7, ptr nonnull %.sroa.0201.sroa.11)
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %.sroa.8206)
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %.sroa.9207)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(216) %.sroa.0201.sroa.0, ptr noundef nonnull align 8 dereferenceable(216) %"context::ProcessContext", i64 216, i1 false)
  %".sroa.0201.sroa.6.0.context::ProcessContext.sroa_idx" = getelementptr inbounds i8, ptr %"context::ProcessContext", i64 216
  %.sroa.0201.sroa.6.0.copyload355 = load i64, ptr %".sroa.0201.sroa.6.0.context::ProcessContext.sroa_idx", align 8
  %".sroa.0201.sroa.7.0.context::ProcessContext.sroa_idx" = getelementptr inbounds i8, ptr %"context::ProcessContext", i64 224
  %.sroa.0201.sroa.7.0.copyload356 = load float, ptr %".sroa.0201.sroa.7.0.context::ProcessContext.sroa_idx", align 8
  %".sroa.0201.sroa.8.0.context::ProcessContext.sroa_idx" = getelementptr inbounds i8, ptr %"context::ProcessContext", i64 228
  %.sroa.0201.sroa.8.0.copyload357 = load float, ptr %".sroa.0201.sroa.8.0.context::ProcessContext.sroa_idx", align 4
  %".sroa.0201.sroa.9.0.context::ProcessContext.sroa_idx" = getelementptr inbounds i8, ptr %"context::ProcessContext", i64 232
  %.sroa.0201.sroa.9.0.copyload358 = load i64, ptr %".sroa.0201.sroa.9.0.context::ProcessContext.sroa_idx", align 8
  %".sroa.0201.sroa.10.0.context::ProcessContext.sroa_idx" = getelementptr inbounds i8, ptr %"context::ProcessContext", i64 240
  %.sroa.0201.sroa.10.0.copyload359 = load i8, ptr %".sroa.0201.sroa.10.0.context::ProcessContext.sroa_idx", align 8
  %".sroa.0201.sroa.11.0.context::ProcessContext.sroa_idx" = getelementptr inbounds i8, ptr %"context::ProcessContext", i64 241
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %.sroa.0201.sroa.11, ptr noundef nonnull align 1 dereferenceable(7) %".sroa.0201.sroa.11.0.context::ProcessContext.sroa_idx", i64 7, i1 false)
  %".sroa.6202.0.context::ProcessContext.sroa_idx" = getelementptr inbounds i8, ptr %"context::ProcessContext", i64 248
  %.sroa.6202.0.copyload203 = load float, ptr %".sroa.6202.0.context::ProcessContext.sroa_idx", align 8
  %".sroa.7204.0.context::ProcessContext.sroa_idx" = getelementptr inbounds i8, ptr %"context::ProcessContext", i64 252
  %.sroa.7204.0.copyload205 = load float, ptr %".sroa.7204.0.context::ProcessContext.sroa_idx", align 4
  %".sroa.8206.0.context::ProcessContext.sroa_idx" = getelementptr inbounds i8, ptr %"context::ProcessContext", i64 256
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %.sroa.8206, ptr noundef nonnull align 8 dereferenceable(24) %".sroa.8206.0.context::ProcessContext.sroa_idx", i64 24, i1 false)
  %".sroa.9207.0.context::ProcessContext.sroa_idx" = getelementptr inbounds i8, ptr %"context::ProcessContext", i64 280
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %.sroa.9207, ptr noundef nonnull align 8 dereferenceable(16) %".sroa.9207.0.context::ProcessContext.sroa_idx", i64 16, i1 false)
  %.sroa.6202.0..sroa_idx = getelementptr inbounds i8, ptr %2, i64 248
  %.sroa.7204.0..sroa_idx = getelementptr inbounds i8, ptr %2, i64 252
  %.sroa.8206.0..sroa_idx = getelementptr inbounds i8, ptr %2, i64 256
  %.sroa.9207.0..sroa_idx = getelementptr inbounds i8, ptr %2, i64 280
  %45 = getelementptr inbounds i8, ptr %2, i64 88
  %.stop_ptr = getelementptr inbounds i8, ptr %2, i64 96
  %46 = getelementptr inbounds i8, ptr %"new::SamplerRangeNDL", i64 8
  %47 = getelementptr inbounds i8, ptr %"new::#calculate##0#calculate##1", i64 8
  %48 = getelementptr inbounds i8, ptr %2, i64 24
  %49 = getelementptr inbounds i8, ptr %2, i64 72
  %root_phi6.offset_ptr = getelementptr inbounds i8, ptr %7, i64 8
  %50 = getelementptr inbounds ptr, ptr %gcframe2, i64 3
  %51 = getelementptr inbounds ptr, ptr %gcframe2, i64 4
  %52 = getelementptr inbounds ptr, ptr %gcframe2, i64 5
  %"[2]_ptr" = getelementptr inbounds i8, ptr %2, i64 80
  %root_phi.temp_ptr = getelementptr inbounds i8, ptr %3, i64 56
  %root_phi19.idxF_ptr = getelementptr inbounds i8, ptr %32, i64 32
  %root_phi19.vals_ptr = getelementptr inbounds i8, ptr %32, i64 16
  %53 = getelementptr inbounds i8, ptr %2, i64 8
  %"algo::CompositeAlgorithm.inc112" = load atomic ptr, ptr %"algo::CompositeAlgorithm" unordered, align 8
  br label %L35

L35:                                              ; preds = %L558, %L31.L35_crit_edge
  %.sroa.0201.sroa.6.0 = phi i64 [ %.sroa.0201.sroa.6.0.copyload355, %L31.L35_crit_edge ], [ %.fr420, %L558 ]
  %.sroa.0201.sroa.7.0 = phi float [ %.sroa.0201.sroa.7.0.copyload356, %L31.L35_crit_edge ], [ %62, %L558 ]
  %.sroa.0201.sroa.8.0 = phi float [ %.sroa.0201.sroa.8.0.copyload357, %L31.L35_crit_edge ], [ %., %L558 ]
  %.sroa.0201.sroa.9.0 = phi i64 [ %.sroa.0201.sroa.9.0.copyload358, %L31.L35_crit_edge ], [ 1, %L558 ]
  %.sroa.0201.sroa.10.0 = phi i8 [ %.sroa.0201.sroa.10.0.copyload359, %L31.L35_crit_edge ], [ %.sroa.9.0, %L558 ]
  %54 = phi i64 [ undef, %L31.L35_crit_edge ], [ %.fr420, %L558 ]
  %.sroa.6202.0 = phi float [ %.sroa.6202.0.copyload203, %L31.L35_crit_edge ], [ %96, %L558 ]
  %.sroa.7204.0 = phi float [ %.sroa.7204.0.copyload205, %L31.L35_crit_edge ], [ %root_phi.temp, %L558 ]
  %value_phi4 = phi i64 [ %"process::InlineProcess.loopidx", %L31.L35_crit_edge ], [ %138, %L558 ]
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(216) %2, ptr noundef nonnull align 8 dereferenceable(216) %.sroa.0201.sroa.0, i64 216, i1 false)
  %.sroa.0201.sroa.6.0..sroa_idx = getelementptr inbounds i8, ptr %2, i64 216
  store i64 %.sroa.0201.sroa.6.0, ptr %.sroa.0201.sroa.6.0..sroa_idx, align 8
  %.sroa.0201.sroa.7.0..sroa_idx = getelementptr inbounds i8, ptr %2, i64 224
  store float %.sroa.0201.sroa.7.0, ptr %.sroa.0201.sroa.7.0..sroa_idx, align 8
  %.sroa.0201.sroa.8.0..sroa_idx = getelementptr inbounds i8, ptr %2, i64 228
  store float %.sroa.0201.sroa.8.0, ptr %.sroa.0201.sroa.8.0..sroa_idx, align 4
  %.sroa.0201.sroa.9.0..sroa_idx = getelementptr inbounds i8, ptr %2, i64 232
  store i64 %.sroa.0201.sroa.9.0, ptr %.sroa.0201.sroa.9.0..sroa_idx, align 8
  %.sroa.0201.sroa.10.0..sroa_idx = getelementptr inbounds i8, ptr %2, i64 240
  store i8 %.sroa.0201.sroa.10.0, ptr %.sroa.0201.sroa.10.0..sroa_idx, align 8
  %.sroa.0201.sroa.11.0..sroa_idx = getelementptr inbounds i8, ptr %2, i64 241
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %.sroa.0201.sroa.11.0..sroa_idx, ptr noundef nonnull align 1 dereferenceable(7) %.sroa.0201.sroa.11, i64 7, i1 false)
  store float %.sroa.6202.0, ptr %.sroa.6202.0..sroa_idx, align 8
  store float %.sroa.7204.0, ptr %.sroa.7204.0..sroa_idx, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %.sroa.8206.0..sroa_idx, ptr noundef nonnull align 8 dereferenceable(24) %.sroa.8206, i64 24, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %.sroa.9207.0..sroa_idx, ptr noundef nonnull align 8 dereferenceable(16) %.sroa.9207, i64 16, i1 false)
  call void @llvm.lifetime.end.p0(i64 216, ptr nonnull %.sroa.0201.sroa.0)
  call void @llvm.lifetime.end.p0(i64 7, ptr nonnull %.sroa.0201.sroa.11)
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %.sroa.8206)
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %.sroa.9207)
  %.stop_ptr.unbox = load i64, ptr %.stop_ptr, align 8
  %.unbox = load i64, ptr %45, align 8
  %.not210.not = icmp slt i64 %.stop_ptr.unbox, %.unbox
  br i1 %.not210.not, label %L58, label %L61

L58:                                              ; preds = %L35
  store ptr null, ptr %gc_slot_addr_19, align 8
  store ptr null, ptr %gc_slot_addr_18, align 8
  store ptr null, ptr %gc_slot_addr_17, align 8
  store ptr null, ptr %gc_slot_addr_16, align 8
  store ptr null, ptr %gc_slot_addr_15, align 8
  store ptr null, ptr %gc_slot_addr_14, align 8
  store ptr null, ptr %gc_slot_addr_13, align 8
  store ptr null, ptr %gc_slot_addr_12, align 8
  store ptr null, ptr %gc_slot_addr_11, align 8
  store ptr null, ptr %gc_slot_addr_10, align 8
  store ptr null, ptr %gc_slot_addr_9, align 8
  store ptr null, ptr %gc_slot_addr_8, align 8
  %55 = call [1 x ptr] @j_ArgumentError_9762(ptr nonnull @"jl_global#9763.jit")
  %gc_slot_addr_5 = getelementptr inbounds ptr, ptr %gcframe2, i64 7
  %56 = extractvalue [1 x ptr] %55, 0
  store ptr %56, ptr %gc_slot_addr_5, align 8
  %ptls_field = getelementptr inbounds i8, ptr %pgcstack, i64 16
  %ptls_load = load ptr, ptr %ptls_field, align 8
  %"box::ArgumentError" = call noalias nonnull align 8 dereferenceable(16) ptr @ijl_gc_small_alloc(ptr %ptls_load, i32 424, i32 16, i64 4996298848) #22
  %"box::ArgumentError.tag_addr" = getelementptr inbounds i64, ptr %"box::ArgumentError", i64 -1
  store atomic i64 4996298848, ptr %"box::ArgumentError.tag_addr" unordered, align 8
  store ptr %56, ptr %"box::ArgumentError", align 8
  store ptr null, ptr %gc_slot_addr_5, align 8
  call void @ijl_throw(ptr nonnull %"box::ArgumentError")
  unreachable

L61:                                              ; preds = %L35
  %57 = add i64 %.stop_ptr.unbox, 1
  %58 = sub i64 %57, %.unbox
  store i64 %.unbox, ptr %"new::SamplerRangeNDL", align 8
  store i64 %58, ptr %46, align 8
  %gc_slot_addr_7 = getelementptr inbounds ptr, ptr %gcframe2, i64 9
  store ptr %"algo::CompositeAlgorithm.inc112", ptr %gc_slot_addr_7, align 8
  %59 = call i64 @j_rand_9765(ptr %32, ptr nocapture nonnull readonly %"new::SamplerRangeNDL")
  %.fr420 = freeze i64 %59
  %root_phi18.state = load atomic ptr, ptr %23 unordered, align 8
  %root_phi18.state.size_ptr = getelementptr inbounds i8, ptr %root_phi18.state, i64 16
  %root_phi18.state.size.0.copyload = load i64, ptr %root_phi18.state.size_ptr, align 8
  %.not211 = icmp eq i64 %root_phi18.state.size.0.copyload, 10000
  br i1 %.not211, label %L87, label %L82

L82:                                              ; preds = %L61
  store ptr null, ptr %gc_slot_addr_19, align 8
  store ptr null, ptr %gc_slot_addr_18, align 8
  store ptr null, ptr %gc_slot_addr_17, align 8
  store ptr null, ptr %gc_slot_addr_16, align 8
  store ptr null, ptr %gc_slot_addr_15, align 8
  store ptr null, ptr %gc_slot_addr_14, align 8
  store ptr null, ptr %gc_slot_addr_13, align 8
  store ptr null, ptr %gc_slot_addr_12, align 8
  store ptr null, ptr %gc_slot_addr_11, align 8
  store ptr null, ptr %gc_slot_addr_10, align 8
  store ptr null, ptr %gc_slot_addr_9, align 8
  store ptr null, ptr %gc_slot_addr_8, align 8
  store ptr null, ptr %gc_slot_addr_7, align 8
  call void @j_throw_dmrsa_9766(ptr nocapture nonnull readonly @"_j_const#2", i64 signext %root_phi18.state.size.0.copyload) #10
  unreachable

L87:                                              ; preds = %L61
  %60 = load ptr, ptr %root_phi18.state, align 8
  %memoryref_offset = shl i64 %.fr420, 2
  %61 = getelementptr i8, ptr %60, i64 %memoryref_offset
  %memoryref_data32 = getelementptr i8, ptr %61, i64 -4
  %62 = load float, ptr %memoryref_data32, align 4
  %63 = icmp slt i64 %.fr420, 10001
  br i1 %63, label %L133, label %L143

L133:                                             ; preds = %L87
  %64 = fcmp une float %62, -1.000000e+00
  %. = select i1 %64, float -1.000000e+00, float 1.000000e+00
  %"new::Tuple.sroa.0.sroa.6.0..sroa_idx" = getelementptr inbounds i8, ptr %"new::#calculate##0#calculate##1", i64 33
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %"new::Tuple.sroa.0.sroa.6.0..sroa_idx", ptr noundef nonnull align 1 dereferenceable(7) %"new::FlipProposal.sroa.11", i64 7, i1 false)
  store i64 %.fr420, ptr %47, align 8
  %"new::Tuple.sroa.0.sroa.2.0..sroa_idx" = getelementptr inbounds i8, ptr %"new::#calculate##0#calculate##1", i64 16
  store float %62, ptr %"new::Tuple.sroa.0.sroa.2.0..sroa_idx", align 8
  %"new::Tuple.sroa.0.sroa.3.0..sroa_idx" = getelementptr inbounds i8, ptr %"new::#calculate##0#calculate##1", i64 20
  store float %., ptr %"new::Tuple.sroa.0.sroa.3.0..sroa_idx", align 4
  %"new::Tuple.sroa.0.sroa.4.0..sroa_idx" = getelementptr inbounds i8, ptr %"new::#calculate##0#calculate##1", i64 24
  store i64 1, ptr %"new::Tuple.sroa.0.sroa.4.0..sroa_idx", align 8
  %"new::Tuple.sroa.0.sroa.5.0..sroa_idx" = getelementptr inbounds i8, ptr %"new::#calculate##0#calculate##1", i64 32
  store i8 0, ptr %"new::Tuple.sroa.0.sroa.5.0..sroa_idx", align 8
  %root_phi.state = load atomic ptr, ptr %3 unordered, align 8
  %root_phi.state.size_ptr = getelementptr inbounds i8, ptr %root_phi.state, i64 16
  %root_phi.state.size.0.copyload = load i64, ptr %root_phi.state.size_ptr, align 8
  %.not216 = icmp eq i64 %root_phi.state.size.0.copyload, 10000
  br i1 %.not216, label %L191, label %L186

L143:                                             ; preds = %L87
  store i64 %54, ptr %"new::Tuple45", align 1
  store i64 %54, ptr %"new::Tuple76", align 1
  %jl_nothing = load ptr, ptr @jl_nothing, align 8
  store ptr null, ptr %gc_slot_addr_19, align 8
  store ptr null, ptr %gc_slot_addr_18, align 8
  store ptr null, ptr %gc_slot_addr_16, align 8
  store ptr null, ptr %gc_slot_addr_15, align 8
  store ptr null, ptr %gc_slot_addr_14, align 8
  store ptr null, ptr %gc_slot_addr_13, align 8
  store ptr null, ptr %gc_slot_addr_12, align 8
  store ptr null, ptr %gc_slot_addr_11, align 8
  store ptr null, ptr %gc_slot_addr_10, align 8
  store ptr null, ptr %gc_slot_addr_9, align 8
  store ptr null, ptr %gc_slot_addr_8, align 8
  store ptr null, ptr %gc_slot_addr_7, align 8
  %box_Float32 = call ptr @ijl_box_float32(float %62)
  %gc_slot_addr_5583 = getelementptr inbounds ptr, ptr %gcframe2, i64 7
  store ptr %box_Float32, ptr %gc_slot_addr_5583, align 8
  store ptr @"jl_global#9777.jit", ptr %jlcallframe1, align 8
  %65 = getelementptr inbounds ptr, ptr %jlcallframe1, i64 1
  store ptr %32, ptr %65, align 8
  %66 = getelementptr inbounds ptr, ptr %jlcallframe1, i64 2
  store ptr %jl_nothing, ptr %66, align 8
  %67 = getelementptr inbounds ptr, ptr %jlcallframe1, i64 3
  store ptr %box_Float32, ptr %67, align 8
  %jl_f_throw_methoderror_ret = call nonnull ptr @jl_f_throw_methoderror(ptr null, ptr nonnull %jlcallframe1, i32 4)
  call void @llvm.trap()
  unreachable

L186:                                             ; preds = %L133
  store ptr null, ptr %gc_slot_addr_19, align 8
  store ptr null, ptr %gc_slot_addr_18, align 8
  store ptr null, ptr %gc_slot_addr_17, align 8
  store ptr null, ptr %gc_slot_addr_16, align 8
  store ptr null, ptr %gc_slot_addr_15, align 8
  store ptr null, ptr %gc_slot_addr_14, align 8
  store ptr null, ptr %gc_slot_addr_13, align 8
  store ptr null, ptr %gc_slot_addr_12, align 8
  store ptr null, ptr %gc_slot_addr_11, align 8
  store ptr null, ptr %gc_slot_addr_10, align 8
  store ptr null, ptr %gc_slot_addr_9, align 8
  store ptr null, ptr %gc_slot_addr_8, align 8
  store ptr null, ptr %gc_slot_addr_7, align 8
  call void @j_throw_dmrsa_9766(ptr nocapture nonnull readonly @"_j_const#2", i64 signext %root_phi.state.size.0.copyload) #10
  unreachable

L191:                                             ; preds = %L133
  %68 = load ptr, ptr %root_phi.state, align 8
  %69 = getelementptr inbounds { ptr, ptr }, ptr %root_phi.state, i64 0, i32 1
  %70 = load ptr, ptr %69, align 8
  %gc_slot_addr_5597 = getelementptr inbounds ptr, ptr %gcframe2, i64 7
  store ptr %70, ptr %gc_slot_addr_5597, align 8
  %ptls_field751 = getelementptr inbounds i8, ptr %pgcstack, i64 16
  %ptls_load752 = load ptr, ptr %ptls_field751, align 8
  %"new::Array44" = call noalias nonnull align 8 dereferenceable(48) ptr @ijl_gc_small_alloc(ptr %ptls_load752, i32 520, i32 48, i64 4586018256) #22
  %"new::Array44.tag_addr" = getelementptr inbounds i64, ptr %"new::Array44", i64 -1
  store atomic i64 4586018256, ptr %"new::Array44.tag_addr" unordered, align 8
  %71 = getelementptr inbounds i8, ptr %"new::Array44", i64 8
  store ptr %68, ptr %"new::Array44", align 8
  store ptr %70, ptr %71, align 8
  %"new::Array44.size_ptr" = getelementptr inbounds i8, ptr %"new::Array44", i64 16
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %"new::Array44.size_ptr", ptr noundef nonnull align 8 dereferenceable(16) @"_j_const#2", i64 16, i1 false)
  %root_phi5.x = load float, ptr %5, align 4
  %root_phi6.vec = load atomic ptr, ptr %7 unordered, align 8
  %root_phi6.vec.size_ptr = getelementptr inbounds i8, ptr %root_phi6.vec, i64 16
  %root_phi6.vec.size.0.copyload = load i64, ptr %root_phi6.vec.size_ptr, align 8
  %72 = add i64 %.fr420, -1
  %.not219 = icmp ult i64 %72, %root_phi6.vec.size.0.copyload
  br i1 %.not219, label %L215, label %L212

L212:                                             ; preds = %L191
  store i64 %.fr420, ptr %"new::Tuple45", align 1
  store ptr null, ptr %gc_slot_addr_19, align 8
  store ptr null, ptr %gc_slot_addr_18, align 8
  store ptr null, ptr %gc_slot_addr_17, align 8
  store ptr null, ptr %gc_slot_addr_16, align 8
  store ptr null, ptr %gc_slot_addr_15, align 8
  store ptr null, ptr %gc_slot_addr_14, align 8
  store ptr null, ptr %gc_slot_addr_13, align 8
  store ptr null, ptr %gc_slot_addr_12, align 8
  store ptr null, ptr %gc_slot_addr_11, align 8
  store ptr null, ptr %gc_slot_addr_10, align 8
  store ptr null, ptr %gc_slot_addr_9, align 8
  store ptr null, ptr %gc_slot_addr_8, align 8
  store ptr null, ptr %gc_slot_addr_7, align 8
  store ptr null, ptr %gc_slot_addr_5597, align 8
  call void @j_throw_boundserror_9776(ptr nonnull %7, ptr nocapture nonnull readonly %"new::Tuple45") #10
  unreachable

L215:                                             ; preds = %L191
  %root_phi6.offset = load i64, ptr %root_phi6.offset_ptr, align 8
  %memoryref_data49 = load ptr, ptr %root_phi6.vec, align 8
  %memoryref_offset51 = shl i64 %.fr420, 3
  %73 = getelementptr i8, ptr %memoryref_data49, i64 %memoryref_offset51
  %memoryref_data57 = getelementptr i8, ptr %73, i64 -8
  %74 = load i64, ptr %memoryref_data57, align 8
  %75 = fpext float %. to double
  %gc_slot_addr_6 = getelementptr inbounds ptr, ptr %gcframe2, i64 8
  store ptr %"new::Array44", ptr %gc_slot_addr_6, align 8
  %76 = call double @"j_#power_by_squaring#401_9769"(double %75, i64 signext 2)
  %.not220 = icmp ult i64 %72, 10000
  br i1 %.not220, label %L261, label %L258

L258:                                             ; preds = %L215
  store i64 %.fr420, ptr %"new::Tuple170", align 8
  store ptr null, ptr %gc_slot_addr_19, align 8
  store ptr null, ptr %gc_slot_addr_18, align 8
  store ptr null, ptr %gc_slot_addr_17, align 8
  store ptr null, ptr %gc_slot_addr_16, align 8
  store ptr null, ptr %gc_slot_addr_15, align 8
  store ptr null, ptr %gc_slot_addr_14, align 8
  store ptr null, ptr %gc_slot_addr_13, align 8
  store ptr null, ptr %gc_slot_addr_12, align 8
  store ptr null, ptr %gc_slot_addr_11, align 8
  store ptr null, ptr %gc_slot_addr_10, align 8
  store ptr null, ptr %gc_slot_addr_9, align 8
  store ptr null, ptr %gc_slot_addr_8, align 8
  store ptr null, ptr %gc_slot_addr_7, align 8
  store ptr null, ptr %gc_slot_addr_6, align 8
  store ptr %"new::Array44", ptr %gc_slot_addr_5597, align 8
  call void @j_throw_boundserror_9771(ptr nonnull %"new::Array44", ptr nocapture nonnull readonly %"new::Tuple170") #10
  unreachable

L261:                                             ; preds = %L215
  %77 = fptrunc double %76 to float
  %78 = add i64 %74, %root_phi6.offset
  %79 = getelementptr i8, ptr %68, i64 %memoryref_offset
  %memoryref_data67 = getelementptr i8, ptr %79, i64 -4
  %80 = load float, ptr %memoryref_data67, align 4
  %81 = fpext float %80 to double
  store ptr null, ptr %gc_slot_addr_6, align 8
  store ptr null, ptr %gc_slot_addr_5597, align 8
  %82 = call double @"j_#power_by_squaring#401_9769"(double %81, i64 signext 2)
  %83 = fptrunc double %82 to float
  %84 = fsub float %77, %83
  %85 = sitofp i64 %78 to float
  %86 = fmul float %root_phi5.x, %85
  %87 = fmul float %86, %84
  %88 = fadd float %87, 0.000000e+00
  store ptr %3, ptr %0, align 8
  store ptr %9, ptr %1, align 8
  store ptr %11, ptr %50, align 8
  store ptr %13, ptr %51, align 8
  store ptr %40, ptr %52, align 8
  %89 = call float @"j_#calculate##0_9770"(ptr nocapture nonnull readonly %"new::#calculate##0#calculate##1", ptr nocapture nonnull readonly %0, float %88, ptr nocapture nonnull readonly %48, ptr nocapture nonnull readonly %1)
  %root_phi.state68 = load atomic ptr, ptr %3 unordered, align 8
  %root_phi.state68.size_ptr = getelementptr inbounds i8, ptr %root_phi.state68, i64 16
  %root_phi.state68.size.0.copyload = load i64, ptr %root_phi.state68.size_ptr, align 8
  %.not221 = icmp eq i64 %root_phi.state68.size.0.copyload, 10000
  br i1 %.not221, label %L292, label %L287

L287:                                             ; preds = %L261
  store ptr null, ptr %gc_slot_addr_19, align 8
  store ptr null, ptr %gc_slot_addr_18, align 8
  store ptr null, ptr %gc_slot_addr_17, align 8
  store ptr null, ptr %gc_slot_addr_16, align 8
  store ptr null, ptr %gc_slot_addr_15, align 8
  store ptr null, ptr %gc_slot_addr_14, align 8
  store ptr null, ptr %gc_slot_addr_13, align 8
  store ptr null, ptr %gc_slot_addr_12, align 8
  store ptr null, ptr %gc_slot_addr_11, align 8
  store ptr null, ptr %gc_slot_addr_10, align 8
  store ptr null, ptr %gc_slot_addr_9, align 8
  store ptr null, ptr %gc_slot_addr_8, align 8
  store ptr null, ptr %gc_slot_addr_7, align 8
  call void @j_throw_dmrsa_9766(ptr nocapture nonnull readonly @"_j_const#2", i64 signext %root_phi.state68.size.0.copyload) #10
  unreachable

L292:                                             ; preds = %L261
  %.unbox77 = load i64, ptr %49, align 8
  %"[2]_ptr.unbox" = load i64, ptr %"[2]_ptr", align 8
  %90 = mul i64 %"[2]_ptr.unbox", %.unbox77
  %.not224 = icmp sgt i64 %90, %72
  br i1 %.not224, label %L333, label %L312

L312:                                             ; preds = %L292
  store i64 %.fr420, ptr %"new::Tuple76", align 1
  store ptr null, ptr %gc_slot_addr_19, align 8
  store ptr null, ptr %gc_slot_addr_18, align 8
  store ptr null, ptr %gc_slot_addr_17, align 8
  store ptr null, ptr %gc_slot_addr_16, align 8
  store ptr null, ptr %gc_slot_addr_15, align 8
  store ptr null, ptr %gc_slot_addr_14, align 8
  store ptr null, ptr %gc_slot_addr_13, align 8
  store ptr null, ptr %gc_slot_addr_12, align 8
  store ptr null, ptr %gc_slot_addr_11, align 8
  store ptr null, ptr %gc_slot_addr_10, align 8
  store ptr null, ptr %gc_slot_addr_9, align 8
  store ptr null, ptr %gc_slot_addr_8, align 8
  store ptr null, ptr %gc_slot_addr_7, align 8
  call void @j_throw_boundserror_9774(ptr nocapture nonnull readonly %49, ptr nocapture nonnull readonly %"new::Tuple76") #10
  unreachable

L333:                                             ; preds = %L292
  %91 = load ptr, ptr %root_phi.state68, align 8
  %92 = getelementptr i8, ptr %91, i64 %memoryref_offset
  %memoryref_data87 = getelementptr i8, ptr %92, i64 -4
  %93 = load float, ptr %memoryref_data87, align 4
  %94 = fsub float %., %93
  %95 = fmul float %94, 0.000000e+00
  %96 = fsub float %89, %95
  %root_phi.temp = load float, ptr %root_phi.temp_ptr, align 8
  %97 = fcmp ugt float %96, 0.000000e+00
  br i1 %97, label %L348, label %L465

L348:                                             ; preds = %L333
  %root_phi19.idxF = load i64, ptr %root_phi19.idxF_ptr, align 8
  %.not226 = icmp eq i64 %root_phi19.idxF, 1002
  br i1 %.not226, label %L351, label %L353

L351:                                             ; preds = %L348
  %98 = call i64 @j_gen_rand_9772(ptr %32)
  %root_phi19.idxF144.pre = load i64, ptr %root_phi19.idxF_ptr, align 8
  br label %L353

L353:                                             ; preds = %L351, %L348
  %root_phi19.idxF144 = phi i64 [ %root_phi19.idxF, %L348 ], [ %root_phi19.idxF144.pre, %L351 ]
  %root_phi19.vals = load atomic ptr, ptr %root_phi19.vals_ptr unordered, align 8
  %99 = add i64 %root_phi19.idxF144, 1
  store i64 %99, ptr %root_phi19.idxF_ptr, align 8
  %memoryref_data147 = load ptr, ptr %root_phi19.vals, align 8
  %memoryref_byteoffset150 = shl i64 %root_phi19.idxF144, 3
  %memoryref_data155 = getelementptr inbounds i8, ptr %memoryref_data147, i64 %memoryref_byteoffset150
  %100 = load i64, ptr %memoryref_data155, align 8
  %101 = trunc i64 %100 to i32
  %102 = and i32 %101, 8388607
  %103 = or disjoint i32 %102, 1065353216
  %bitcast_coercion156 = bitcast i32 %103 to float
  %104 = fadd float %bitcast_coercion156, -1.000000e+00
  %105 = fneg float %96
  %106 = fdiv float %105, %root_phi.temp
  %107 = fmul float %106, 0x3FF7154760000000
  %108 = call float @llvm.rint.f32(float %107)
  %109 = fptosi float %108 to i32
  %110 = freeze i32 %109
  %111 = fmul contract float %108, 0x3FE62E4000000000
  %112 = fsub contract float %106, %111
  %113 = fmul contract float %108, 0x3EB7F7D1C0000000
  %114 = fsub contract float %112, %113
  %115 = fmul contract float %114, 0x3F2A1D7140000000
  %116 = fadd contract float %115, 0x3F56DA7560000000
  %117 = fmul contract float %114, %116
  %118 = fadd contract float %117, 0x3F811105C0000000
  %119 = fmul contract float %114, %118
  %120 = fadd contract float %119, 0x3FA5554640000000
  %121 = fmul contract float %114, %120
  %122 = fadd contract float %121, 0x3FC5555560000000
  %123 = fmul contract float %114, %122
  %124 = fadd contract float %123, 5.000000e-01
  %125 = fmul contract float %114, %124
  %126 = fadd contract float %125, 1.000000e+00
  %127 = fmul contract float %114, %126
  %128 = fadd contract float %127, 1.000000e+00
  %129 = fcmp ule float %106, 0x40562E4300000000
  br i1 %129, label %L412, label %L463

L412:                                             ; preds = %L353
  %130 = fcmp uge float %106, 0xC059FE3680000000
  br i1 %130, label %L456, label %L463

L456:                                             ; preds = %L412
  %131 = fcmp ugt float %106, 0xC055D58A00000000
  %132 = fmul float %128, 0x3E70000000000000
  %value_phi159 = select i1 %131, float %128, float %132
  %.not227 = icmp eq i32 %110, 128
  %133 = fmul float %value_phi159, 2.000000e+00
  %value_phi161 = select i1 %.not227, float %133, float %value_phi159
  %value_phi158.v = select i1 %131, i32 127, i32 151
  %value_phi158 = add i32 %110, %value_phi158.v
  %134 = sext i1 %.not227 to i32
  %value_phi160 = add i32 %value_phi158, %134
  %135 = shl i32 %value_phi160, 23
  %bitcast_coercion164 = bitcast i32 %135 to float
  %136 = fmul float %value_phi161, %bitcast_coercion164
  br label %L463

L463:                                             ; preds = %L456, %L412, %L353
  %value_phi157 = phi float [ %136, %L456 ], [ 0x7FF0000000000000, %L353 ], [ 0.000000e+00, %L412 ]
  %137 = fcmp olt float %104, %value_phi157
  br i1 %137, label %L465, label %guard_pass189

L465:                                             ; preds = %L463, %L333
  %root_phi18.state89 = load atomic ptr, ptr %23 unordered, align 8
  %root_phi18.state89.size_ptr = getelementptr inbounds i8, ptr %root_phi18.state89, i64 16
  %root_phi18.state89.size.0.copyload = load i64, ptr %root_phi18.state89.size_ptr, align 8
  %.not228 = icmp eq i64 %root_phi18.state89.size.0.copyload, 10000
  br i1 %.not228, label %guard_pass184, label %L473

L473:                                             ; preds = %L465
  store ptr null, ptr %gc_slot_addr_19, align 8
  store ptr null, ptr %gc_slot_addr_18, align 8
  store ptr null, ptr %gc_slot_addr_17, align 8
  store ptr null, ptr %gc_slot_addr_16, align 8
  store ptr null, ptr %gc_slot_addr_15, align 8
  store ptr null, ptr %gc_slot_addr_14, align 8
  store ptr null, ptr %gc_slot_addr_13, align 8
  store ptr null, ptr %gc_slot_addr_12, align 8
  store ptr null, ptr %gc_slot_addr_11, align 8
  store ptr null, ptr %gc_slot_addr_10, align 8
  store ptr null, ptr %gc_slot_addr_9, align 8
  store ptr null, ptr %gc_slot_addr_8, align 8
  store ptr null, ptr %gc_slot_addr_7, align 8
  call void @j_throw_dmrsa_9766(ptr nocapture nonnull readonly @"_j_const#2", i64 signext %root_phi18.state89.size.0.copyload) #10
  unreachable

L554.L559_crit_edge:                              ; preds = %pass110
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(216) %.sroa.0.sroa.0, ptr noundef nonnull align 8 dereferenceable(216) %"new::NamedTuple.sroa.0.sroa.0", i64 216, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %.sroa.0.sroa.11, ptr noundef nonnull align 1 dereferenceable(7) %.sroa.6344, i64 7, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %.sroa.8, ptr noundef nonnull align 8 dereferenceable(24) %.sroa.8206.0..sroa_idx, i64 24, i1 false)
  br label %L559

L558:                                             ; preds = %pass110
  %138 = add i64 %value_phi4, 1
  call void @llvm.lifetime.start.p0(i64 216, ptr nonnull %.sroa.0201.sroa.0)
  call void @llvm.lifetime.start.p0(i64 7, ptr nonnull %.sroa.0201.sroa.11)
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %.sroa.8206)
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %.sroa.9207)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(216) %.sroa.0201.sroa.0, ptr noundef nonnull align 8 dereferenceable(216) %"new::NamedTuple.sroa.0.sroa.0", i64 216, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %.sroa.0201.sroa.11, ptr noundef nonnull align 1 dereferenceable(7) %.sroa.6344, i64 7, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %.sroa.8206, ptr noundef nonnull align 8 dereferenceable(24) %.sroa.8206.0..sroa_idx, i64 24, i1 false)
  br label %L35

L559:                                             ; preds = %L554.L559_crit_edge, %L31.L559_crit_edge
  %.sroa.0.sroa.6.0 = phi i64 [ %.sroa.0.sroa.6.0.copyload, %L31.L559_crit_edge ], [ %.fr420, %L554.L559_crit_edge ]
  %.sroa.0.sroa.7.0 = phi float [ %.sroa.0.sroa.7.0.copyload, %L31.L559_crit_edge ], [ %62, %L554.L559_crit_edge ]
  %.sroa.0.sroa.8.0 = phi float [ %.sroa.0.sroa.8.0.copyload, %L31.L559_crit_edge ], [ %., %L554.L559_crit_edge ]
  %.sroa.0.sroa.9.0 = phi i64 [ %.sroa.0.sroa.9.0.copyload, %L31.L559_crit_edge ], [ 1, %L554.L559_crit_edge ]
  %.sroa.0.sroa.10.0 = phi i8 [ %.sroa.0.sroa.10.0.copyload, %L31.L559_crit_edge ], [ %.sroa.9.0, %L554.L559_crit_edge ]
  %.sroa.6.0 = phi float [ %.sroa.6.0.copyload197, %L31.L559_crit_edge ], [ %96, %L554.L559_crit_edge ]
  %.sroa.7.0 = phi float [ %.sroa.7.0.copyload198, %L31.L559_crit_edge ], [ %root_phi.temp, %L554.L559_crit_edge ]
  %139 = call i64 @jlplt_ijl_hrtime_9759_got.jit()
  %"process::InlineProcess.endtime_ptr" = getelementptr inbounds i8, ptr %"process::InlineProcess", i64 392
  %"process::InlineProcess.endtime.tindex_ptr" = getelementptr inbounds i8, ptr %"process::InlineProcess", i64 400
  store i8 2, ptr %"process::InlineProcess.endtime.tindex_ptr", align 1
  store i64 %139, ptr %"process::InlineProcess.endtime_ptr", align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(216) %sret_return, ptr noundef nonnull align 8 dereferenceable(216) %.sroa.0.sroa.0, i64 216, i1 false)
  %"new::ProcessContext140.sroa.0.sroa.2.0.sret_return.sroa_idx" = getelementptr inbounds i8, ptr %sret_return, i64 216
  store i64 %.sroa.0.sroa.6.0, ptr %"new::ProcessContext140.sroa.0.sroa.2.0.sret_return.sroa_idx", align 8
  %"new::ProcessContext140.sroa.0.sroa.3.0.sret_return.sroa_idx" = getelementptr inbounds i8, ptr %sret_return, i64 224
  store float %.sroa.0.sroa.7.0, ptr %"new::ProcessContext140.sroa.0.sroa.3.0.sret_return.sroa_idx", align 8
  %"new::ProcessContext140.sroa.0.sroa.4.0.sret_return.sroa_idx" = getelementptr inbounds i8, ptr %sret_return, i64 228
  store float %.sroa.0.sroa.8.0, ptr %"new::ProcessContext140.sroa.0.sroa.4.0.sret_return.sroa_idx", align 4
  %"new::ProcessContext140.sroa.0.sroa.5.0.sret_return.sroa_idx" = getelementptr inbounds i8, ptr %sret_return, i64 232
  store i64 %.sroa.0.sroa.9.0, ptr %"new::ProcessContext140.sroa.0.sroa.5.0.sret_return.sroa_idx", align 8
  %"new::ProcessContext140.sroa.0.sroa.6.0.sret_return.sroa_idx" = getelementptr inbounds i8, ptr %sret_return, i64 240
  store i8 %.sroa.0.sroa.10.0, ptr %"new::ProcessContext140.sroa.0.sroa.6.0.sret_return.sroa_idx", align 8
  %"new::ProcessContext140.sroa.0.sroa.7.0.sret_return.sroa_idx" = getelementptr inbounds i8, ptr %sret_return, i64 241
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %"new::ProcessContext140.sroa.0.sroa.7.0.sret_return.sroa_idx", ptr noundef nonnull align 1 dereferenceable(7) %.sroa.0.sroa.11, i64 7, i1 false)
  %"new::ProcessContext140.sroa.2.0.sret_return.sroa_idx" = getelementptr inbounds i8, ptr %sret_return, i64 248
  store float %.sroa.6.0, ptr %"new::ProcessContext140.sroa.2.0.sret_return.sroa_idx", align 8
  %"new::ProcessContext140.sroa.3.0.sret_return.sroa_idx" = getelementptr inbounds i8, ptr %sret_return, i64 252
  store float %.sroa.7.0, ptr %"new::ProcessContext140.sroa.3.0.sret_return.sroa_idx", align 4
  %"new::ProcessContext140.sroa.4.0.sret_return.sroa_idx" = getelementptr inbounds i8, ptr %sret_return, i64 256
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %"new::ProcessContext140.sroa.4.0.sret_return.sroa_idx", ptr noundef nonnull align 8 dereferenceable(24) %.sroa.8, i64 24, i1 false)
  store ptr %3, ptr %return_roots, align 8
  %140 = getelementptr inbounds i8, ptr %return_roots, i64 8
  store ptr %5, ptr %140, align 8
  %141 = getelementptr inbounds i8, ptr %return_roots, i64 16
  store ptr %7, ptr %141, align 8
  %142 = getelementptr inbounds i8, ptr %return_roots, i64 24
  store ptr %9, ptr %142, align 8
  %143 = getelementptr inbounds i8, ptr %return_roots, i64 32
  store ptr %11, ptr %143, align 8
  %144 = getelementptr inbounds i8, ptr %return_roots, i64 40
  store ptr %13, ptr %144, align 8
  %145 = getelementptr inbounds i8, ptr %return_roots, i64 48
  store <2 x ptr> %15, ptr %145, align 8
  %146 = getelementptr inbounds i8, ptr %return_roots, i64 64
  store <2 x ptr> %17, ptr %146, align 8
  %147 = getelementptr inbounds i8, ptr %return_roots, i64 80
  store <2 x ptr> %19, ptr %147, align 8
  %148 = getelementptr inbounds i8, ptr %return_roots, i64 96
  store <2 x ptr> %21, ptr %148, align 8
  %149 = getelementptr inbounds i8, ptr %return_roots, i64 112
  store ptr %23, ptr %149, align 8
  %150 = getelementptr inbounds i8, ptr %return_roots, i64 120
  store <2 x ptr> %25, ptr %150, align 8
  %151 = getelementptr inbounds i8, ptr %return_roots, i64 136
  store <2 x ptr> %27, ptr %151, align 8
  %152 = getelementptr inbounds i8, ptr %return_roots, i64 152
  store ptr %29, ptr %152, align 8
  %frame.prev825 = load ptr, ptr %frame.prev, align 8
  store ptr %frame.prev825, ptr %pgcstack, align 8
  ret void

pass110:                                          ; preds = %guard_pass189, %guard_pass184
  %.sroa.9.0 = phi i8 [ 1, %guard_pass184 ], [ 0, %guard_pass189 ]
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %.sroa.6344, ptr noundef nonnull align 1 dereferenceable(7) %.sroa.10, i64 7, i1 false)
  call void @llvm.lifetime.end.p0(i64 7, ptr nonnull %.sroa.10)
  %"new::NamedTuple.sroa.0.sroa.0.8.sroa_idx" = getelementptr inbounds i8, ptr %"new::NamedTuple.sroa.0.sroa.0", i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(80) %"new::NamedTuple.sroa.0.sroa.0.8.sroa_idx", ptr noundef nonnull align 8 dereferenceable(80) %53, i64 80, i1 false)
  %"new::NamedTuple.sroa.0.sroa.0.88.sroa_idx" = getelementptr inbounds i8, ptr %"new::NamedTuple.sroa.0.sroa.0", i64 88
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(120) %"new::NamedTuple.sroa.0.sroa.0.88.sroa_idx", ptr noundef nonnull align 8 dereferenceable(120) %45, i64 120, i1 false)
  store i64 1, ptr %"algo::CompositeAlgorithm.inc112", align 8
  %"process::InlineProcess.loopidx114" = load i64, ptr %"process::InlineProcess.loopidx_ptr", align 8
  %153 = add i64 %"process::InlineProcess.loopidx114", 1
  store i64 %153, ptr %"process::InlineProcess.loopidx_ptr", align 8
  %.not231.not.not = icmp eq i64 %value_phi4, %value_phi
  br i1 %.not231.not.not, label %L554.L559_crit_edge, label %L558

guard_pass184:                                    ; preds = %L465
  %154 = load ptr, ptr %root_phi18.state89, align 8
  %155 = getelementptr i8, ptr %154, i64 %memoryref_offset
  %memoryref_data106 = getelementptr i8, ptr %155, i64 -4
  store float %., ptr %memoryref_data106, align 4
  call void @llvm.lifetime.start.p0(i64 7, ptr nonnull %.sroa.10)
  br label %pass110

guard_pass189:                                    ; preds = %L463
  call void @llvm.lifetime.start.p0(i64 7, ptr nonnull %.sroa.10)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %.sroa.10, ptr noundef nonnull align 1 dereferenceable(7) %"new::FlipProposal.sroa.11", i64 7, i1 false)
  br label %pass110
}
