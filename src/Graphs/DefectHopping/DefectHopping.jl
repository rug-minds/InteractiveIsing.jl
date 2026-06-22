export DefectHopping,
       DefectsModel,
       MobileChargeHopping,
       MobileVacancies,
       MobileCharges,
       ChargeHopProposer,
       LocalPotentialShiftCoupling,
       LocalPotentialScaleCoupling,
       ExtFieldShiftCoupling,
       ExternalFieldShiftCoupling,
       ExtFieldChargeCoupling,
       ExternalFieldChargeCoupling,
       CoulombChargeCoupling,
       CoulombChargeShift,
       LocalPotentialShift,
       ExtFieldShift

abstract type AbstractDefectCoupling <: HamiltonianTerm end
const AbstractDefectMode = AbstractDefectCoupling

include("CouplingHamiltonians.jl")
include("Model.jl")
include("Binding.jl")
include("Proposals.jl")
include("Energy.jl")
include("CoulombCoupling.jl")
