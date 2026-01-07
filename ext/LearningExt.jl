module LearningExt
    using Lux, InteractiveIsing

    struct IsingLearningModel{IG, IL, P} <: Lux.AbstractModel
        graph::IG
        layers::IL
        process::P
    end

    


end