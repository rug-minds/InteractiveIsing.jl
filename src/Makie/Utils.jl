import CairoMakie as CM
"""
Plot using CairoMakie backend so that the output can be displayed inline in Jupyter notebooks.
"""
function inlineplot(func)
    CM.activate!()
    func()
    GLMakie.activate!()
end
export inlineplot
