import CairoMakie as CM
"""
Plot using CairoMakie backend so that the output can be displayed inline in Jupyter notebooks.
Func should return a figure or something that is displayable.
"""
function inlineplot(func)
    CM.activate!()
    display(func())
    GLMakie.activate!()
end
export inlineplot
