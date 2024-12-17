using MacroTools
function accessor_var(exp)
    @capture(exp, accssr_(strct_) )
    name = Symbol(strct, :_, accssr)
    expr = quote $name = $accssr(strct) end
    remove_line_number_nodes!(expr)
end

accessor_var(:(state(g) + c))

