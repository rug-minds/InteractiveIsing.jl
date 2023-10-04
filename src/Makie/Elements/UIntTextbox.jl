function UIntTextbox(f; upper = 100, onfunc::Function = identity, kwargs...)
    if typeof(upper) <: Number
        upper = () -> upper
    end
    # Validator
    val(x) = try 0 < parse(UInt64,x) < upper(); catch; false; end
    # Textbox
    tb = Textbox(f; validator = val, defocus_on_submit = true, reset_on_defocus = true, kwargs...)
    on(tb.stored_string) do s
        if s != nothing
            num = parse(UInt64, s)
            onfunc(num)
            tb.stored_string[] = nothing
        end
    end
    return tb
end