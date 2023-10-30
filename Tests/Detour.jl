struct Outer
    inner::NamedTuple
end

getparam(outer::Outer, ) = outer.inner.x