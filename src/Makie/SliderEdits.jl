# const dragstop = Dict{Ptr{Nothing}, Observable{Any}}()

# function add_dragstop!(t::Slider, val)
#     return dragstop[pointer_from_objref(t)] = Observable(val)
# end

# set_draggstop(t::Slider, val) = dragstop[pointer_from_objref(t)] = val

# value_dragstop(t::Slider) = dragstop[pointer_from_objref(t)]
# export value_dragstop

# function current_sliderfraction(sl, event, endpoints, sliderrange)
#     fraction = clamp(if sl.horizontal[]
#         (event.px[1] - endpoints[][1][1]) / (endpoints[][2][1] - endpoints[][1][1])
#         else
#              (event.px[2] - endpoints[][1][2]) / (endpoints[][2][2] - endpoints[][1][2])
#               end, 0, 1
#     )

#     newindex = Makie.closest_fractionindex(sliderrange[], fraction)
#     if sl.snap[]
#         fraction = (newindex - 1) / (length(sliderrange[]) - 1)
#     end

#     return fraction
# end


# Makie.initialize_block!(sl::Slider) =
# begin

#     add_dragstop!(sl, 0)
#     topscene = sl.blockscene

#     sliderrange = sl.range

#     onany(sl.linewidth, sl.horizontal) do lw, horizontal
#         if horizontal
#             sl.layoutobservables.autosize[] = (nothing, Float32(lw))
#         else
#             sl.layoutobservables.autosize[] = (Float32(lw), nothing)
#         end
#     end

#     sliderbox = lift(identity, topscene, sl.layoutobservables.computedbbox)

#     endpoints = lift(topscene, sliderbox, sl.horizontal) do bb, horizontal

#         h = GLMakie.height(bb)
#         w = GLMakie.width(bb)

#         if horizontal
#             y = GLMakie.bottom(bb) + h / 2
#             [Point2f(GLMakie.left(bb) + h/2, y),
#              Point2f(GLMakie.right(bb) - h/2, y)]
#         else
#             x = left(bb) + w / 2
#             [Point2f(x, GLMakie.bottom(bb) + w/2),
#              Point2f(x, GLMakie.top(bb) - w/2)]
#         end
#     end

#     # this is the index of the selected value in the slider's range
#     selected_index = Observable(1)
#     setfield!(sl, :selected_index, selected_index)

#     # the fraction on the slider corresponding to the selected_index
#     # this is only used after dragging
#     sliderfraction = lift(topscene, selected_index, sliderrange) do i, r
#         (i - 1) / (length(r) - 1)
#     end

#     dragging = Observable(false)

#     # what the slider actually displays currently (also during dragging when
#     # the slider position is in an "invalid" position given the slider's range)
#     displayed_sliderfraction = Observable(0.0)

#     on(topscene, sliderfraction) do frac
#         # only update displayed fraction through sliderfraction if not dragging
#         # dragging overrides the value so there is clear mouse interaction
#         if !dragging[]
#             displayed_sliderfraction[] = frac
#         end
#     end

#     # when the range is changed, switch to closest value
#     on(topscene, sliderrange) do rng
#         selected_index[] = Makie.closest_index(rng, sl.value[])
#     end

#     on(topscene, selected_index) do i
#         sl.value[] = sliderrange[][i]
#         if !dragging[]
#             value_dragstop(sl)[] = sliderrange[][i]
#         end
#     end

#     # initialize slider value with closest from range
#     selected_index[] = Makie.closest_index(sliderrange[], sl.startvalue[])

#     middlepoint = lift(topscene, endpoints, displayed_sliderfraction) do ep, sf
#         Point2f(ep[1] .+ sf .* (ep[2] .- ep[1]))
#     end

#     linepoints = lift(topscene, endpoints, middlepoint) do eps, middle
#         [eps[1], middle, middle, eps[2]]
#     end

#     linecolors = lift(topscene, sl.color_active_dimmed, sl.color_inactive) do ca, ci
#         [ca, ci]
#     end

#     endbuttons = scatter!(topscene, endpoints, color = linecolors,
#         markersize = sl.linewidth, strokewidth = 0, inspectable = false, marker=Circle)

#     linesegs = linesegments!(topscene, linepoints, color = linecolors,
#         linewidth = sl.linewidth, inspectable = false)

#     button_magnification = Observable(1.0)
#     buttonsize = lift(*, topscene, sl.linewidth, button_magnification)
#     button = scatter!(topscene, middlepoint, color = sl.color_active, strokewidth = 0,
#         markersize = buttonsize, inspectable = false, marker=Circle)

#     mouseevents = addmouseevents!(topscene, sl.layoutobservables.computedbbox)

#     onmouseleftdrag(mouseevents) do event
#         dragging[] = true

#         fraction = current_sliderfraction(sl, event, endpoints, sliderrange)
#         displayed_sliderfraction[] = fraction

#         newindex = Makie.closest_fractionindex(sliderrange[], fraction)
#         if sl.snap[]
#             fraction = (newindex - 1) / (length(sliderrange[]) - 1)
#         end

#         newindex = Makie.closest_fractionindex(sliderrange[], fraction)
#         if selected_index[] != newindex
#             selected_index[] = newindex
#         end

#         return Consume(true)
#     end

#     onmouseleftdragstop(mouseevents) do event
#         dragging[] = false
#         # selected_index should be set correctly in onmouseleftdrag
#         value_dragstop(sl)[] = sliderrange[][selected_index[]]
#         # adjust slider to closest legal value
#         # This line is not necessary?
#         # sliderfraction[] = sliderfraction[]
#         linecolors[] = [sl.color_active_dimmed[], sl.color_inactive[]]
#         return Consume(true)
#     end

#     onmouseleftdown(mouseevents) do event
#         pos = event.px
#         dim = sl.horizontal[] ? 1 : 2
#         frac = (pos[dim] - endpoints[][1][dim]) / (endpoints[][2][dim] - endpoints[][1][dim])
#         selected_index[] = Makie.closest_fractionindex(sliderrange[], frac)
#         value_dragstop(sl)[] = sliderrange[][selected_index[]]
#         # linecolors[] = [color_active[], color_inactive[]]
#         return Consume(true)
#     end

#     onmouseleftdoubleclick(mouseevents) do event
#         selected_index[] = Makie.closest_index(sliderrange[], sl.startvalue[])
#         return Consume(true)
#     end

#     onmouseenter(mouseevents) do event
#         button_magnification[] = 1.25
#         return Consume(false)
#     end

#     onmouseout(mouseevents) do event
#         button_magnification[] = 1.0
#         linecolors[] = [sl.color_active_dimmed[], sl.color_inactive[]]
#         return Consume(false)
#     end

#     # trigger autosize through linewidth for first layout
#     notify(sl.linewidth)
#     sl
# end

# Makie.set_close_to!(slider::Slider, value) =
# begin
#     closest = Makie.closest_index(slider.range[], value)
#     slider.selected_index = closest
#     slider.range[][closest]
#     set_draggstop(slider,slider.range[][closest])
# end