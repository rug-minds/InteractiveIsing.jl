# For debugging
# struct TreeID{IDS, Level} end
struct TreeID{IDS} end

Base.show(io::IO, tid::TreeID) = print(io, "Some TreeID")

Base.iterate(tid::TreeID{IDS}, state = 1) where IDS = state <= length(IDS) ? (IDS[state], state + 1) : nothing
Base.eachindex(tid::TreeID) = Base.OneTo(length(getids(tid)))
Base.length(tid::TreeID) = length(getids(tid))


# TreeID(a::Any) = TreeID{tuple(hash(a)), 1}()
TreeID(a::Any) = TreeID{tuple(hash(a))}()

# TreeID(first::UInt64, level, ids::UUID...) = TreeID{tuple(first,ids...), level}()
TreeID(first::UInt64, ids::UUID...) = TreeID{tuple(first,ids...)}()

TreeID() = TreeID(uuid4())
# TreeID(tid::TreeID, id::UUID = uuid4(), level = 1) = TreeID(getids(tid)[1], level, (getids(tid)[2:end])..., id)
TreeID(tid::TreeID, id::UUID = uuid4()) = TreeID(getids(tid)[1], getids(tid)[2:end]..., id)

# nextID(tid::TreeID, level = 1) = TreeID(tid, uuid4(), level)
nextID(tid::TreeID) = TreeID(tid, uuid4())

getids(tid::TreeID{IDS}) where IDS = IDS
# level(tid::TreeID{IDS, Level}) where {IDS, Level} = Level
topid(tid::TreeID) = getids(tid)[1]
Base.getindex(tid::TreeID, idx) = getids(tid)[idx]


mutable struct GenExpressionTree
    id::TreeID
    name::Symbol
    exps::Expr
    tree::Vector{Any}
end

# Init Tree Never used except for overwriting
GenExpressionTree() = GenExpressionTree(TreeID(), :empty, Expr(:block), [])
function setexpr!(tree::GenExpressionTree, expr)
    tree.exps = remove_line_number_nodes(expr)
end

# getid(tree::GenExpressionTree{ID}) where ID = ID.id
getid(tree::GenExpressionTree) = tree.id
getids(tree::GenExpressionTree) = getids(getid(tree))
topid(tree::GenExpressionTree) = topid(getid(tree))
thisid(tree::GenExpressionTree) = getids(tree)[end]
nextID(tree::GenExpressionTree) = nextID(getid(tree))

function GenExpressionTree(id::Union{Nothing, Type{Nothing}, TreeID}, name)
    if isnothing(id) || id == Nothing
        id = TreeID()
    end 
    tree = GenExpressionTree(id, name, Expr(:block), [])
    return tree
end

Base.iterate(tree::GenExpressionTree, state = 1) = state <= length(tree.tree) ? (tree.tree[state], state + 1) : nothing
Base.push!(tree::GenExpressionTree, tree2::GenExpressionTree) = push!(tree.tree, tree2)

is_newtree(tree::GenExpressionTree) = length(getids(tree)) == 1

"""
If the same UUID is passed, then we will merge the trees 
"""
function mergetree(tree1::GenExpressionTree, tree2::GenExpressionTree) 
    if is_newtree(tree2)
        return tree2
    end

    if topid(tree1) == topid(tree2)
        this_tree = tree1
        for id2idx in 1:length(getid(tree2))-1
            id2 = getid(tree2)[id2idx]
            for node in this_tree
                if thisid(node) == id2
                    this_tree = node
                end
            end
        end
        # resize!(this_tree.tree, Base.max(length(this_tree.tree), level(getid(tree2))))
        # this_tree.tree[level(getid(tree2))] = tree2
        push!(this_tree, tree2)
        return tree1
    end
    # If new UUID then just return the new tree
    return tree2
end

function Base.show(io::IO, tree::GenExpressionTree)
    indentio = NextIndentIO(io, VLine())
    println(io, "Name: $(tree.name)")
    println(io, "Expression: $(tree.exps)")
    for node in tree
        # invoke(show, Tuple{IO, typeof(node)}, next(indentio), node)
        show(next(indentio), node)
    end
end

# function Base.show(io::IO, ca::CompositeAlgorithm)
#     indentio = NextIndentIO(io, VLine(), "Composite Algorithm")
#     _intervals = intervals(ca)
#     q_postfixes(indentio, ("\texecuting every $interval time(s)" for interval in _intervals)...)
#     for thisfunc in ca.funcs
#         if thisfunc isa CompositeAlgorithm || thisfunc isa Routine
#             invoke(show, Tuple{IO, typeof(thisfunc)}, next(indentio), thisfunc)
#         else
#             invoke(show, Tuple{IndentIO, Any}, next(indentio), thisfunc)
#         end
#     end
# end