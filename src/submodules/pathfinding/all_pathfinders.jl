export Pathfinding

"""
    Pathfinding
Submodule containing functionality for path-finding based on the A* algorithm.
Currently available only for [`GridSpace`](@ref).

You can enable path-finding and set it's options by passing an instance of a
[`Pathfinding.Pathfinder`](@ref) struct to the `pathfinder`
parameter of the [`GridSpace`](@ref) constructor. During the simulation, call
[`Pathfinding.set_target!`](@ref) to set the target destination for an agent.
This triggers the algorithm to calculate a path from the agent's current position to the one
specified. You can alternatively use [`Pathfinding.set_best_target!`](@ref) to choose the
best target from a list. Once a target has been set, you can move an agent one step along
its precalculated path using the [`move_along_route!`](@ref) function.

Refer to the [Maze Solver](@ref) and [Mountain Runners](@ref) examples using path-finding
and see the available functions below as well.
"""
module Pathfinding

using Agents
using DataStructures
using LinearAlgebra

include("metrics.jl")
include("grid_pathfinder.jl")

export CostMetric,
    DirectDistance,
    MaxDistance,
    HeightMap,
    Pathfinder,
    Profile,
    delta_cost,
    set_target!,
    set_best_target!,
    heightmap,
    walkmap

struct Pathfinder{M,W}
    agent_profile::M
    profiles::W
end

Pathfinder(space::GridSpace{D,P}, profile::Profile) where {D,P} = 
    Pathfinder(nothing, AStar(size(space.s), P, profile))

Pathfinder(space::GridSpace{D,P}, profiles::Vector{Profile}) where {D,P} = 
    Pathfinder(Dict{Int,Int}(), [AStar(size(space.s), P, p) for p in profiles])

    """
    Pathfinding.set_target!(agent, target::NTuple{D,Int}, model)
Calculate and store the shortest path to move the agent from its current position to
`target` (a grid position e.g. `(1, 5)`) for models using [`Pathfinding`](@ref).

Use this method in conjuction with [`move_along_route!`](@ref).
"""
function set_target!(
    agent::A,
    target::Dims{D},
    model::ABM{<:GridSpace{D},A,F,P,R,<:Pathfinder{Nothing}},
) where {D,A<:AbstractAgent,F,P,R}
    model.pathfinder.profiles.agent_paths[agent.id] =
        find_path(model.pathfinder.profiles, agent.pos, target)
end

function set_target!(
    agent::A,
    target::Dims{D},
    model::ABM{<:GridSpace{D},A,F,P,R,<:Pathfinder},
    profile
) where {D,A<:AbstractAgent,F,P,R}
    model.pathfinder.profiles[profile].agent_paths[agent.id] =
        find_path(model.pathfinder.profiles[profile], agent.pos, target)
    model.pathfinder.agent_profile[agent.id] = profile
end

"""
    Pathfinding.set_best_target!(agent, targets::Vector{NTuple{D,Int}}, model)

Calculate and store the best path to move the agent from its current position to
a chosen target position taken from `targets` for models using [`Pathfinding`](@ref).

The `condition = :shortest` keyword retuns the shortest path which is shortest
(allowing for the conditions of the models pathfinder) out of the possible target
positions. Alternatively, the `:longest` path may also be requested.

Returns the position of the chosen target.
"""
# function set_best_target!(
#     agent::A,
#     targets::Vector{Dims{D}},
#     model::ABM{<:GridSpace{D,P,<:AStar{D}},A};
#     condition::Symbol = :shortest,
# ) where {D,P,A<:AbstractAgent}
#     @assert condition ∈ (:shortest, :longest)
#     compare = condition == :shortest ? (a, b) -> a < b : (a, b) -> a > b
#     best_path = Path{D}()
#     best_target = nothing
#     for target in targets
#         path = find_path(model.space.pathfinder, agent.pos, target)
#         if isempty(best_path) || compare(length(path), length(best_path))
#             best_path = path
#             best_target = target
#         end
#     end

#     model.space.pathfinder.agent_paths[agent.id] = best_path
#     return best_target
# end

# Agents.is_stationary(
#     agent::A,
#     model::ABM{<:GridSpace{D,P,<:AStar{D}},A},
# ) where {D,P,A<:AbstractAgent} = isempty(agent.id, model.space.pathfinder)

Base.isempty(id::Int, pathfinder::AStar) =
    !haskey(pathfinder.agent_paths, id) || isempty(pathfinder.agent_paths[id])

"""
    Pathfinding.heightmap(model)
Return the heightmap of a [`Pathfinding.Pathfinder`](@ref) if the
[`Pathfinding.HeightMap`](@ref) metric is in use, `nothing` otherwise.

It is possible to mutate the map directly, for example
`Pathfinding.heightmap(model)[15, 40] = 115`
or `Pathfinding.heightmap(model) .= rand(50, 50)`. If this is mutated,
a new path needs to be planned using [`Pathfinding.set_target!`](@ref).
"""
# function heightmap(model::ABM{<:GridSpace{D,P,<:AStar{D}}}) where {D,P}
#     if model.space.pathfinder.cost_metric isa HeightMap
#         return model.space.pathfinder.cost_metric.hmap
#     else
#         return nothing
#     end
# end

"""
    Pathfinding.walkmap(model)
Return the walkable map of a [`Pathfinding.Pathfinder`](@ref).

It is possible to mutate the map directly, for example
`Pathfinding.walkmap(model)[15, 40] = false`.
If this is mutated, a new path needs to be planned using [`Pathfinding.set_target!`](@ref).
"""
# walkmap(model::ABM{<:GridSpace{D,P,<:AStar{D}}}) where {D,P} =
#     model.space.pathfinder.walkable

"""
    move_along_route!(agent, model_with_pathfinding)
Move `agent` for one step along the route toward its target set by [`Pathfinding.set_target!`](@ref)
for agents on a [`GridSpace`](@ref) using a [`Pathfinding.Pathfinder`](@ref).
If the agent does not have a precalculated path or the path is empty, it remains stationary.
"""
function Agents.move_along_route!(
    agent::A,
    model::ABM{<:GridSpace{D},A,F,P,R,<:Pathfinder{Nothing}},
) where {D,A<:AbstractAgent,F,P,R}
    isempty(agent.id, model.pathfinder.profiles) && return

    move_agent!(agent, first(model.pathfinder.profiles.agent_paths[agent.id]), model)
    popfirst!(model.pathfinder.profiles.agent_paths[agent.id])
end

function Agents.kill_agent!(
    agent::A,
    model::ABM{<:GridSpace{D},A,F,P,R,<:Pathfinder{Nothing}},
) where {D,A<:AbstractAgent,F,P,R}
    delete!(model.pathfinder.profiles.agent_paths, agent.id)
    delete!(model.agents, agent.id)
    Agents.remove_agent_from_space!(agent, model)
end

function Agents.kill_agent!(
    agent::A,
    model::ABM{<:GridSpace{D},A,F,P,R,<:Pathfinder},
) where {D,A<:AbstractAgent,F,P,R}
    profile = model.pathfinder.agent_profile[agent.id]
    delete!(model.pathfinder.profiles[profile].agent_paths, agent.id)
    delete!(model.agents, agent.id)
    Agents.remove_agent_from_space!(agent, model)
end

end