#!/usr/bin/env julia
# Combine per-rank DNS 3D-fields snapshots into a single merged JLD2 file.
#
# Usage:
#   julia --project analysis/combine_dns_snapshots.jl <input> [output]
#
# `<input>` may be:
#   - a single .jld2 file (no-op merge / pass-through copy);
#   - a directory containing one or more `*_3d_fields*.jld2` files (one per
#     rank if the run was distributed). Files are treated as rank shards
#     and stitched.
#
# `[output]` is the merged-file path; defaults to
#     <input_dir_or_file>_merged.jld2
#
# The merged file has the same JLD2 structure as a single-rank
# Oceananigans JLD2Writer 3D-fields output, so downstream code can load
# it via `FieldTimeSeries(merged_file, "u")` (etc.) without changes.
#
# Schema we rely on:
#   serialized/{grid, ...}                 — global grid info (same across ranks)
#   timeseries/{u, v, w, c, t}             — per-field timeseries groups
#   timeseries/<field>/serialized/{indices, location, boundary_conditions}
#       indices ::Tuple{UnitRange,UnitRange,UnitRange} — *global* index
#       range that this rank's array occupies in the global field. Single-
#       rank files have indices == (1:Nx, 1:Ny, 1:Nz); per-rank distributed
#       files have a strict subset.
#   timeseries/<field>/<iteration>         — the array, size matching `indices`.

using JLD2
using Printf

"""
    discover_rank_files(input)

Resolve `input` (a file or directory) into the list of rank-shard files
to combine. If `input` is a single file, returns `[input]`. If `input`
is a directory, looks for `*_3d_fields*.jld2` files inside it.
"""
function discover_rank_files(input::AbstractString)
    if isfile(input)
        return [abspath(input)]
    end
    if isdir(input)
        all_entries = readdir(input; join=true)
        candidates = sort(filter(p -> isfile(p) &&
                                       endswith(p, ".jld2") &&
                                       occursin("_3d_fields", basename(p)),
                                 all_entries))
        if isempty(candidates)
            error("no *_3d_fields*.jld2 files found in $input")
        end
        return abspath.(candidates)
    end
    error("input $input is neither a file nor a directory")
end

"""
    field_names(file)

Return the list of field-name strings stored in `timeseries/`,
excluding the time vector `t`.
"""
function field_names(file)
    haskey(file, "timeseries") || error("file has no timeseries group")
    return [k for k in keys(file["timeseries"]) if k != "t"]
end

"""
    iteration_keys(file, name)

Sorted (numeric) list of iteration keys for `timeseries/<name>`,
excluding the `serialized` subgroup.
"""
function iteration_keys(file, name)
    g = file["timeseries/$name"]
    iters = Int[]
    for k in keys(g)
        k == "serialized" && continue
        try
            push!(iters, parse(Int, k))
        catch
            # skip non-numeric keys
        end
    end
    sort!(iters)
    return iters
end

"""
    indices_for(file, name)

Read `timeseries/<name>/serialized/indices` — a Tuple of UnitRanges
giving the global index range this rank holds for field `name`.
"""
function indices_for(file, name)
    return file["timeseries/$name/serialized/indices"]
end

"""
    global_extent_for(rank_files, name)

Across all rank files, compute the global extent (Nx, Ny, Nz) for field
`name`. Each rank reports its own indices; the global extent is the
maximum stop value seen on each axis.
"""
function global_extent_for(rank_files, name)
    Nx = Ny = Nz = 0
    for fpath in rank_files
        jldopen(fpath, "r") do file
            ix, iy, iz = indices_for(file, name)
            Nx = max(Nx, last(ix))
            Ny = max(Ny, last(iy))
            Nz = max(Nz, last(iz))
        end
    end
    return Nx, Ny, Nz
end

"""
    stitch_field_iteration(rank_files, name, iter, Nx, Ny, Nz, T)

Read one field's array at one iteration from each rank shard and place
it into a freshly-allocated global array of size (Nx, Ny, Nz).
"""
function stitch_field_iteration(rank_files, name, iter, Nx, Ny, Nz, T)
    out = zeros(T, Nx, Ny, Nz)
    for fpath in rank_files
        jldopen(fpath, "r") do file
            haskey(file, "timeseries/$name/$iter") || return
            ix, iy, iz = indices_for(file, name)
            arr = file["timeseries/$name/$iter"]
            # Shape check: arr should be sized as length-of-each-range
            sz = (length(ix), length(iy), length(iz))
            size(arr) == sz || error(
                "shape mismatch in $fpath / timeseries/$name/$iter: got $(size(arr)), expected $sz")
            out[ix, iy, iz] = arr
        end
    end
    return out
end

"""
    copy_metadata!(out_file, in_file, name)

Copy non-iteration metadata (serialized/* under timeseries/<name>) from
the input file to the output file. Indices are *overwritten* with the
global indices.
"""
function copy_field_metadata!(out_file, in_file, name, Nx, Ny, Nz)
    src = in_file["timeseries/$name/serialized"]
    dst_path = "timeseries/$name/serialized"
    for k in keys(src)
        v = in_file["timeseries/$name/serialized/$k"]
        if k == "indices"
            # Overwrite with global indices
            out_file["$dst_path/indices"] = (1:Nx, 1:Ny, 1:Nz)
        else
            out_file["$dst_path/$k"] = v
        end
    end
end

"""
    copy_top_level_metadata!(out_file, in_file)

Copy the file-level `serialized/*` group and any other top-level
non-timeseries entries.
"""
function copy_top_level_metadata!(out_file, in_file)
    for k in keys(in_file)
        k == "timeseries" && continue
        if k == "serialized"
            for sk in keys(in_file[k])
                out_file["serialized/$sk"] = in_file["serialized/$sk"]
            end
        else
            out_file[k] = in_file[k]
        end
    end
end

function combine_snapshots(input::AbstractString;
                           output::Union{Nothing,AbstractString} = nothing,
                           field_names_filter = nothing)
    rank_files = discover_rank_files(input)
    @info "Found $(length(rank_files)) shard(s) to combine"

    if output === nothing
        base = isfile(input) ? input : joinpath(input, basename(rstrip(input, '/')))
        output = base * "_merged.jld2"
    end

    # Single-rank pass-through (just copy)
    if length(rank_files) == 1
        @info "Single shard — copying $(rank_files[1]) → $output"
        cp(rank_files[1], output; force=true)
        return output
    end

    # Multi-rank stitch
    isfile(output) && rm(output; force=true)

    jldopen(rank_files[1], "r") do ref_file
        names = field_names_filter === nothing ?
                field_names(ref_file) :
                String.(field_names_filter)
        @info "Fields: $(join(names, ", "))"

        # Determine iteration list from the t timeseries (reference rank)
        # — should be identical across ranks.
        t_iters = iteration_keys(ref_file, "t")
        @info "Iterations: $(length(t_iters))"

        jldopen(output, "w") do out
            # Copy file-level non-timeseries metadata once
            copy_top_level_metadata!(out, ref_file)

            # Copy the time vector entries from the reference rank
            for it in t_iters
                t_val = ref_file["timeseries/t/$it"]
                out["timeseries/t/$it"] = t_val
            end

            for name in names
                @info "Stitching field: $name"
                Nx, Ny, Nz = global_extent_for(rank_files, name)
                @info "  global extent: $(Nx) × $(Ny) × $(Nz)"

                # Copy per-field serialized metadata (with indices overwritten)
                copy_field_metadata!(out, ref_file, name, Nx, Ny, Nz)

                # Determine eltype from rank0 first iteration array
                iters = iteration_keys(ref_file, name)
                isempty(iters) && continue
                T = eltype(ref_file["timeseries/$name/$(iters[1])"])

                for iter in iters
                    arr = stitch_field_iteration(rank_files, name, iter, Nx, Ny, Nz, T)
                    out["timeseries/$name/$iter"] = arr
                end
            end
        end
    end

    @info "Merged output: $output"
    return output
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 1 || error("usage: julia combine_dns_snapshots.jl <input> [output]")
    input = ARGS[1]
    output = length(ARGS) >= 2 ? ARGS[2] : nothing
    combine_snapshots(input; output)
end
