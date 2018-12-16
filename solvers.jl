
using DataStructures

import Base.isequal
import Base.(==)
import Base.setindex!
import Base.getindex

function solve011(filename)
    println(filename)
    basefreq = 0
    visited = Set([basefreq])
    open(filename) do file
        for line in eachline(file)
            change = parse(Int, line)
            basefreq += change
        end
    end
    return basefreq
end

function solve012(filename)
    nstep = 0
    basefreq = 0
    visited = Set([basefreq])
    changes = readlines(filename)
    while true
        for change_str in changes
            change = parse(Int, change_str)
            basefreq += change
            if in(basefreq, visited)
                return basefreq
            end
            union!(visited, Set([basefreq]))
        end
    end
end

function solve021(filename)
    count2, count3 = open(filename) do input
        count2 = 0
        count3 = 0
        counts = zeros(Int, 26)
        offset = Int('a')
        for id in eachline(input)
            # println(id)
            for char in id
                loc = 1 + Int(char) - offset
                counts[loc] += 1
            end
            # println(counts)
            n2 = any(counts .== 2)
            n3 = any(counts .== 3)
            # println("$n2 $n3")
            count2 += Int(n2)
            count3 += Int(n3)
            # println("$count2 $count3")
            # Reset and check next id.
            counts .= 0
        end
        return count2, count3
    end
    return count2 * count3
end

function solve022(filename)
    ids = readlines(filename)
    dist = id_dist.(reshape(ids, :, 1), reshape(ids, 1, :))
    # println(dist)

    id_indx = findfirst(dist) do d
        d == 1
    end
    correct_id1 = ids[id_indx[1]]
    correct_id2 = ids[id_indx[2]]
    println(correct_id1)
    println(correct_id2)
    matching = Array{Char}(undef, length(correct_id1) - 1)
    i = 1
    for cc in zip(correct_id1, correct_id2)
        c1, c2 = cc
        if c1 == c2
            matching[i] = c1
            i += 1
        end
    end
    return string(matching...)
end

function id_dist(id1, id2)
    diff = 0
    # println(id1)
    # println(id2)
    for cc in zip(split(id1, ""), split(id2, ""))
        c1, c2 = cc
        diff += Int(c1 != c2)
    end
    # println("dist $diff")
    return diff
end

function solve031(filename)
    claim_count = count_claims(readlines(filename))
    # display(claim_count[1:9, 1:9]); println()
    return sum(claim_count .>= 2)
end

function count_claims(claims)
    claim_count = zeros(Int, 1000, 1000)
    for claim in claims
        id, left, top, width, height = parse_claim(claim)
        claim_count[1+top:top+height, 1+left:left+width] .+= 1
    end
    return claim_count
end

function parse_claim(claim_str)
    claim_regex = r"#([0-9]*) @ ([0-9]*),([0-9]*): ([0-9]*)x([0-9]*)"
    id, left, top, width, height = match(claim_regex, claim_str).captures
    left, top, width, height = parse.(Int, [left, top, width, height])
    return id, left, top, width, height
end

function solve032(filename)
    claims = readlines(filename)
    claim_count = count_claims(claims)
    # display(claim_count[1:9, 1:9]); println()
    for claim in claims
        id, left, top, width, height = parse_claim(claim)
        claimed_section = claim_count[1+top:top+height, 1+left:left+width]
        # display(claimed_section); println()
        if all(claimed_section .== 1)
            return id
        end
    end
end

function solve041(filename)
    records = readlines(filename)
    minutes_asleep = create_sleep_table(records)
    most_sleep = first(sort(
        collect(minutes_asleep), by=entry->sum(entry[2]), rev=true))
    most_sleep_id = first(most_sleep)
    id_number = parse(Int, most_sleep_id[2:lastindex(most_sleep_id)])
    asleep = last(most_sleep)
    most_sleep_minute = argmax(asleep) - 1
    return id_number * most_sleep_minute
end

function create_sleep_table(records)
    sort!(records)
    n = length(records)
    minutes_asleep = Dict{String, Array{Int}}()
    record_regex = r"\[.*:([0-9]*)\].*(#[0-9]*|falls|wakes).*"
    asleep = Array{Int}(undef, 60)
    sleep_start::Int = 0
    sleep_end::Int = 0
    for record in records
        println(record)
        m = match(record_regex, record)
        if m[2][1] == '#'
            id = m[2]
            if haskey(minutes_asleep, id)
                asleep = minutes_asleep[id]
            else
                asleep = zeros(Int, 60)
                minutes_asleep[id] = asleep
            end
        elseif m[2] == "falls"
            sleep_start = parse(Int, m[1])
        else
            sleep_end = parse(Int, m[1])
            # Array is 1 to 60, minutes 0 to 59, so offset by 1.
            asleep[1+sleep_start:sleep_end] .+= 1
        end
    end
    return minutes_asleep
end

function print_sleep_table(sleep_table)
    print("    ")
    for i in 1:60
        print(div(i-1, 10))
    end
    print("\n    ")
    for i in 1:60
        print(mod(i-1, 10))
    end
    println()

    for entry in sleep_table
        id, asleep = entry
        print("$id ")
        for a in asleep
            print(a)
        end
        println()
    end
end

function solve042(filename)
    records = readlines(filename)
    minutes_asleep = create_sleep_table(records)
    print_sleep_table(minutes_asleep)
    most_sleep = first(sort(
        collect(minutes_asleep), by=entry->maximum(entry[2]), rev=true))
    most_sleep_id = first(most_sleep)
    id_number = parse(Int, most_sleep_id[2:lastindex(most_sleep_id)])
    asleep = last(most_sleep)
    most_sleep_minute = argmax(asleep) - 1
    return id_number, most_sleep_minute, id_number * most_sleep_minute
end

function solve051(filename)
    polymer = readlines(filename)[1]
    remaining = react_polymer(polymer)
    return length(remaining)
end

function react_polymer(polymer)
    # println(polymer)
    remaining = Stack{Char}()
    current = 'a'
    for current in polymer
        # println(current)
        if length(remaining) > 0
            lastc = pop!(remaining)
            if lowercase(lastc) == lowercase(current) && lastc != current
                # lastc and current cancel, so do nothing and they are gone.
            else
                push!(remaining, lastc)
                push!(remaining, current)
            end
        else
            push!(remaining, current)
        end
        # println(remaining)
    end
    return remaining
end

function solve052(filename)
    polymer = readlines(filename)[1]
    polymer_lengths = Array{Int}(undef, 26)
    for entry in enumerate("abcdefghijklmnopqrstuvwxyz")
        i, c = entry
        stripped = strip_polymer(polymer, c)
        # println(reverse(collect(stripped)))
        remaining = react_polymer(reverse(collect(stripped)))
        n = length(remaining)
        println(entry..., ": ", n)
        polymer_lengths[i] = n
    end
    return minimum(polymer_lengths)
end

function strip_polymer(polymer, c)
    remaining = Stack{Char}()
    for current in polymer
        if lowercase(c) != lowercase(current)
            push!(remaining, current)
        end
    end
    # println(c)
    # println(remaining)
    return remaining
end

function solve061(filename)
    point_strs = readlines(filename)
    points = transpose(parse.(Int, hcat(split.(point_strs, ",")...)))
    # display(points)
    dist = Array{Int}(undef, size(points, 1))
    grid = zeros(Int, maximum(points[:,1])+2, maximum(points[:,2])+2)
    grid_x_axis, grid_y_axis = axes(grid)
    for ix in grid_x_axis, iy in grid_y_axis
        x, y = ix-1, iy-1
        for i in axes(points,1)
            dist[i] = manhattan_dist([x,y], points[i,:])
        end
        # Leave ties unclaimed.
        claimed = argmin(dist)
        if sum(dist .== dist[claimed]) > 1
            continue
        end
        grid[ix, iy] = claimed
    end
    # display(grid)

    # Look at boundary to find unbounded point regions.
    bounded = fill(true, size(points, 1))
    for ix in grid_x_axis
        iymin, iymax = firstindex(grid, 2), lastindex(grid, 2)
        if grid[ix, iymin] > 0
            bounded[grid[ix, iymin]] = false
        end
        if grid[ix, iymax] > 0
            bounded[grid[ix, iymax]] = false
        end
    end
    for iy in grid_y_axis
        ixmin, ixmax = firstindex(grid, 1), lastindex(grid, 1)
        if grid[ixmin, iy] > 0
            bounded[grid[ixmin, iy]] = false
        end
        if grid[ixmax, iy] > 0
            bounded[grid[ixmax, iy]] = false
        end
    end
    # display(bounded)

    areas = Array{Int}(undef, size(points, 1))
    for i in axes(points, 1)
        areas[i] = sum(grid .== i)
    end
    largest_bounded_area = maximum(areas[bounded])
    return largest_bounded_area
end

function manhattan_dist(x1, x2)
    abs(x1[1]-x2[1]) + abs(x1[2]-x2[2])
end

function solve062(filename)
    MAX_DIST = 10000
    point_strs = readlines(filename)
    points = transpose(parse.(Int, hcat(split.(point_strs, ",")...)))
    # display(points)
    dist = Array{Int}(undef, size(points, 1))
    grid = zeros(Int, maximum(points[:,1])+2, maximum(points[:,2])+2)
    grid_x_axis, grid_y_axis = axes(grid)
    for iy in grid_y_axis, ix in grid_x_axis
        x, y = ix-1, iy-1
        for i in axes(points,1)
            dist[i] = manhattan_dist([x,y], points[i,:])
        end
        if sum(dist) < MAX_DIST
            grid[ix, iy] = 1
        end
    end
    # display(grid)
    return sum(grid)
end

function solve071(filename)
    nodes, pred_dag, succ_dag = open(build_dag, filename)
    next_nodes = binary_minheap(Char)
    remaining_nodes = Set(nodes)
    completed_nodes = fill('0', length(remaining_nodes))
    schedule_nodes!(next_nodes, remaining_nodes, pred_dag)
    # display(pred_dag)
    # display(next_nodes)
    # display(remaining_nodes)
    # display(completed_nodes)
    for i in axes(completed_nodes, 1)
        n = pop!(next_nodes)
        completed_nodes[i] = n
        clear_pred!(pred_dag, n)
        schedule_nodes!(next_nodes, remaining_nodes, pred_dag)
        # display(pred_dag)
        # display(next_nodes)
        # display(remaining_nodes)
        # display(completed_nodes)
    end
    return string(completed_nodes...)
end

function schedule_nodes!(next_nodes, remaining_nodes, pred_dag)
    # Find workable nodes and queue them up.
    for n in copy(remaining_nodes)
        if length(pred_dag[n]) == 0
            push!(next_nodes, n)
            pop!(remaining_nodes, n)
        end
    end
end

function clear_pred!(pred_dag, n)
    for arcset in pred_dag
        orig, pred = arcset
        if n in pred
            pop!(pred, n)
        end
    end
end

function build_dag(input)
    nodes = Set{Char}()
    pred_dag = Dict{Char, Set{Char}}()
    succ_dag = Dict{Char, Set{Char}}()
    for line = eachline(input)
        step_re = r"Step ([A-Z]).*step ([A-Z]).*"
        pred, succ = match(step_re, line).captures
        pred, succ = pred[1], succ[1]  # String to Char.
        if haskey(pred_dag, succ)
            pred_dag[succ] = push!(pred_dag[succ], pred)
        else
            pred_dag[succ] = Set{Char}([pred])
        end
        if haskey(succ_dag, pred)
            succ_dag[pred] = push!(succ_dag[pred], succ)
        else
            succ_dag[pred] = Set{Char}([succ])
        end
        push!(nodes, pred)
        push!(nodes, succ)
    end
    for n in nodes
        if !haskey(pred_dag, n); pred_dag[n] = Set{Char}(); end
        if !haskey(succ_dag, n); succ_dag[n] = Set{Char}(); end
    end
    return nodes, pred_dag, succ_dag
end

function solve072(filename)
    N_WORKERS = 5
    TIME_OFFSET = 60
    CHAR_OFFSET = Int('A') - 1
    time_required = [Int(c) - CHAR_OFFSET + TIME_OFFSET for c in 'A':'Z']
    nodes, pred_dag, succ_dag = open(build_dag, filename)
    next_nodes = binary_minheap(Char)
    available_workers = Set(1:N_WORKERS)
    next_workers = PriorityQueue{Tuple{Int, Char, Int}, Int}()
    remaining_nodes = Set(nodes)
    completed_nodes = fill('0', length(remaining_nodes))
    schedule_nodes!(next_nodes, remaining_nodes, pred_dag)
    t = 0
    schedule_workers!(next_workers, next_nodes, available_workers, t, time_required)
    # display(pred_dag)
    # display(next_workers)
    # display(remaining_nodes)
    # display(completed_nodes)
    for i in axes(completed_nodes, 1)
        w, n, tf = dequeue!(next_workers)
        completed_nodes[i] = n
        push!(available_workers, w)
        t = tf
        clear_pred!(pred_dag, n)
        schedule_nodes!(next_nodes, remaining_nodes, pred_dag)
        schedule_workers!(next_workers, next_nodes, available_workers, t, time_required)
        # display(pred_dag)
        # display(next_workers)
        # display(remaining_nodes)
        # display(completed_nodes)
    end
    return string(completed_nodes...), t
end

function schedule_workers!(next_workers, next_nodes, available_workers, t, time_required)
    CHAR_OFFSET = Int('A') - 1
    while length(next_nodes) > 0 && length(available_workers) > 0
        n = pop!(next_nodes)
        w = pop!(available_workers)
        i = Int(n) - CHAR_OFFSET
        dt = time_required[i]
        enqueue!(next_workers, (w, n, t+dt)=>t+dt)
    end
end

function solve081(filename)
    CHILD = 1
    METADATA = 2
    HEADER1 = 3
    HEADER2 = 4
    inputs = readline(filename)
    tot::Int = 0
    next_mode = Stack{Int}()
    push!(next_mode, HEADER1)
    local mode::Int = 0
    n_child::Int = 0
    n_metadata::Int = 0
    for c in split(inputs)
        mode = pop!(next_mode)
        val = parse(Int, c)
        # println("Pre ", mode, " ", val, " ", next_mode)
        if mode == HEADER1
            # Reading number of children
            n_child = val
            push!(next_mode, HEADER2)
        elseif mode == HEADER2
            # Reading number of metadata
            n_metadata = val
            if n_metadata > 0
                push!(next_mode, repeat([METADATA], n_metadata)...)
            end
            if n_child > 0
                push!(next_mode, repeat([HEADER1], n_child)...)
            end
        elseif mode == METADATA
            tot += val
        end
        # println("Post ", tot, " ", next_mode)
    end
    return tot
end

mutable struct Node{T}
    val::T
    children::Array{T,1}
    metadata::Array{T,1}
end

function solve082(filename)
    CHILD = 1
    METADATA = 2
    HEADER1 = 3
    HEADER2 = 4
    inputs = readline(filename)
    tot::Int = 0
    tree = Node{Int}[]
    push!(tree, Node{Int}(-1, [], []))
    max_node_idx::Int = 1
    next_mode = Stack{Tuple{Int, Int, Int}}()
    push!(next_mode, (HEADER1, 0, max_node_idx))
    local mode::Int = 0
    n_child::Int = 0
    n_metadata::Int = 0
    for c in split(inputs)
        mode, parent, node_idx = pop!(next_mode)
        val = parse(Int, c)
        # println("Pre ", mode, " ", parent, " ", next_mode)
        # display(tree); println()
        if mode == HEADER1
            # Reading number of children
            n_child = val
            push!(next_mode, (HEADER2, parent, node_idx))
        elseif mode == HEADER2
            # Reading number of metadata
            n_metadata = val
            if n_metadata > 0
                push!(next_mode, repeat([(METADATA, node_idx, 0)], n_metadata)...)
            end
            if n_child > 0
                push!(next_mode, [(HEADER1, node_idx, max_node_idx + i) for i in n_child:-1:1]...)
                for i in 1:n_child
                    push!(tree, Node{Int}(-1, [], []))
                end
                max_node_idx += n_child
            end
            if parent > 0
                parent_node = tree[parent]
                push!(parent_node.children, node_idx)
            end
        elseif mode == METADATA
            parent_node = tree[parent]
            push!(parent_node.metadata, val)
        end
        # println("Post ", parent, " ", next_mode)
        # display(tree); println()
    end

    setval!(tree, tree[1])
    display(tree); println()
    return tree[1].val
end


function setval!(tree::Array{Node{T},1}, n::Node{T}) where T<:Integer
    if n.val >= 0
        return
    elseif length(n.children) == 0
        n.val = sum(n.metadata)
    else
        vals = similar(n.children)
        for tup in enumerate(n.children)
            idx, i = tup
            c = tree[i]
            if c.val < 0
                setval!(tree, c)
            end
            vals[idx] = c.val
        end
        n.val = 0
        for m in n.metadata
            if m <= length(n.children)
                n.val += vals[m]
            end
        end
    end
end
