
using DataStructures

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
