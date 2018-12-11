include("solvers.jl")

function main()
    solvers = [
        solve011 solve012;
        solve021 solve022;
        solve031 solve032;
        solve041 solve042;
        solve051 solve052
    ]

    if length(ARGS) != 3
        println("ARGS[1]  Day 1-25.")
        println("ARGS[2]  Part 1 or 2.")
        println("ARGS[3]  Name of input file to solve.")
        println("")
        exit()
    end

    day, part, filename = ARGS
    day = parse(Int, day)
    part = parse(Int, part)
    if !(firstindex(solvers) <= day <= lastindex(solvers))
        println("Day $day not implemented yet.")
        exit()
    elseif !(1 <= part <= 2)
        println("Part must be 1 or 2.")
        exit()
    end

    solve = solvers[day, part]
    result = solve(filename)
    if result === nothing
        println("Solve found nothing.")
    else
        println(result)
    end
end

main()
