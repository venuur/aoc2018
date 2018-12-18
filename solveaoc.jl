include("solvers.jl")

function main()
    solvers = [
        solve011 solve012;
        solve021 solve022;
        solve031 solve032;
        solve041 solve042;
        solve051 solve052;
        solve061 solve062;
        solve071 solve072;
        solve081 solve082;
        solve091 solve092;
    ]

    I_EXTRA_ARG = 3
    if length(ARGS) < I_EXTRA_ARG
        println("ARGS[1]    Day 1-25.")
        println("ARGS[2]    Part 1 or 2.")
        println("ARGS[$I_EXTRA_ARG:]   Args passed to solver, meaning is solver dependent.")
        println("")
        exit()
    end

    day = parse(Int, ARGS[1])
    part = parse(Int, ARGS[2])

    if !(firstindex(solvers) <= day <= lastindex(solvers))
        println("Day $day not implemented yet.")
        exit()
    elseif !(1 <= part <= 2)
        println("Part must be 1 or 2.")
        exit()
    end

    println("ARGS: ", ARGS)
    solve = solvers[day, part]

    solver_args = length(ARGS) >= I_EXTRA_ARG ? ARGS[I_EXTRA_ARG:end] : ()
    println("Solver Args: ", solver_args)

    result = solve(solver_args...)
    result === nothing ? println("Solve found nothing.") : println(result)
end

main()
