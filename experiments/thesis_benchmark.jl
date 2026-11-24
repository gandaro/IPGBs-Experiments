using Distributed

const NUM_WORKERS = 2
if nprocs() == 1
    addprocs(NUM_WORKERS)
end

@everywhere begin
    using IPGBs

    # Define your baseline and custom optimization variants here:
    const VARIANTS = [
        ("Baseline (Binary Trunc)", path -> groebner_basis(path, quiet=false, use_binary_truncation=true)),
        #("Opt_V1 (My Custom Opt)",   path -> groebner_basis(path, quiet=false, use_binary_truncation=true)) # <-- adjust kwargs/function here
    ]

    function run_instance(path::String)
        for (label, solve_fn) in VARIANTS
            try
                println("IPGBs, ", label)
                GC.gc() # Prevent GC spillover from the previous variant
                stats = @timed solve_fn(path)
                println("time => ", stats.time)
                println("memory => ", stats.bytes / 1024^3)
            catch e
                println("Error in $label: ", e)
            end
        end
    end

    function process_file(filename::String)
        println("[Worker $(myid())] Running ", filename)
        open(filename * ".log", "w") do f
            redirect_stdout(f) do
                run_instance(filename)
            end
        end
    end
end

const EXCLUDE_PATTERNS = [
    "knapsack_binary_50_",
    "knapsack_binary_multidimensional_20_5_",
    "knapsack_binary_multidimensional_25_3_",
    "set_packing_100_35_",
    "set_cover_30_17_",
    "set_cover_30_18_",
    "set_cover_30_19_",
    "knapsack_multidimensional_90_",
    "knapsack_multidimensional_100_"
]

is_excluded(filename::String) = any(p -> occursin(p, basename(filename)), EXCLUDE_PATTERNS)

const INSTANCE_DIRS = [
    "knapsack_binary",
    "knapsack_unbounded",
    "knapsack_multidimensional",
    "set_packing",
    "set_cover",
    "knapsack_binary_multidimensional"
]

function run_scaling_suite()
    # 1. Clean old logs
    for dirname in INSTANCE_DIRS
        if isdir(dirname)
            for file in readdir(dirname, join=true)
                endswith(file, ".log") && rm(file; force=true)
            end
        end
    end

    # 2. Parallel JIT warmup for ALL variants on all 4 workers
    warmup_file = "knapsack_binary/knapsack_binary_10_1.mps"
    if isfile(warmup_file)
        pmap(workers()) do _
            redirect_stdout(devnull) do
                run_instance(warmup_file)
            end
        end
    end

    # 3. Collect representative scaling instances (_1.mps)
    selected_files = String[]
    for dirname in INSTANCE_DIRS
        if isdir(dirname)
            for filename in sort(readdir(dirname, join=true))
                if endswith(filename, "_1.mps") && !is_excluded(filename)
                    push!(selected_files, filename)
                end
            end
        end
    end

    # Sort largest files first for optimal 4-core load balancing
    sort!(selected_files, by=filesize, rev=true)

    t_start = time()
    println("=== Running scaling benchmark ($(length(selected_files)) instances x $(length(VARIANTS)) variants on $NUM_WORKERS workers) ===")
    pmap(process_file, selected_files)
    println("=== Done in $(round(time() - t_start, digits=2)) seconds! ===")
end

run_scaling_suite()
