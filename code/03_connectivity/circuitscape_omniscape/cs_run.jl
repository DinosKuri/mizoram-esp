# Circuitscape pairwise (cost-adjacent source pairs) and Omniscape for one guild: julia cs_run.jl <guild> <radius_cells> <block_size> [tag]
using Circuitscape, Omniscape
g = ARGS[1]; rad = ARGS[2]; blk = ARGS[3]; tag = length(ARGS) > 3 ? ARGS[4] : g
mkpath("cs/out")
ini = """
[Circuitscape Mode]
data_type = raster
scenario = pairwise
[Habitat raster or graph]
habitat_file = cs/R_$(tag).asc
habitat_map_is_resistances = True
[Options for pairwise and one-to-all and all-to-one modes]
point_file = cs/src_$(tag).asc
use_included_pairs = True
included_pairs_file = cs/pairs_$(tag).txt
[Connection Scheme for raster habitat data]
connect_four_neighbors_only = False
connect_using_avg_resistances = False
[Calculation options]
solver = cholmod
[Output options]
output_file = cs/out/cs_$(tag).out
write_cur_maps = True
write_cum_cur_map_only = True
log_transform_maps = False
"""
write("cs/cs_$(tag).ini", ini)
if !("--omni-only" in ARGS)
    @time compute("cs/cs_$(tag).ini")
end
if !("--cs-only" in ARGS)
    oini = """
[Required]
resistance_file = cs/R_$(tag).asc
radius = $(rad)
block_size = $(blk)
project_name = cs/out/omni_$(tag)
[General options]
source_file = cs/omsrc_$(tag).asc
calc_normalized_current = true
calc_flow_potential = true
parallelize = true
parallel_batch_size = 10
solver = $(get(ENV,"OMNI_SOLVER","cg+amg"))
write_raw_currmap = true
mask_nodata = true
write_as_tif = false
"""
    write("cs/omni_$(tag).ini", oini)
    @time run_omniscape("cs/omni_$(tag).ini")
end
