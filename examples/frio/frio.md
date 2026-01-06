# Notes about Rio Frio example

file: frio.csv
previously: q_river_nearest_face_RioFrio.csv 
source: Natasha Flores natasha.flores@deltares.nl on 17 Sept 2025

file: frio.toml

### watershed notes:
- Rio Frio, Colombia
- Point in montain catchement in tropical forest
- No observational data available

### migration requirement notes:
1. First migration (at onset of low flow season).
    - Upstream: 1 Jan-1 Mar (dowy: 306-365).
    - Downstream (return): 1 May-30 Jun (dowy: 61-122).
2. Second migration (at onset of high flow season).
    - Upstream: 1 Jul-30 Aug (dowy: 122-243).
    - Downstream (return): 1 Oct-1 Dec (dowy: 274-335).
3. Seperate fish species.

### notes:
1. Focus only on instream suitability for fish.
2. Should return to `1` apart from flow values required for fish passage, what else is required, for example:
    - what do the fishes eat, are there eco-hydrology needs for lower trophic levels?
    - what is the fishes habitat requirements, perhaps a sediment transport component is need for habitat maintenance?
 