# APSP with SIMD-X

# Custom modifications from upstream
- Reindented a bunch of stuff because VSCode _hates_ ifdefs that starts on first column inside indented blocks.
- `sssp_routing` built into APSP test based on original `sssp_high_diameter`.
- Atomic float comparision hack added.
- Added vert_data_out for SSSP routing table output.
- Removed _massive_ device memory leak from improper metadata lifecycle. Kernels can be launched more than once per program now.

# NOTE 
Due to time constrains, these items are left in a known-broken state:
- Only those code path used by `sssp_routing` were modified for proper operation.
Other codepaths are only duct-taped together to compile. `vert_data_out` and related additions WILL NOT WORK CORRECTLY ELSEWHERE.
- Graph traversal is done in reverse. (Each iteration compute SSSP _to_ selected node.) This is only fine in undirected graph.
- Routing is in `[to_target][from_start] = next_node` format for simpler interface with existing code.

Also, the `invert` flag in `text_to_bin` _actually_ means double up the edges.

# Run instruction (for APSP)

The relevant folders in `test/optimal` are as follows:
- `sssp_bellman_ford`: Bellman-Ford implementation
- `sssp_bellman_ford_unified`: Unified memory version of Bellman-Ford (untested)
- `sssp_routing`: APSP with SIMD-X

The other folders are the unmodified code from upstream.

To prep the system for running the code:
1. Compile the graph converter from https://github.com/asherliu/graph_project_start/. Use the `tuple_text_to_binary_csr_mem_weight` converter. Make sure to have the flag for reversed edge enabled to make the output graph undirected.
2. Download the graph data of interest from Stanford SNAP collection (http://snap.stanford.edu/data/index.html).
3. Using the executable compiled in 1, convert the graph to the binary representation. You will get 3 files.

To run the code:
1. Make sure the Makefile in the reledvant folder have the correct GPU arch in the `cuflags`
2. Run `make`
3. Run the executable with the flags as required. Run the executable without any flags to see the flag listing. The three input files mentioned are those generated in the system prep step.


-----------------------------------------------------------------------------------
# Original README kept below for reference

-----
Custom modifications from upstream
-----
- Reindented a bunch of stuff because VSCode _hates_ ifdefs that starts on first column inside indented blocks.
- `sssp_routing` built into APSP test based on original `sssp_high_diameter`.
- Atomic float comparision hack added.
- Added vert_data_out for SSSP routing table output.
- Removed _massive_ device memory leak from improper metadata lifecycle. Kernels can be launched more than once per program now.

**NOTE**: 
Due to time constrains, these items are left in a known-broken state:
- Only those code path used by `sssp_routing` were modified for proper operation.
Other codepaths are only duct-taped together to compile. `vert_data_out` and related additions WILL NOT WORK CORRECTLY ELSEWHERE.
- Graph traversal is done in reverse. (Each iteration compute SSSP _to_ selected node.) This is only fine in undirected graph.
- Routing is in `[to_target][from_start] = next_node` format for simpler interface with existing code.

-----
Software requirement
-----
gcc 4.4.7 or higher 

CUDA 7.5 or higher 

-----
Hardware
------
GPU: K20, K40, P100, V100 (tested)
> Generally, the GPU needs to support shuffle instructions.

-----
Compile
-----
nvcc 7.5 or higher

cd simd-x/test/optimal/``app``
-type ``make``

> For instance, one can enter ``simd-x/test/optimal/bfs_high_diameter/`` and type ``make``


-----
Execute
------

For each application, once you type the compiled executable, the binary file will remind you the files of interest. 

> Using ``bfs_high_diameter`` as an example, one will need ``/path/to/exe /path/to/beg_pos /path/to/adj_list /path/weight_list src blk_size swith_iter`` to execute the file. Below are the explanation of each parameter:
> - `path/to/exe`: the path to this executable.
> - `/path/to/beg_pos`: the path to the begin_position array of the graph dataset. We explained the begin position file [here](https://github.com/asherliu/graph_project_start/blob/master/README.md).
> Similarly for `/path/to/adj_list` and `/path/weight_list`. It is important to note that, for applications that do not need weight file (such as bfs), we can provide an invalid path to the `/path/weight_list` parameter.
> - `src` stands for where the users want the BFS starts.
> - `blk_size` means the number of thread blocks we want the kernel to have.  
> - `swith_iter` means which iteration to switch the BFS direction from top-down to bottom-up. 



**Should you have any questions about this project, please contact us by asher.hangliu@gmail.com.**

-----
Reference
-------
   [USENIX ATC '19] SIMD-X: Programming and Processing of Graph Algorithms on GPUs [[PDF](https://arxiv.org/pdf/1812.04070.pdf)]


