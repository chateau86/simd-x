#include "header.h"
#include "util.h"
#include "mapper.cuh"
#include "reducer.cuh"
#include "wtime.h"
#include "barrier.cuh"
#include "gpu_graph.cuh"
#include "meta_data.cuh"
#include "mapper_enactor.cuh"
#include "reducer_enactor.cuh"
#include "cpu_sssp_route.hpp"

/*user defined vertex behavior function*/
__inline__ __host__ __device__ feature_t user_mapper_push(
    vertex_t 	src,
    vertex_t	dest,
    feature_t	level,
    index_t*	beg_pos,
    weight_t	edge_weight,
    feature_t* vert_status,
    feature_t* vert_status_prev
){
	feature_t dist = (vert_status[src]>>32) + edge_weight;
	return (dist<<32) + src;
	//return dist;
} 

/*user defined vertex behavior function*/
__inline__ __host__ __device__ bool vertex_selector_push(
    vertex_t 		vert_id, 
    feature_t 		level,
    vertex_t 		*adj_list, 
    index_t 		*beg_pos, 
    feature_t* 	vert_status,
    feature_t* 	vert_status_prev
){
	return (vert_status[vert_id] != vert_status_prev[vert_id]);
}

/*user defined vertex behavior function*/
__inline__ __host__ __device__ feature_t user_mapper_pull(
    vertex_t 	src,
    vertex_t	dest,
    feature_t	level,
    index_t*	beg_pos,
    weight_t	edge_weight,
    feature_t* vert_status,
    feature_t* vert_status_prev
){
	//return vert_status[src] + edge_weight;
	feature_t dist = (vert_status[src]>>32) + edge_weight;
	return (dist<<32) + src;
	//return dist;
}

/*user defined vertex behavior function*/
__inline__ __host__ __device__ bool vertex_selector_pull(
    vertex_t 	vert_id, 
    feature_t 	level,
    vertex_t* 	adj_list, 
    index_t* 	beg_pos, 
    feature_t* vert_status,
    feature_t* vert_status_prev
){
	return true;
}

__device__ cb_reducer vert_selector_push_d = vertex_selector_push;
__device__ cb_reducer vert_selector_pull_d = vertex_selector_pull;
__device__ cb_mapper vert_behave_push_d = user_mapper_push;
__device__ cb_mapper vert_behave_pull_d = user_mapper_pull;


/*init sssp*/
__global__ void
init(vertex_t src_v, vertex_t vert_count, meta_data mdata)
{
	index_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while(tid < vert_count) {
		if(tid != src_v) {
			mdata.vert_status[tid] = INFTY;
			mdata.vert_status_prev[tid] = INFTY;
		} else {
			mdata.vert_status[tid] = ((1<<31) - 1);
			//mdata.vert_status[tid] = 1;
			mdata.vert_status_prev[tid] = INFTY;
			
			mdata.worklist_mid[0] = src_v;
			mdata.worklist_sz_sml[0] = 0;
			mdata.worklist_sz_mid[0] = 1;
			mdata.worklist_sz_lrg[0] = 0;
			//mdata.bitmap[src_v>>3] |= (1<<(src_v & 7));
		}
		tid += blockDim.x * gridDim.x;
	}
}

void unpack_cpu_dist(
	feature_t* packed_cpu,
	feature_t* unpacked_dist,
	vertex_t* unpacked_route,
	vertex_t count
){
	for(vertex_t i = 0; i < count; i++) {
		unpacked_dist[i] = (packed_cpu[i] >> 32);
		unpacked_route[i] = (packed_cpu[i] & ((1<<32) - 1));
	}
}

int main(int args, char **argv)
{
    // Based on the high-diameter SSSP
	std::cout<<"Input: /path/to/exe /path/to/beg_pos /path/to/adj_list /path/weight_list src blk_size switch_iter\n";
	if(args<5){
        std::cout<<"Wrong input\n";exit(-1);
    }
    for(int i = 0; i < args; i++) {
		std::cout<<argv[i]<<" ";
	}
    std::cout<<"\n";
		
	double tm_map,tm_red,tm_scan;
	char *file_beg_pos = argv[1];
	char *file_adj_list = argv[2];
	char *file_weight_list = argv[3];
	vertex_t src_v = (vertex_t)atol(argv[4]);
    int blk_size = atoi(argv[5]);
    int switch_iter = atoi(argv[6]);
	
	//Read graph to CPU
	graph<long, long, long,vertex_t, index_t, weight_t>
		*ginst=new graph<long, long, long,vertex_t, index_t, weight_t>
		(file_beg_pos, file_adj_list, file_weight_list);
	
	feature_t *level, *level_h;
	cudaMalloc((void **)&level, sizeof(feature_t));
	cudaMallocHost((void **)&level_h, sizeof(feature_t));
    cudaMemset(level, 0, sizeof(feature_t));

	cb_reducer vert_selector_push_h;
	cb_reducer vert_selector_pull_h;
	cudaMemcpyFromSymbol(&vert_selector_push_h,vert_selector_push_d,sizeof(cb_reducer));
	cudaMemcpyFromSymbol(&vert_selector_pull_h,vert_selector_pull_d,sizeof(cb_reducer));
	
	cb_mapper vert_behave_push_h;
	cb_mapper vert_behave_pull_h;
	cudaMemcpyFromSymbol(&vert_behave_push_h,vert_behave_push_d,sizeof(cb_reducer));
	cudaMemcpyFromSymbol(&vert_behave_pull_h,vert_behave_pull_d,sizeof(cb_reducer));
	
	for(vertex_t st = 0; st < ginst->vert_count; st++) {
		//Init three data structures
		printf("---at node %d---\n", st);
		gpu_graph ggraph(ginst);
		meta_data mdata(ginst->vert_count, ginst->edge_count);
		Barrier global_barrier(BLKS_NUM);
		
		init<<<256,256>>>(st, ginst->vert_count, mdata);
		mapper compute_mapper(ggraph, mdata, vert_behave_push_h, vert_behave_pull_h);
		reducer worklist_gather(ggraph, mdata, vert_selector_push_h, vert_selector_pull_h);
		H_ERR(cudaThreadSynchronize());
		
		double time = wtime();

		//* necessary for high diameter graph, e.g., euro.osm and roadnet.ca
		mapper_merge_push(blk_size, level, ggraph, mdata, compute_mapper, worklist_gather, global_barrier);
		H_ERR(cudaThreadSynchronize());
		
		time = wtime() - time;
		std::cout<<"Total time: "<<time<<" second(s).\n";
		
		cudaMemcpy(level_h, level, sizeof(feature_t), cudaMemcpyDeviceToHost);	
		std::cout<<"Total iteration: "<<level_h[0]<<"\n";
		
		feature_t *packed_gpu_dist = new feature_t[ginst->vert_count];
		H_ERR(cudaMemcpy(packed_gpu_dist, mdata.vert_status, 
				sizeof(feature_t) * ginst->vert_count, cudaMemcpyDeviceToHost));

		feature_t *unpacked_gpu_dist = new feature_t[ginst->vert_count];
		vertex_t *unpacked_gpu_route = new vertex_t[ginst->vert_count];
		unpack_cpu_dist(packed_gpu_dist, unpacked_gpu_dist, unpacked_gpu_route, ginst->vert_count);

		feature_t *cpu_dist;
		vertex_t *cpu_routes;
		cpu_sssp<index_t, vertex_t, weight_t, feature_t>
			(cpu_dist, cpu_routes, st, ginst->vert_count, ginst->edge_count, ginst->beg_pos,
			ginst->adj_list, ginst->weight);

		if (memcmp(cpu_dist, unpacked_gpu_dist, sizeof(feature_t) * ginst->vert_count) == 0) {
			printf("Distance result correct\n");
		} else {
			printf("Distance result wrong!\n");
			printf("GPU - CPU\n");
			for(int i = 0; i < ginst->vert_count; i ++) {
				if(unpacked_gpu_dist[i] != cpu_dist[i]) {
					printf("%d: %d - %d\n", i, unpacked_gpu_dist[i], cpu_dist[i]);
				}
			}
			break;
		}
		delete[]  packed_gpu_dist;
		delete[]  unpacked_gpu_dist;
		delete[]  unpacked_gpu_route;
		delete[] cpu_dist;
		mdata.free_md();
	}
}
