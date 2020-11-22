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
#include <pthread.h>

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
			mdata.vert_status[tid] = (((feature_t)1)<<32) - 1;
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
	feature_t MASK = (((feature_t)1)<<32) - 1;
	//printf("Mask: %lld\n", MASK);
	for(vertex_t i = 0; i < count; i++) {
		feature_t dist = (packed_cpu[i] >> 32);
		unpacked_dist[i] = dist;
		if (dist == SMOL_INFTY || dist == 0) {
			unpacked_route[i] = -1;
		} else {
			unpacked_route[i] = packed_cpu[i] & MASK;
		}
	}
	//printf("Unpack ok\n");
}

typedef struct {
	int thread_id;
	int thread_total;
	vertex_t vert_count;
	int blk_size;
	int DEBUG;
	double* thread_gpu_time;

	graph<long, long, long,vertex_t, index_t, weight_t>
		*ginst;
	gpu_graph ggraph;

	cb_reducer* vert_selector_push_ptr;
	cb_reducer* vert_selector_pull_ptr;
	cb_mapper* vert_behave_push_ptr;
	cb_mapper* vert_behave_pull_ptr;

	vertex_t* all_routes;
} thread_info;

void* launch_kernel(void* thread_arg){
	thread_info* t_info = (thread_info*) thread_arg;

	meta_data mdata(t_info->vert_count, t_info->ginst->edge_count);
	mapper compute_mapper(t_info->ggraph, mdata, *(t_info->vert_behave_push_ptr), *(t_info->vert_behave_pull_ptr));
	reducer worklist_gather(t_info->ggraph, mdata, *(t_info->vert_selector_push_ptr), *(t_info->vert_selector_pull_ptr));


	feature_t *level, *level_h;
	cudaMalloc((void **)&level, sizeof(feature_t));
	cudaMallocHost((void **)&level_h, sizeof(feature_t));
	cudaMemset(level, 0, sizeof(feature_t));

	Barrier global_barrier(BLKS_NUM);
	
	double thread_total_gpu_time = 0;
	for(vertex_t st = t_info->thread_id; st < t_info->vert_count; st+= t_info->thread_total) {
		cudaMemset(level, 0, sizeof(feature_t));
		//Init three data structures
		//printf("---at node %d/%d---\n", st, t_info->vert_count);
		double time = wtime();
		init<<<256,256>>>(st, t_info->vert_count, mdata);
		H_ERR(cudaThreadSynchronize());
		//printf("Launching kernel\n");
		//* necessary for high diameter graph, e.g., euro.osm and roadnet.ca
		mapper_merge_push(t_info->blk_size, level, t_info->ggraph, mdata, compute_mapper, worklist_gather, global_barrier);
		H_ERR(cudaThreadSynchronize());
		//printf("Kernel ok\n");
		time = wtime() - time;
		//std::cout<<"Node time: "<<time<<" second(s).\n";
		thread_total_gpu_time += time;
		
		cudaMemcpy(level_h, level, sizeof(feature_t), cudaMemcpyDeviceToHost);	
		//std::cout<<"Total iteration: "<<level_h[0]<<"\n";
		
		feature_t *packed_gpu_dist = new feature_t[t_info->vert_count];
		H_ERR(cudaMemcpy(packed_gpu_dist, mdata.vert_status, 
				sizeof(feature_t) * t_info->vert_count, cudaMemcpyDeviceToHost));

		feature_t *unpacked_gpu_dist = new feature_t[t_info->vert_count];
		vertex_t *unpacked_gpu_route = new vertex_t[t_info->vert_count];
		unpack_cpu_dist(packed_gpu_dist, unpacked_gpu_dist, unpacked_gpu_route, t_info->vert_count);

		vertex_t* route_out_ptr = t_info->all_routes + ((index_t)st * t_info->vert_count);
		//printf("Result offset: %llx\n", route_out_ptr);
		memcpy(route_out_ptr, unpacked_gpu_route, sizeof(vertex_t) * t_info->vert_count);
		//printf("Route saved\n");
		if(t_info->DEBUG) {
			feature_t *cpu_dist;
			vertex_t *cpu_routes;
			cpu_sssp<index_t, vertex_t, weight_t, feature_t> (
				cpu_dist, 
				cpu_routes, 
				st, 
				t_info->vert_count, 
				t_info->ginst->edge_count, 
				t_info->ginst->beg_pos,
				t_info->ginst->adj_list, 
				t_info->ginst->weight
			);

			if (memcmp(cpu_dist, unpacked_gpu_dist, sizeof(feature_t) * t_info->vert_count) == 0) {
				printf(" Distance result correct\n");
				if (memcmp(cpu_routes, route_out_ptr, sizeof(vertex_t) * t_info->vert_count) == 0) {
					printf(" Route result correct (quick)\n");
				} else {
					int num_wrong = 0;
					printf(" Route result check\n");
					printf("GPU - CPU\n");
					for(int i = 0; i < t_info->vert_count; i ++) {
						if(route_out_ptr[i] != cpu_routes[i]) {
							if (route_out_ptr[i] > 1 && cpu_routes[i] > 1) {
								printf("%d: %d %d - %d %llx\n", i, cpu_dist[i], route_out_ptr[i], cpu_routes[i], packed_gpu_dist[i]);
								num_wrong ++;
							}
						}
					}
					if (num_wrong > 0) {
						break;
					}
				}
			} else {
				printf(" Distance result wrong!\n");
				printf(" GPU - CPU\n");
				for(int i = 0; i < t_info->vert_count; i ++) {
					if(unpacked_gpu_dist[i] != cpu_dist[i]) {
						printf("%d: %d - %d\n", i, unpacked_gpu_dist[i], cpu_dist[i]);
					}
				}
				//break;
			}

			delete[] cpu_dist;
		}
		delete[] packed_gpu_dist;
		delete[] unpacked_gpu_dist;
		delete[] unpacked_gpu_route;
	}
	mdata.free_md();
	t_info->thread_gpu_time[t_info->thread_id] = thread_total_gpu_time;
	pthread_exit(NULL);
}

int main(int args, char **argv)
{
    // Based on the high-diameter SSSP
	std::cout<<"Input: /path/to/exe /path/to/beg_pos /path/to/adj_list /path/weight_list src blk_size launcher_threads debug=1\n";
	if(args<8){
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
    int launcher_threads = atoi(argv[6]);  // Not used
    int DEBUG = atoi(argv[7]);
	
	//Read graph to CPU
	graph<long, long, long,vertex_t, index_t, weight_t>
		*ginst=new graph<long, long, long,vertex_t, index_t, weight_t>
		(file_beg_pos, file_adj_list, file_weight_list);
	
	double walltime = wtime();

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
	
	vertex_t* all_routes;
	all_routes = (vertex_t*) malloc(sizeof(vertex_t) * ginst->vert_count * ginst->vert_count);
	double total_time = 0;
	

	gpu_graph ggraph(ginst);
	H_ERR(cudaThreadSynchronize());

	assert(launcher_threads > 0 && "Needs 1 or more threads");
	pthread_t threads[launcher_threads];
	double* thread_gpu_time = (double*) malloc(sizeof(double) * launcher_threads);
	thread_info* thread_param = (thread_info*) malloc(sizeof(thread_info) * launcher_threads);

	for(int thread_id = 0; thread_id < launcher_threads; thread_id++) {
		thread_param[thread_id].thread_id = thread_id;
		thread_param[thread_id].thread_total = launcher_threads;

		thread_param[thread_id].vert_count = ginst->vert_count;
		thread_param[thread_id].blk_size = blk_size;
		thread_param[thread_id].DEBUG = DEBUG;
		thread_param[thread_id].thread_gpu_time = thread_gpu_time;
	
		thread_param[thread_id].ginst = ginst;
		thread_param[thread_id].ggraph = ggraph;

		thread_param[thread_id].vert_selector_push_ptr = &vert_selector_push_h;
		thread_param[thread_id].vert_selector_pull_ptr = &vert_selector_pull_h;
		thread_param[thread_id].vert_behave_push_ptr = &vert_behave_push_h;
		thread_param[thread_id].vert_behave_pull_ptr = &vert_behave_pull_h;
	
		thread_param[thread_id].all_routes = all_routes;

		int rc = pthread_create(
				&threads[thread_id], 
				NULL, 
				launch_kernel, 
				(void *) &thread_param[thread_id]
			);
		if (rc) {
			printf("ERROR; return code from pthread_create() on thread %d is %d\n", thread_id, rc);
			exit(-1);
		}
	}
	//printf("Waiting for workers\n");
	for(int thread_id = 0; thread_id < launcher_threads; thread_id++) {
		pthread_join(threads[thread_id], NULL);
		total_time += thread_gpu_time[thread_id];
	}
	//printf("All worker done\n");
	
	walltime = wtime()-walltime;
	printf("Algo, In_file, threads, blk_size, gpu_t, wall_t\n");
	printf("### %s, %s, %d, %d, %.06f, %.06f,\n",
		"sssp",
		file_beg_pos,
		launcher_threads,
		blk_size,
		total_time,
		walltime
	);
	//std::cout<<"Total GPU time: "<<total_time<<" second(s).\n";
	//std::cout<<"Total wall time: "<<walltime<<" second(s).\n";
}
