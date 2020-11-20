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

void bellman_ford_outbound_cpu(
    graph<long, long, long,vertex_t, index_t, weight_t> *ggraph, 
    feature_t *vert_status,
    data_out_cell_t *vert_data_out
){
    vertex_t vert_count = ggraph->vert_count;
    
    // vert_data_out is [to_final_node][from_at]
    printf("CPU data init started\n");
    for(feature_t fr = 0; fr < vert_count; fr++) {
        for(feature_t to = 0; to < vert_count; to++) {
            if(fr == to) {
                vert_status[to * vert_count + fr] = 0;
                vert_data_out[to * vert_count + fr] = -1;
            } else {
                vert_status[to * vert_count + fr] = INFTY;
                vert_data_out[to * vert_count + fr] = -1;
            }
        }
    }
    printf("CPU data init ok\n");
    int changed = true;
    int iter = 0;
    while(changed) {
        changed = false;
        for(vertex_t node_at = 0; node_at < vert_count; node_at++) {
            vertex_t my_beg = ggraph->beg_pos[node_at];
            vertex_t my_end = ggraph->beg_pos[node_at+1];
            for(;my_beg < my_end; my_beg ++) {
                vertex_t nebr = ggraph->adj_list[my_beg];
                weight_t weit = ggraph->weight[my_beg];
                // Now steal their routing table (pull)
                for(vertex_t row = 0; row < vert_count; row++) {
                    if(vert_status[row * vert_count + node_at] == INFTY){
                        continue;
                    }
                    feature_t my_dist = vert_status[row * vert_count + nebr];
                    feature_t new_dist =  vert_status[row * vert_count + node_at] + weit;
                    if(new_dist < my_dist) {
                        vert_status[row * vert_count + nebr] = new_dist;
                        vert_data_out[row * vert_count + nebr] = node_at;
                        changed = true;
                    } else if (new_dist == my_dist && node_at < vert_data_out[row * vert_count + nebr]){
                        vert_data_out[row * vert_count + nebr] = node_at;
                        changed = true;
                    }
                }
            }
        }
        printf("iter %d ok\n", iter);
        iter++;
    }
    printf("BF converged at %d\n", iter);
}

void bellman_ford_inbound_cpu(
    graph<long, long, long,vertex_t, index_t, weight_t> *ggraph, 
    feature_t *vert_status,
    data_out_cell_t *vert_data_out
){
    // Standard graph is outbound edges.
    // RUN THIS ON INVERTED GRAPH
    vertex_t vert_count = ggraph->vert_count;
    
    // vert_data_out is [to_final_node][from_at]
    printf("CPU data init started\n");
    for(feature_t fr = 0; fr < vert_count; fr++) {
        for(feature_t to = 0; to < vert_count; to++) {
            if(fr == to) {
                vert_status[to * vert_count + fr] = 0;
                vert_data_out[to * vert_count + fr] = -1;
            } else {
                vert_status[to * vert_count + fr] = INFTY;
                vert_data_out[to * vert_count + fr] = -1;
            }
        }
    }
    printf("CPU data init ok\n");
    int changed = true;
    int iter = 0;
    while(changed) {
        changed = false;
        for(vertex_t node_at = 0; node_at < vert_count; node_at++) {
            vertex_t my_beg = ggraph->beg_pos[node_at];
            vertex_t my_end = ggraph->beg_pos[node_at+1];
            for(;my_beg < my_end; my_beg ++) {
                vertex_t nebr = ggraph->adj_list[my_beg];
                weight_t weit = ggraph->weight[my_beg];
                // Now steal their routing table (pull)
                for(vertex_t row = 0; row < vert_count; row++) {
                    if(vert_status[row * vert_count + nebr] == INFTY){
                        continue;
                    }
                    feature_t my_dist = vert_status[row * vert_count + node_at];
                    feature_t new_dist =  vert_status[row * vert_count + nebr] + weit;
                    if(new_dist < my_dist) {
                        vert_status[row * vert_count + node_at] = new_dist;
                        vert_data_out[row * vert_count + node_at] = nebr;
                        changed = true;
                    } else if (new_dist == my_dist && nebr < vert_data_out[row * vert_count + node_at]){
                        vert_data_out[row * vert_count + node_at] = nebr;
                        changed = true;
                    }
                }
            }
        }
        printf("iter %d ok\n", iter);
        iter++;
    }
    printf("BF converged at %d\n", iter);
}

__global__ void bf_init_data_kernel(
    vertex_t vert_count, 
    feature_t *vert_status,
    data_out_cell_t *vert_data_out
){
	index_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid < vert_count) {
        for(feature_t to = 0; to < vert_count; to++) {
            if(gid == to) {
                vert_status[to * vert_count + gid] = 0;
                vert_data_out[to * vert_count + gid] = -1;
            } else {
                vert_status[to * vert_count + gid] = INFTY;
                vert_data_out[to * vert_count + gid] = -1;
            }
        }
    }
}

__global__ void bf_iterate_data_kernel(
    vertex_t vert_count,
    feature_t *vert_status,
    data_out_cell_t *vert_data_out,
    vertex_t *beg_pos,
    vertex_t *adj_list,
    feature_t *weight_list,
    bool* changed
){
    index_t node_at = threadIdx.x + blockIdx.x * blockDim.x;
    if(node_at == 0) {
        *changed = false;
    }
    __syncthreads();
    // bool* changed will only be written to with `true`, so race cond within iteration is ok
    if(node_at < vert_count) {
        vertex_t my_beg = beg_pos[node_at];
        vertex_t my_end = beg_pos[node_at+1];
        for(;my_beg < my_end; my_beg ++) {
            vertex_t nebr = adj_list[my_beg];
            weight_t weit = weight_list[my_beg];
            // Now steal their routing table (pull)
            for(vertex_t row = 0; row < vert_count; row++) {
                if(vert_status[row * vert_count + nebr] == INFTY){
                    continue;
                }
                feature_t my_dist = vert_status[row * vert_count + node_at];
                feature_t new_dist =  vert_status[row * vert_count + nebr] + weit;
                if(new_dist < my_dist) {
                    vert_status[row * vert_count + node_at] = new_dist;
                    vert_data_out[row * vert_count + node_at] = nebr;
                    *changed = true;
                } else if (new_dist == my_dist && nebr < vert_data_out[row * vert_count + node_at]){
                    vert_data_out[row * vert_count + node_at] = nebr;
                    *changed = true;
                }
            }
        }
    }
}

void bellman_ford_inbound_gpu(
    gpu_graph *ggraph,
    feature_t *vert_status,
    data_out_cell_t *vert_data_out,
    feature_t block_size
){
    // Standard graph is outbound edges.
    // RUN THIS ON INVERTED GRAPH
    vertex_t vert_count = ggraph->vert_count;
    
    // vert_data_out is [to_final_node][from_at]
    
    // do data init
    feature_t grid_size = (vert_count + block_size - 1)/block_size;
    bf_init_data_kernel<<<grid_size,block_size>>>(vert_count, vert_status, vert_data_out);
    bool* g_changed;
    H_ERR(cudaMalloc((void **)&g_changed, sizeof(bool)));
    H_ERR(cudaThreadSynchronize());
    printf("GPU data init ok\n");
    bool changed = true;
    int iter = 0;
    while(changed) {
        changed = false;
        bf_iterate_data_kernel<<<grid_size,block_size>>>(
            vert_count,
            vert_status,
            vert_data_out,
            ggraph->beg_pos,
            ggraph->adj_list,
            ggraph->weight_list,
            g_changed
        );
        H_ERR(cudaThreadSynchronize());
        H_ERR(cudaMemcpy(&changed, g_changed, sizeof(bool), cudaMemcpyDeviceToHost));
        printf("iter %d ok\n", iter);
        iter++;
    }
    printf("BF_gpu converged at %d\n", iter);
}

int main(int args, char **argv)
{
    // Based on the high-diameter SSSP
    std::cout<<"Input: /path/to/exe beg_inv csr_inv wt_inv gpu=1 block_size=?? show_debug=1\n";
    if(args<7){
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
    const int ENABLE_GPU = atoi(argv[4]);
    const int BLOCK_SIZE = atoi(argv[5]);
    const int ENABLE_DEBUG = atoi(argv[6]);
    
    //Read graph to CPU
    graph<long, long, long,vertex_t, index_t, weight_t>
        *ginst=new graph<long, long, long,vertex_t, index_t, weight_t>
        (file_beg_pos, file_adj_list, file_weight_list);

    double total_time = 0;


    feature_t *vert_status;
    data_out_cell_t *vert_data_out;
    vertex_t vert_count = ginst->vert_count;
    const vertex_t STATUS_SZ = sizeof(feature_t) * vert_count * vert_count;
    const vertex_t DATA_SZ = sizeof(data_out_cell_t) * vert_count * vert_count;
    vert_status = (feature_t*) malloc(STATUS_SZ);
    vert_data_out = (data_out_cell_t*) malloc(DATA_SZ);
    // vert_data_out is [to_final_node][from_at]
    gpu_graph ggraph(ginst);
    feature_t *g_vert_status;
    data_out_cell_t *g_vert_data_out;
    if(ENABLE_GPU) {
        H_ERR(cudaMalloc((void **)&g_vert_status, sizeof(feature_t) * vert_count * vert_count));
        H_ERR(cudaMalloc((void **)&g_vert_data_out, sizeof(data_out_cell_t) * vert_count * vert_count));
    }
    printf("Init ok\n");
    double time = wtime();
    //bellman_ford_outbound_cpu(ginst, vert_status, vert_data_out);
    if(!ENABLE_GPU){
        printf("CPU run started\n");
        bellman_ford_inbound_cpu(ginst, vert_status, vert_data_out);
    } else {
        printf("GPU run started\n");
        bellman_ford_inbound_gpu(&ggraph, g_vert_status, g_vert_data_out, BLOCK_SIZE);
    }

    time = wtime() - time;
    std::cout<<"Total APSP time: "<<time<<" second(s).\n";
    if(ENABLE_GPU) {
		H_ERR(cudaMemcpy(vert_status, g_vert_status, STATUS_SZ, cudaMemcpyDeviceToHost));
		H_ERR(cudaMemcpy(vert_data_out, g_vert_data_out, STATUS_SZ, cudaMemcpyDeviceToHost));
    }

    for(vertex_t src_v = 0; src_v < ginst->vert_count; src_v++) {
        if(ENABLE_DEBUG) {
            printf("\t\t--- At start node %d ---\n", src_v);
            feature_t *cpu_dist;
            data_out_cell_t *cpu_routes;
            cpu_sssp<index_t, vertex_t, weight_t, feature_t>
                (cpu_dist, cpu_routes, src_v, ginst->vert_count, ginst->edge_count, ginst->beg_pos,
                ginst->adj_list, ginst->weight);
            feature_t *gpu_dist = &(vert_status[src_v * vert_count]);
            data_out_cell_t *gpu_routes = &(vert_data_out[src_v * vert_count]);
            if (memcmp(cpu_dist, gpu_dist, sizeof(feature_t) * ginst->vert_count) == 0) {
                printf("Distance result correct\n");
                //Now check route
                if (memcmp(cpu_routes, gpu_routes, sizeof(data_out_cell_t) * ginst->vert_count) == 0) {
                    printf("Route result correct\n");
                }else{
                    printf("Route result wrong!\n");
                    //TODO: "deep inspect" route by traversing back to root and check weight
                    printf("GPU - CPU\n");
                    for(vertex_t i = 0; i < ginst->vert_count; i++){
                        if (gpu_routes[i] != cpu_routes[i]) {
                            printf("%d: (%d, %d) - (%d, %d): %d\n", i,
                                gpu_routes[i],
                                gpu_dist[gpu_routes[i]],
                                cpu_routes[i],
                                cpu_dist[cpu_routes[i]],
                                gpu_dist[gpu_routes[i]] - cpu_dist[cpu_routes[i]]
                            );
                            printf("\tG: ");
                            vertex_t current = i;
                            while(current != -1) {
                                printf("%d->", current);
                                current = gpu_routes[current];
                            }
                            printf("\n");
                            printf("\tC: ");
                            current = i;
                            while(current != -1) {
                                printf("%d->", current);
                                current = cpu_routes[current];
                            }
                            printf("\n");
                        }
                    }
                    break;
                }

            } else {
                printf("Distance result wrong!\n");
                for(int i = 0; i < 10; i ++) {
                    std::cout<<gpu_dist[i]<<" "<<cpu_dist[i]<<"\n";
                }
                break;
            }
        }
    }
    

}
