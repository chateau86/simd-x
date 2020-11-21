#ifndef HEADER_H
#define HEADER_H
typedef unsigned int index_t;
typedef unsigned int vertex_t;
typedef int feature_t;
typedef int weight_t;
typedef int data_out_cell_t;
typedef long long unsigned int ptr_t;
typedef struct {
    feature_t feature;
    data_out_cell_t data_out;
} data_return_t;
typedef unsigned char bit_t;

#define INFTY			(int)(1<<20)
#define BIN_SZ			4096	
#endif
