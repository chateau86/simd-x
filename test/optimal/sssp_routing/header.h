#ifndef HEADER_H
#define HEADER_H
typedef long long unsigned int index_t;
typedef int vertex_t;
//typedef int feature_t;
typedef long long int feature_t;
// Upper half: Distance
// Lower half: Next ptr
typedef int weight_t;
typedef int data_out_cell_t;
typedef struct {
    feature_t feature;
    data_out_cell_t data_out;
} data_return_t;
typedef unsigned char bit_t;

#define INFTY			(((feature_t)1)<<60)
#define SMOL_INFTY			(((feature_t)1)<<28)
#define BIN_SZ			4096	
#endif
