#ifndef DMERCATOR_GPU_KERNELS_CUH
#define DMERCATOR_GPU_KERNELS_CUH

#include <cuda_runtime.h>

namespace dmercator::gpu::kernels {

__global__ void evaluate_refine_s1_candidates_kernel(const double *theta,
                                                     const double *pair_prefactor,
                                                     const int *neighbors,
                                                     int neighbor_count,
                                                     const double *candidate_angles,
                                                     int nb_candidates,
                                                     int v1,
                                                     int nb_vertices,
                                                     double beta,
                                                     double *out_scores);

__global__ void evaluate_refine_sd_candidates_kernel(const double *positions,
                                                     int position_stride,
                                                     const double *pair_prefactor,
                                                     const int *neighbors,
                                                     int neighbor_count,
                                                     const double *candidate_positions,
                                                     int nb_candidates,
                                                     int v1,
                                                     int nb_vertices,
                                                     double beta,
                                                     double numerical_zero,
                                                     double *out_scores);

__global__ void inferred_expected_degrees_s1_kernel(const double *theta,
                                                    const double *kappa,
                                                    int nb_vertices,
                                                    double beta,
                                                    double prefactor,
                                                    double *out_expected_degrees);

__global__ void inferred_expected_degrees_sd_kernel(const double *positions,
                                                    int position_stride,
                                                    const double *kappa,
                                                    int nb_vertices,
                                                    int dim,
                                                    double beta,
                                                    double mu,
                                                    double radius,
                                                    double numerical_zero,
                                                    double *out_expected_degrees);

} // namespace dmercator::gpu::kernels

#endif // DMERCATOR_GPU_KERNELS_CUH
