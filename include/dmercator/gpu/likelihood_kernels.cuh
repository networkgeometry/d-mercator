#ifndef DMERCATOR_GPU_LIKELIHOOD_KERNELS_CUH
#define DMERCATOR_GPU_LIKELIHOOD_KERNELS_CUH

#include <cuda_runtime.h>

namespace dmercator::gpu::kernels {

void launch_score_candidates_s1(const double *theta,
                                const double *kappa,
                                int nb_vertices,
                                const int *row_offsets,
                                const int *col_indices,
                                int v1,
                                double prefactor,
                                double beta,
                                const double *candidate_theta,
                                int nb_candidates,
                                double *out_scores,
                                cudaStream_t stream);

void launch_score_candidates_sd(const double *positions_soa,
                                int dim_plus_one,
                                const double *kappa,
                                int nb_vertices,
                                const int *row_offsets,
                                const int *col_indices,
                                int v1,
                                double radius,
                                double mu,
                                double beta,
                                double numerical_zero,
                                const double *candidate_positions_soa,
                                int nb_candidates,
                                double *out_scores,
                                cudaStream_t stream);

void launch_expected_degrees_s1(const double *theta,
                                const double *kappa,
                                int nb_vertices,
                                double beta,
                                double prefactor,
                                double *out_expected_degrees,
                                cudaStream_t stream);

void launch_expected_degrees_sd(const double *positions_soa,
                                int dim_plus_one,
                                const double *kappa,
                                int nb_vertices,
                                int dim,
                                double beta,
                                double mu,
                                double radius,
                                double numerical_zero,
                                double *out_expected_degrees,
                                cudaStream_t stream);

} // namespace dmercator::gpu::kernels

#endif // DMERCATOR_GPU_LIKELIHOOD_KERNELS_CUH
