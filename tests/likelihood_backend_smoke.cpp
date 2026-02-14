#include "dmercator/gpu/likelihood_backend.hpp"

int main()
{
  auto backend = dmercator::gpu::create_likelihood_backend();
  if(!backend)
  {
    return 1;
  }

  dmercator::gpu::CsrGraph empty_graph;
  empty_graph.nb_vertices = 0;
  const auto status = backend->initialize(empty_graph, true);
  if(status.available)
  {
    return 1;
  }
  return 0;
}
