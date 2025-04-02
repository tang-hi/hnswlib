#include "catch2/catch_test_macros.hpp"
#include "catch2/benchmark/catch_benchmark.hpp"
#include "hnswlib/utils.h"

TEST_CASE("Orthogonal Matrix Test", "[orthogonal]") {
  int dim = 4;
  auto orthogonal = createOrthogonal(dim);
  auto identity = orthogonal.transpose() * orthogonal;
  CHECK(identity.isApprox(Eigen::MatrixXf::Identity(dim, dim), 1e-5));
}

TEST_CASE("Orthogonal Matrix Bench", "[benchhmark]") {
    BENCHMARK("createOrthogonal 8") {
        auto orthogonal = createOrthogonal(8);
    };

    BENCHMARK("createOrthogonal 16") {
        auto orthogonal = createOrthogonal(16);
    };

    BENCHMARK("createOrthogonal 128") {
        auto orthogonal = createOrthogonal(128);
    };

}