#pragma once
#include "hnswlib/hnswlib.h"
#include "hnswlib/space_l2.h"
#include "Eigen/Dense"
#include "hnswlib/utils.h"
#include <cstddef>

namespace hnswlib {

static float L2Sqr(const void *vec1, const void *vec2, const void *dim);

class AdSampling : public SpaceInterface<float> {
private:
  DISTFUNC<float> fstdistfunc_;
  size_t data_size_;
  size_t dim_;
  Eigen::MatrixXf orthogonal_matrix_;

  constexpr static const int batch_size = 8;

  constexpr static const double eps0 = 2.1;

public:
  AdSampling(size_t dim) {
    fstdistfunc_ = L2Sqr;
    dim_ = dim;
    data_size_ = dim * sizeof(float);
    orthogonal_matrix_ = createOrthogonal(dim);
  }

  float L2Distance(const void *vec1, const void *vec2) {
    return L2Sqr(vec1, vec2, &dim_);
  }

  float L2Distance(float UpperBound, const void *vec1, const void *vec2) {
    float estimate = 0.0f;

    for (size_t i = 0; i < dim_ / batch_size; i++) {
      float *a = (float *)vec1 + i * batch_size;
      float *b = (float *)vec2 + i * batch_size;

      __m256 diff, v1, v2;
      __m256 sum = _mm256_set1_ps(0);

      v1 = _mm256_loadu_ps(a);
      v2 = _mm256_loadu_ps(b);
      diff = _mm256_sub_ps(v1, v2);
      sum = _mm256_fmadd_ps(diff, diff, sum);

      estimate += sum[0] + sum[1] + sum[2] + sum[3];
      double ratio = (1.0 * i / dim_) * (1.0 + eps0 / std::sqrt(i)) *
                     (1.0 + eps0 / std::sqrt(i));
      if (estimate > UpperBound * ratio) {
        return estimate * dim_ / i;
      }

      std::advance(a, batch_size);
      std::advance(b, batch_size);
    }
    return estimate;
  }

  DISTFUNC<float> get_dist_func() { return fstdistfunc_; }

  void *get_dist_func_param() { return &dim_; }
};

}; // namespace hnswlib
