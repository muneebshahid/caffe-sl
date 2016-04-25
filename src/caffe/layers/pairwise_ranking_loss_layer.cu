#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/pair_wise_ranking_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void PairwiseRankingLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* pos_sim = bottom[0]->gpu_data();
  const Dtype* neg_sim = bottom[1]->gpu_data();
  Dtype* per_triplet_loss = bottom[0]->mutable_gpu_diff();
  int count = bottom[0]->count();

  Dtype* loss = top[0]->mutable_cpu_data();
  loss[0] = 0;
  for (int i=0; i<count; ++i) {
    per_triplet_loss[i] = std::max(Dtype(0),
        this->layer_param_.triplet_loss_param().margin()
        - pos_sim[i] + neg_sim[i]);
    loss[0] += per_triplet_loss[i];
  }
  loss[0] /= count;
}

template <typename Dtype>
void PairwiseRankingLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    bool sample = this->layer_param_.triplet_loss_param().sample();
    Dtype* pos_diff = bottom[0]->mutable_gpu_diff();
    Dtype* neg_diff = bottom[1]->mutable_gpu_diff();
    const Dtype* pos_sim = bottom[0]->gpu_data();
    const Dtype* neg_sim = bottom[1]->gpu_data();
    int count = bottom[0]->count();
    for (int i=0; i<count; ++i) {
      if (pos_diff[i] && (!sample || pos_sim[i] > neg_sim[i])) {
        pos_diff[i] = -1;
        neg_diff[i] = 1;
      } else {
        pos_diff[i] = 0;
        neg_diff[i] = 0;
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PairwiseRankingLossLayer);
}  // namespace caffe
