#ifndef KUIPER_COURSE_INCLUDE_LAYER_MAXPOOLING_LAYER_HPP_
#define KUIPER_COURSE_INCLUDE_LAYER_MAXPOOLING_LAYER_HPP_
#include "layer.hpp"
#include "ops/maxpooling_op.hpp"

namespace kuiper_infer {
class MaxPoolingLayer : public Layer {
public:
  ~MaxPoolingLayer() override = default;
  explicit MaxPoolingLayer(const std::shared_ptr<Operator>& op);
  void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;
  static std::shared_ptr<Layer>
  CreateInstance(const std::shared_ptr<Operator> &op);

private:
  std::unique_ptr<MaxPoolingOperator> op_;
};
} // namespace kuiper_infer
#endif // KUIPER_COURSE_INCLUDE_LAYER_MAXPOOLING_LAYER_HPP_