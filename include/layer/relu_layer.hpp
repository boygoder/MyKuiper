#ifndef KUIPER_COURSE_INCLUDE_LAYER_RELU_LAYER_HPP_
#define KUIPER_COURSE_INCLUDE_LAYER_RELU_LAYER_HPP_
#include "layer.hpp"
#include "ops/relu_op.hpp"

namespace kuiper_infer {
class ReluLayer : public Layer {
public:
  ~ReluLayer() override = default;

  // 通过这里，把relu_op中的thresh告知给relu layer, 因为计算的时候要用到
  explicit ReluLayer(const std::shared_ptr<Operator> &op);

  // 执行relu 操作的具体函数Forwards
  void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

  // 下节的内容，不用管
  static std::shared_ptr<Layer>
  CreateInstance(const std::shared_ptr<Operator> &op);

private:
  std::unique_ptr<ReluOperator> op_;
};
} // namespace kuiper_infer
#endif // KUIPER_COURSE_INCLUDE_LAYER_RELU_LAYER_HPP_