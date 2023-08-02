#ifndef KUIPER_COURSE_INCLUDE_LAYER_SIGMOID_LAYER_HPP_
#define KUIPER_COURSE_INCLUDE_LAYER_SIGMOID_LAYER_HPP_
#include "layer.hpp"
#include "ops/sigmoid_op.hpp"

namespace kuiper_infer {
class SigmoidLayer : public Layer {
public:
  ~SigmoidLayer() override = default;

  explicit SigmoidLayer(const std::shared_ptr<Operator> &op);

  // 执行sigmoid操作的具体函数Forwards
  void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

  static std::shared_ptr<Layer>
  CreateInstance(const std::shared_ptr<Operator> &op);

private:
  std::unique_ptr<SigmoidOperator> op_;
};
} // namespace kuiper_infer
#endif // KUIPER_COURSE_INCLUDE_LAYER_SIGMOID_LAYER_HPP_