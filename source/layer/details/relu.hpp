#ifndef KUIPER_INFER_SOURCE_LAYER_BINOCULAR_RELU_HPP_
#define KUIPER_INFER_SOURCE_LAYER_BINOCULAR_RELU_HPP_
#include "layer/abstract/layer.hpp"
namespace kuiper_infer {
/**
 * @brief Relu层，当x>0时，返回x;否则返回0。
 *
 */
class ReluLayer : public Layer {
public:
  ReluLayer() : Layer("Relu") {}
  InferStatus
  Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
          std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

  static ParseParameterAttrStatus
  GetInstance(const std::shared_ptr<RuntimeOperator> &op,
              std::shared_ptr<Layer> &relu_layer);
};
} // namespace kuiper_infer
#endif // KUIPER_INFER_SOURCE_LAYER_BINOCULAR_RELU_HPP_
