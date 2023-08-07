#ifndef KUIPER_INFER_SOURCE_LAYER_PARAM_LAYER_HPP_
#define KUIPER_INFER_SOURCE_LAYER_PARAM_LAYER_HPP_
#include "layer.hpp"

namespace kuiper_infer {
class ParamLayer : public Layer {
public:
  explicit ParamLayer(const std::string &layer_name);
  /**
   * @brief 初始化权重参数
   * 
   * @param param_count 一个批次内权重的数目
   * @param param_channel 权重的通道数
   * @param param_height 权重的行数
   * @param param_width 权重的列数
   */
  void InitWeightParam(const uint32_t param_count, const uint32_t param_channel,
                       const uint32_t param_height, const uint32_t param_width);
  /**
   * @brief 初始化偏置参数
   *
   * @param param_count 一个批次内偏置的数目
   * @param param_channel 偏置的通道数
   * @param param_height 偏置的行数
   * @param param_width 偏置的列数
   */
  void InitBiasParam(const uint32_t param_count, const uint32_t param_channel,
                     const uint32_t param_height, const uint32_t param_width);

  const std::vector<std::shared_ptr<Tensor<float>>> &weights() const override;

  const std::vector<std::shared_ptr<Tensor<float>>> &bias() const override;

  void set_weights(const std::vector<float> &weights) override;

  void set_bias(const std::vector<float> &bias) override;

  void set_weights(
      const std::vector<std::shared_ptr<Tensor<float>>> &weights) override;

  void
  set_bias(const std::vector<std::shared_ptr<Tensor<float>>> &bias) override;

protected:
  std::vector<std::shared_ptr<Tensor<float>>> weights_;
  std::vector<std::shared_ptr<Tensor<float>>> bias_;
};

} // namespace kuiper_infer

#endif // KUIPER_INFER_SOURCE_LAYER_PARAM_LAYER_HPP_
