#include "layer/maxpooling_layer.hpp"
#include "factory/layer_factory.hpp"
#include "ops/maxpooling_op.hpp"
#include <glog/logging.h>
namespace kuiper_infer {
MaxPoolingLayer::MaxPoolingLayer(const std::shared_ptr<Operator> &op)
    : Layer("MaxPooling") {
  CHECK(op->op_type_ == OpType::kOperatorMaxPooling)
      << "Operator has a wrong type: " << int(op->op_type_);
  MaxPoolingOperator *maxpooling_op =
      dynamic_cast<MaxPoolingOperator *>(op.get());
  //检查op中的裸指针是否是对应的类型
  CHECK(maxpooling_op != nullptr) << "MaxPooling operator is empty";
  // uint32_t pooling_h = maxpooling_op->pooling_height();
  // uint32_t pooling_w = maxpooling_op->padding_width();
  // uint32_t stride_h = maxpooling_op->stride_height();
  // uint32_t stride_w = maxpooling_op->stride_width();
  // uint32_t padding_h = maxpooling_op->padding_height();
  // uint32_t padding_w = maxpooling_op->padding_width();
  // this->op_ = std::make_unique<MaxPoolingOperator>(
  //     pooling_h, pooling_w, stride_h, stride_w, padding_h, padding_w);
  this->op_ = std::make_unique<MaxPoolingOperator>(*maxpooling_op);
};

void MaxPoolingLayer::Forwards(
    const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
    std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  CHECK(this->op_ != nullptr);
  CHECK(this->op_->op_type_ == OpType::kOperatorMaxPooling);
  CHECK(!inputs.empty());
  // 获取到池化相关的属性
  const uint32_t padding_h = this->op_->padding_height();
  const uint32_t padding_w = this->op_->padding_width();
  const uint32_t kernel_h = this->op_->pooling_height();
  const uint32_t kernel_w = this->op_->pooling_width();
  const uint32_t stride_h = this->op_->stride_height();
  const uint32_t stride_w = this->op_->stride_width();

  const uint32_t batch_size = inputs.size();
  for (int i = 0; i < batch_size; ++i) {
    CHECK(!inputs.at(i)->empty());
    const std::shared_ptr<Tensor<float>> &input_data = inputs.at(i);
    // 如果padding_h,padding_w不为0的话，就做一个填充操作，周围填充一圈负无穷
    input_data->Padding({padding_h, padding_h, padding_w, padding_w},
                        std::numeric_limits<float>::lowest());
    // 获得输入特征图的大小、通道数量
    const uint32_t input_h = input_data->rows();
    const uint32_t input_w = input_data->cols();
    const uint32_t input_c = input_data->channels();
    const uint32_t output_c = input_c;

    // input_h 输入的高度
    // input_w 输入的宽度
    // kernel_h 窗口的高度
    // kernel_w 窗口的宽度
    // 计算输出特征图的大小
    const uint32_t output_h =
        uint32_t(std::floor((input_h - kernel_h) / stride_h + 1));
    const uint32_t output_w =
        uint32_t(std::floor((input_w - kernel_w) / stride_w + 1));
    CHECK(output_w > 0 && output_h > 0);
    std::shared_ptr<Tensor<float>> output_data =
        std::make_shared<Tensor<float>>(output_c, output_h, output_w);
    for (uint32_t ic = 0; ic < input_c; ++ic) {
      const arma::fmat &input_channel = input_data->at(ic);
      // 池化操作也是逐个通道进行的
      arma::fmat &output_channel = output_data->at(ic);
      // 每一行
      for (uint32_t r = 0; r < input_h - kernel_h + 1; r += stride_h) {
        //每一列
        for (uint32_t c = 0; c < input_w - kernel_w + 1; c += stride_w) {
          // 已知窗口开始位置和结束位置的时候，取得这一块区域
          const arma::fmat &region =
              input_channel.submat(r, c, r + kernel_h - 1, c + kernel_w - 1);
          // 取合围范围内的最大值
          output_channel.at(int(r / stride_h), int(c / stride_w)) =
              region.max();
        }
      }
    }
    outputs.push_back(output_data);
  }
};

std::shared_ptr<Layer>
MaxPoolingLayer::CreateInstance(const std::shared_ptr<Operator> &op) {
  CHECK(op->op_type_ == OpType::kOperatorMaxPooling);
  std::shared_ptr<Layer> max_layer = std::make_shared<MaxPoolingLayer>(op);
  return max_layer;
}

LayerRegistererWrapper kMaxPoolingLayer(OpType::kOperatorMaxPooling,
                                        MaxPoolingLayer::CreateInstance);

}; // namespace kuiper_infer
