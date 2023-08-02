#include "layer/sigmoid_layer.hpp"
#include "factory/layer_factory.hpp"
#include "ops/sigmoid_op.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>
namespace kuiper_infer {

SigmoidLayer::SigmoidLayer(const std::shared_ptr<Operator> &op)
    : Layer("Sigmoid") {
  CHECK(op->op_type_ == OpType::kOperatorSigmoid)
      << "Operator has a wrong type: " << int(op->op_type_);
  // dynamic_cast用来判断op指针是不是指向一个sigmoid_op类的指针
  SigmoidOperator *sigmoid_op = dynamic_cast<SigmoidOperator *>(op.get());
  CHECK(sigmoid_op != nullptr) << "Sigmoid operator is empty";
  // 从shared_ptr获取的裸指针构造出unique_ptr
  this->op_ = std::make_unique<SigmoidOperator>();
};

void SigmoidLayer::Forwards(
    const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
    std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  CHECK(this->op_ != nullptr);
  CHECK(this->op_->op_type_ == OpType::kOperatorSigmoid);
  CHECK(!inputs.empty());
  const u_int32_t batch_size = inputs.size();
  for (int i = 0; i < batch_size; ++i) {
    CHECK(!inputs.at(i)->empty());
    //取出批次当中的一个张量
    const std::shared_ptr<Tensor<float>> &input_data = inputs.at(i);
    //使用arma自带的transform,对张量中的每一个元素进行运算
    input_data->data().transform(
        [&](float value) { return 1.0 / (1.0 + exp(-value)); });

    // 把结果y放在outputs中
    outputs.push_back(input_data);
  }
};
//待加入注册表的函数
std::shared_ptr<Layer>
SigmoidLayer::CreateInstance(const std::shared_ptr<Operator> &op) {
  std::shared_ptr<Layer> sigmoid_layer = std::make_shared<SigmoidLayer>(op);
  return sigmoid_layer;
};
//加入注册表
LayerRegistererWrapper kSigmoidLayer(OpType::kOperatorSigmoid,
                                     SigmoidLayer::CreateInstance);

} // namespace kuiper_infer