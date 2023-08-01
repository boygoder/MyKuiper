#include "layer/relu_layer.hpp"
#include "factory/layer_factory.hpp"
#include "ops/relu_op.hpp"
#include <glog/logging.h>

namespace kuiper_infer {
ReluLayer::ReluLayer(const std::shared_ptr<Operator> &op) : Layer("Relu") {
  CHECK(op->op_type_ == OpType::kOperatorRelu)
      << "Operator has a wrong type: " << int(op->op_type_);
  // dynamic_cast是什么意思？ 就是判断一下op指针是不是指向一个relu_op类的指针
  // 这边的op不是ReluOperator类型的指针，就报错
  // 我们这里只接受ReluOperator类型的指针
  // 父类指针必须指向子类ReluOperator类型的指针
  ReluOperator *relu_op = dynamic_cast<ReluOperator *>(op.get());

  CHECK(relu_op != nullptr) << "Relu operator is empty";
  // 一个op实例和一个layer 一一对应 这里relu op对一个relu layer
  // 从shared_ptr获取的裸指针构造出unique_ptr
  this->op_ = std::make_unique<ReluOperator>(relu_op->get_thresh());
}

void ReluLayer::Forwards(
    const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
    std::vector<std::shared_ptr<Tensor<float>>> &outputs)

{
  CHECK(this->op_ != nullptr);
  CHECK(this->op_->op_type_ == OpType::kOperatorRelu);

  const uint32_t batch_size =
      inputs
          .size(); //一批x，放在vec当中，理解为batchsize数量的tensor，需要进行relu操作
  for (int i = 0; i < batch_size; ++i) {

    CHECK(!inputs.at(i)->empty());
    const std::shared_ptr<Tensor<float>> &input_data =
        inputs.at(i); //取出批次当中的一个张量

    //使用arma自带的transform,对张量中的每一个元素进行运算，进行relu运算
    input_data->data().transform([&](float value) {
      // 对张量中的每一个元素进行运算
      // 从operator中得到存储的属性
      float thresh = op_->get_thresh();
      // x >= thresh
      if (value >= thresh) {
        return value; // return x
      } else {
        // x<= thresh return 0.f;
        return 0.f;
      }
    });

    // 把结果y放在outputs中
    outputs.push_back(input_data);
  }
}
//创建该层的函数，用于工厂模式
std::shared_ptr<Layer>
ReluLayer::CreateInstance(const std::shared_ptr<Operator> &op) {
  // op本来存放在Operator的智能指针中，修改为子类ReluLayer.
  std::shared_ptr<Layer> relu_layer = std::make_shared<ReluLayer>(op);
  return relu_layer;
}
//注册函数，算子：创建函数
LayerRegistererWrapper kReluLayer(OpType::kOperatorRelu,
                                  ReluLayer::CreateInstance);
} // namespace kuiper_infer