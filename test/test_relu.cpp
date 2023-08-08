#include "../source/layer/details/relu.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

TEST(test_layer, forward_relu1) {
  using namespace kuiper_infer;
  float thresh = 0.f;
  // 初始化一个relu operator 并设置属性
  std::shared_ptr<ReluLayer> relu_layer = std::make_shared<ReluLayer>();

  // 有三个值的一个tensor<float>
  std::shared_ptr<Tensor<float>> input =
      std::make_shared<Tensor<float>>(1, 1, 3);
  input->index(0) = -1.f; // output对应的应该是0
  input->index(1) = -2.f; // output对应的应该是0
  input->index(2) = 3.f;  // output对应的应该是3
  // 主要第一个算子，经典又简单，我们这里开始！

  std::vector<std::shared_ptr<Tensor<float>>> inputs; //作为一个批次去处理
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1); //放结果
  outputs.at(0) = std::make_shared<ftensor>(1, 1, 3);
  relu_layer->Forward(inputs, outputs);
  ASSERT_EQ(outputs.size(), 1);

  for (int i = 0; i < outputs.size(); ++i) {
    ASSERT_EQ(outputs.at(i)->index(0), 0.f);
    ASSERT_EQ(outputs.at(i)->index(1), 0.f);
    ASSERT_EQ(outputs.at(i)->index(2), 3.f);
  }
}

TEST(test_layer, forward_relu2) {
  using namespace kuiper_infer;
  float thresh = 0.f;
  std::shared_ptr<ReluLayer> relu_layer = std::make_shared<ReluLayer>();

  std::shared_ptr<Tensor<float>> input =
      std::make_shared<Tensor<float>>(1, 1, 3);
  input->index(0) = -1.f;
  input->index(1) = -2.f;
  input->index(2) = 3.f;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  outputs.at(0) = std::make_shared<ftensor>(1, 1, 3);
  inputs.push_back(input);
  relu_layer->Forward(inputs, outputs);
  ASSERT_EQ(outputs.size(), 1);
  for (int i = 0; i < outputs.size(); ++i) {
    ASSERT_EQ(outputs.at(i)->index(0), 0.f);
    ASSERT_EQ(outputs.at(i)->index(1), 0.f);
    ASSERT_EQ(outputs.at(i)->index(2), 3.f);
  }
}