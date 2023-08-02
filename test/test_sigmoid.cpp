#include "factory/layer_factory.hpp"
#include "layer/sigmoid_layer.hpp"
#include "ops/sigmoid_op.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

TEST(test_layer, forward_sigmoid1) {
  using namespace kuiper_infer;
  std::shared_ptr<Operator> sigmoid_op = std::make_shared<SigmoidOperator>();
  std::shared_ptr<Tensor<float>> input =
      std::make_shared<Tensor<float>>(1, 1, 3);
  input->index(0) = 0.f;
  input->index(1) = -1.f;
  input->index(2) = 1.f;
  std::shared_ptr<Tensor<float>> real_output =
      std::make_shared<Tensor<float>>(1, 1, 3);
  for (int i = 0; i < 3; ++i) {
    real_output->index(i) = 1.0 / (1 + exp(-input->index(i)));
  }
  std::vector<std::shared_ptr<Tensor<float>>> inputs; //作为一个批次去处理
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs; //放结果
  SigmoidLayer layer(sigmoid_op);

  layer.Forwards(inputs, outputs);
  ASSERT_EQ(outputs.size(), 1);

  for (int i = 0; i < outputs.size(); ++i) {
    ASSERT_EQ(outputs.at(i)->index(0), real_output->index(0));
    ASSERT_EQ(outputs.at(i)->index(1), real_output->index(1));
    ASSERT_EQ(outputs.at(i)->index(2), real_output->index(2));
  }
};

TEST(test_layer, forward_sigmoid2) {
  using namespace kuiper_infer;
  std::shared_ptr<Operator> sigmoid_op = std::make_shared<SigmoidOperator>();

  std::shared_ptr<Tensor<float>> input =
      std::make_shared<Tensor<float>>(1, 1, 3);
  input->index(0) = 0.f;
  input->index(1) = -1.f;
  input->index(2) = 1.f;
  std::shared_ptr<Tensor<float>> real_output =
      std::make_shared<Tensor<float>>(1, 1, 3);
  for (int i = 0; i < 3; ++i) {
    real_output->index(i) = 1.0 / (1 + exp(-input->index(i)));
  }
  std::vector<std::shared_ptr<Tensor<float>>> inputs; //作为一个批次去处理
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs; //放结果
  std::shared_ptr<Layer> layer = LayerRegisterer::CreateLayer(sigmoid_op);

  layer->Forwards(inputs, outputs);
  ASSERT_EQ(outputs.size(), 1);

  for (int i = 0; i < outputs.size(); ++i) {
    ASSERT_EQ(outputs.at(i)->index(0), real_output->index(0));
    ASSERT_EQ(outputs.at(i)->index(1), real_output->index(1));
    ASSERT_EQ(outputs.at(i)->index(2), real_output->index(2));
  }
};