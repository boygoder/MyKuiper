#include "../source/layer/details/expression.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

TEST(test_expression, add) {
  using namespace kuiper_infer;
  const std::string &expr = "add(@0,@1)";
  std::shared_ptr<ExpressionLayer> expression_layer =
      std::make_shared<ExpressionLayer>(expr);
  std::vector<std::shared_ptr<ftensor>> inputs;
  std::vector<std::shared_ptr<ftensor>> outputs;

  int batch_size = 4;
  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
    input->Fill(1.f);
    inputs.push_back(input);
  }

  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
    input->Fill(2.f);
    inputs.push_back(input);
  }

  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<ftensor> output = std::make_shared<ftensor>(3, 224, 224);
    outputs.push_back(output);
  }
  expression_layer->Forward(inputs, outputs);
  for (int i = 0; i < batch_size; ++i) {
    const auto &result = outputs.at(i);
    for (int j = 0; j < result->size(); ++j) {
      ASSERT_EQ(result->index(j), 3.f);
    }
  }
}

TEST(test_expression, mul_and_add) {
  using namespace kuiper_infer;
  const std::string &expr = "add(mul(@0,@1),@2)";
  std::shared_ptr<ExpressionLayer> expression_layer =
      std::make_shared<ExpressionLayer>(expr);
  std::vector<std::shared_ptr<ftensor>> inputs;
  std::vector<std::shared_ptr<ftensor>> outputs;

  int batch_size = 4;
  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
    input->Fill(1.f);
    inputs.push_back(input);
  }

  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
    input->Fill(2.f);
    inputs.push_back(input);
  }

  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
    input->Fill(3.f);
    inputs.push_back(input);
  }

  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<ftensor> output = std::make_shared<ftensor>(3, 224, 224);
    outputs.push_back(output);
  }
  expression_layer->Forward(inputs, outputs);
  for (int i = 0; i < batch_size; ++i) {
    const auto &result = outputs.at(i);
    for (int j = 0; j < result->size(); ++j) {
      ASSERT_EQ(result->index(j), 5.f);
    }
  }
}
