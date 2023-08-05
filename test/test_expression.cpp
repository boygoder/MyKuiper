#include "layer/expression_layer.hpp"
#include "ops/expression_op.hpp"
#include "parser/parse_expression.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

// static void ShowNodes(const std::shared_ptr<kuiper_infer::TokenNode> &node) {
//   if (!node) {
//     return;
//   }
//   // 中序遍历,展示抽象语法树的节点
//   ShowNodes(node->left);
//   if (node->num_index < 0) {
//     if (node->num_index == -int(kuiper_infer::TokenType::TokenAdd)) {
//       LOG(INFO) << "ADD";
//     } else if (node->num_index == -int(kuiper_infer::TokenType::TokenMul)) {
//       LOG(INFO) << "MUL";
//     } else if (node->num_index == -int(kuiper_infer::TokenType::TokenDiv)) {
//       LOG(INFO) << "DIV";
//     }
//   } else {
//     LOG(INFO) << "NUM: " << node->num_index;
//   }
//   ShowNodes(node->right);
// }

// TEST(test_expression, expression1) {
//   using namespace kuiper_infer;
//   const std::string &statement = "add(@1,@2)";
//   ExpressionParser parser(statement);
//   const auto &node_tokens = parser.Generate();
//   ShowNodes(node_tokens);
// }

// TEST(test_expression, expression2) {
//   using namespace kuiper_infer;
//   const std::string &statement = "add(mul(@0,@1),@2)";
//   ExpressionParser parser(statement);
//   const auto &node_tokens = parser.Generate();
//   ShowNodes(node_tokens);
// }

// TEST(test_expression, expression3) {
//   using namespace kuiper_infer;
//   const std::string &statement = "add(mul(@0,@1),mul(@2,add(@3,@4)))";
//   ExpressionParser parser(statement);
//   const auto &node_tokens = parser.Generate();
//   ShowNodes(node_tokens);
// }

// TEST(test_expression, expression4) {
//   using namespace kuiper_infer;
//   const std::string &statement = "add(div(@0,@1),@2)";
//   ExpressionParser parser(statement);
//   const auto &node_tokens = parser.Generate();
//   ShowNodes(node_tokens);
// }

TEST(test_expression, add) {
  using namespace kuiper_infer;
  const std::string &expr = "add(@0,@1)";
  std::shared_ptr<ExpressionOp> expression_op =
      std::make_shared<ExpressionOp>(expr);
  ExpressionLayer layer(expression_op);
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
  layer.Forwards(inputs, outputs);
  for (int i = 0; i < batch_size; ++i) {
    const auto &result = outputs.at(i);
    for (int j = 0; j < result->size(); ++j) {
      ASSERT_EQ(result->index(j), 3.f);
    }
  }
}

TEST(test_expression, div) {
  using namespace kuiper_infer;
  const std::string &expr = "div(@0,@1)";
  std::shared_ptr<ExpressionOp> expression_op =
      std::make_shared<ExpressionOp>(expr);
  ExpressionLayer layer(expression_op);
  std::vector<std::shared_ptr<ftensor>> inputs;
  std::vector<std::shared_ptr<ftensor>> outputs;

  int batch_size = 4;
  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
    input->Fill(2.f);
    inputs.push_back(input);
  }

  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
    input->Fill(4.f);
    inputs.push_back(input);
  }

  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<ftensor> output = std::make_shared<ftensor>(3, 224, 224);
    outputs.push_back(output);
  }
  layer.Forwards(inputs, outputs);

  for (int i = 0; i < batch_size; ++i) {
    const auto &result = outputs.at(i);
    for (int j = 0; j < result->size(); ++j) {
      ASSERT_EQ(result->index(j), 0.5f);
    }
  }
}

TEST(test_expression, mul_and_add) {
  using namespace kuiper_infer;
  const std::string &expr = "add(mul(@0,@1),@2)";
  std::shared_ptr<ExpressionOp> expression_op =
      std::make_shared<ExpressionOp>(expr);
  ExpressionLayer layer(expression_op);
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
  layer.Forwards(inputs, outputs);
  for (int i = 0; i < batch_size; ++i) {
    const auto &result = outputs.at(i);
    for (int j = 0; j < result->size(); ++j) {
      ASSERT_EQ(result->index(j), 5.f);
    }
  }
}
