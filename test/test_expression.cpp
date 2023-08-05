#include "parser/parse_expression.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

static void ShowNodes(const std::shared_ptr<kuiper_infer::TokenNode> &node) {
  if (!node) {
    return;
  }
  // 中序遍历,展示抽象语法树的节点
  ShowNodes(node->left);
  if (node->num_index < 0) {
    if (node->num_index == -int(kuiper_infer::TokenType::TokenAdd)) {
      LOG(INFO) << "ADD";
    } else if (node->num_index == -int(kuiper_infer::TokenType::TokenMul)) {
      LOG(INFO) << "MUL";
    } else if (node->num_index == -int(kuiper_infer::TokenType::TokenDiv)) {
      LOG(INFO) << "DIV";
    }
  } else {
    LOG(INFO) << "NUM: " << node->num_index;
  }
  ShowNodes(node->right);
}

TEST(test_expression, expression1) {
  using namespace kuiper_infer;
  const std::string &statement = "add(@1,@2)";
  ExpressionParser parser(statement);
  const auto &node_tokens = parser.Generate();
  ShowNodes(node_tokens);
}

TEST(test_expression, expression2) {
  using namespace kuiper_infer;
  const std::string &statement = "add(mul(@0,@1),@2)";
  ExpressionParser parser(statement);
  const auto &node_tokens = parser.Generate();
  ShowNodes(node_tokens);
}

TEST(test_expression, expression3) {
  using namespace kuiper_infer;
  const std::string &statement = "add(mul(@0,@1),mul(@2,add(@3,@4)))";
  ExpressionParser parser(statement);
  const auto &node_tokens = parser.Generate();
  ShowNodes(node_tokens);
}

TEST(test_expression, expression4) {
  using namespace kuiper_infer;
  const std::string &statement = "add(div(@0,@1),@2)";
  ExpressionParser parser(statement);
  const auto &node_tokens = parser.Generate();
  ShowNodes(node_tokens);
}
