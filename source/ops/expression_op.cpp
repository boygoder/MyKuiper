#include "ops/expression_op.hpp"
#include <glog/logging.h>

namespace kuiper_infer {

ExpressionOp::ExpressionOp(const std::string &expr)
    : Operator(OpType::kOperatorExpression), expr_(expr) {
  this->parser_ = std::make_shared<ExpressionParser>(this->expr_);
}

std::vector<std::shared_ptr<TokenNode>> ExpressionOp::Generate() {
  CHECK(this->parser_ != nullptr);
  this->nodes_ = this->parser_->Generate();
  return this->nodes_;
}
} // namespace kuiper_infer