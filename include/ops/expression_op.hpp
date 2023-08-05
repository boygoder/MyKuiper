#ifndef KUIPER_COURSE_INCLUDE_OPS_EXPRESSION_OP_HPP_
#define KUIPER_COURSE_INCLUDE_OPS_EXPRESSION_OP_HPP_

#include "op.hpp"
#include "parser/parse_expression.hpp"
#include <memory>
#include <string>
#include <vector>

namespace kuiper_infer {
//根据表达式进行运算的算子
class ExpressionOp : public Operator {
public:
  explicit ExpressionOp(const std::string &expr);
  std::vector<std::shared_ptr<TokenNode>> Generate();

private:
  //词法和语法解释器
  std::shared_ptr<ExpressionParser> parser_;
  // expr_得到的逆波兰表达式
  std::vector<std::shared_ptr<TokenNode>> nodes_;
  // pnnx的表达式字符串
  std::string expr_;
};
} // namespace kuiper_infer
#endif // KUIPER_COURSE_INCLUDE_OPS_EXPRESSION_OP_HPP_
