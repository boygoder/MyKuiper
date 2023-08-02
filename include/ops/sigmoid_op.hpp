#ifndef KUIPER_COURSE_INCLUDE_OPS_SIGMOID_OP_HPP_
#define KUIPER_COURSE_INCLUDE_OPS_SIGMOID_OP_HPP_
#include "op.hpp"
namespace kuiper_infer {
// y = 1/(1+e^{-x}),不需要额外的信息
class SigmoidOperator : public Operator {
public:
  ~SigmoidOperator() override = default;

  explicit SigmoidOperator();
};
} // namespace kuiper_infer
#endif // KUIPER_COURSE_INCLUDE_OPS_SIGMOID_OP_HPP_