#ifndef KUIPER_COURSE_INCLUDE_OPS_OP_HPP_
#define KUIPER_COURSE_INCLUDE_OPS_OP_HPP_
namespace kuiper_infer {
enum class OpType {
  kOperatorUnknown = -1,
  kOperatorRelu = 0,
  kOperatorSigmoid = 1,
};

class Operator {
public:
  OpType op_type_ = OpType::kOperatorUnknown; //不是一个具体节点 制定为unknown

  virtual ~Operator() = default; //虚基类

  explicit Operator(OpType op_type);
};

} // namespace kuiper_infer
#endif // KUIPER_COURSE_INCLUDE_OPS_OP_HPP_