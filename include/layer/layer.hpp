#ifndef KUIPER_COURSE_INCLUDE_LAYER_LAYER_HPP_
#define KUIPER_COURSE_INCLUDE_LAYER_LAYER_HPP_
#include "data/tensor.hpp"
#include <string>
namespace kuiper_infer {
class Layer {
public:
  //每一个Layer有自己的名字
  explicit Layer(const std::string &layer_name);

  virtual void
  Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
           std::vector<std::shared_ptr<Tensor<float>>> &outputs);
  // reluLayer中 inputs 等于 x , outputs 等于 y= x，if x>0
  // 计算得到的结果放在y当中，x是输入，放在inputs中

  virtual ~Layer() = default;

private:
  std::string layer_name_; // relu layer "relu"
};
} // namespace kuiper_infer
#endif // KUIPER_COURSE_INCLUDE_LAYER_LAYER_HPP_