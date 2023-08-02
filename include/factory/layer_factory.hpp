#ifndef KUIPER_COURSE_INCLUDE_FACTORY_LAYER_FACTORY_HPP_
#define KUIPER_COURSE_INCLUDE_FACTORY_LAYER_FACTORY_HPP_
#include "layer/layer.hpp"
#include "ops/op.hpp"

namespace kuiper_infer {
class LayerRegisterer {
public:
  // typedef std::shared_ptr<Layer> (*Creator)(
  //   const std::shared_ptr<Operator> &op);
  // 函数指针
  using Creator =
      std::shared_ptr<Layer> (*)(const std::shared_ptr<Operator> &op);
  // typedef std::map<OpType, Creator> CreateRegistry;
  //注册表，算子类型：注册函数指针，
  using CreateRegistry = std::map<OpType, Creator>;
  //将op_type和对应的构建函数，加入注册表
  static void RegisterCreator(OpType op_type, const Creator &creator);
  //根据算子类型，查看注册表，构造对应的layer
  static std::shared_ptr<Layer>
  CreateLayer(const std::shared_ptr<Operator> &op);
  //创建注册表
  static CreateRegistry &Registry();
};

class LayerRegistererWrapper {
public:
  LayerRegistererWrapper(OpType op_type,
                         const LayerRegisterer::Creator &creator) {
    LayerRegisterer::RegisterCreator(op_type, creator);
  }
};

} // namespace kuiper_infer
#endif // KUIPER_COURSE_INCLUDE_FACTORY_LAYER_FACTORY_HPP_