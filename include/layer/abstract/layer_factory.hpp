#ifndef KUIPER_COURSE_INCLUDE_LAYER_LAYER_FACTORY_HPP_
#define KUIPER_COURSE_INCLUDE_LAYER_LAYER_FACTORY_HPP_
#include "layer.hpp"
#include "runtime/runtime_op.hpp"

namespace kuiper_infer {
class LayerRegisterer {
public:
  using Creator =
      ParseParameterAttrStatus (*)(const std::shared_ptr<RuntimeOperator> &op,
                                   std::shared_ptr<Layer> &layer);

  using CreateRegistry = std::map<std::string, Creator>;

  static void RegisterCreator(const std::string &layer_type,
                              const Creator &creator);

  static std::shared_ptr<Layer>
  CreateLayer(const std::shared_ptr<RuntimeOperator> &op);

  static CreateRegistry &Registry();
};

// Wrapper:装饰器，将函数调用的写法，更换为创建类对象的写法。从而可以直接写在类定义的cpp文件中。
class LayerRegistererWrapper {
public:
  LayerRegistererWrapper(const std::string &layer_type,
                         const LayerRegisterer::Creator &creator) {
    LayerRegisterer::RegisterCreator(layer_type, creator);
  }
};

} // namespace kuiper_infer
#endif // KUIPER_COURSE_INCLUDE_FACTORY_LAYER_FACTORY_HPP_