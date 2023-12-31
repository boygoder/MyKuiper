#include "layer/abstract/layer_factory.hpp"
#include "runtime/runtime_ir.hpp"
#include <glog/logging.h>

namespace kuiper_infer {
void LayerRegisterer::RegisterCreator(const std::string &layer_type,
                                      const Creator &creator) {
  CHECK(creator != nullptr) << "Layer creator is empty";
  CreateRegistry &registry = Registry();
  CHECK_EQ(registry.count(layer_type), 0)
      << "Layer type: " << layer_type << " has already registered!";
  registry.insert({layer_type, creator});
}

std::shared_ptr<Layer>
LayerRegisterer::CreateLayer(const std::shared_ptr<RuntimeOperator> &op) {
  CreateRegistry &registry = Registry();

  const std::string &layer_type = op->type;
  LOG_IF(FATAL, registry.count(layer_type) <= 0)
      << "Can not find the layer type: " << layer_type;
  const auto &creator = registry.find(layer_type)->second;

  LOG_IF(FATAL, !creator) << "Layer creator is empty!";
  std::shared_ptr<Layer> layer;
  const auto &status = creator(op, layer);
  LOG_IF(FATAL, status != ParseParameterAttrStatus::kParameterAttrParseSuccess)
      << "Create the layer: " << layer_type
      << " failed, error code: " << int(status);
  return layer;
}

LayerRegisterer::CreateRegistry &LayerRegisterer::Registry() {
  //单例模式的关键，static只初始化一次的注册表
  static CreateRegistry *kRegistry = new CreateRegistry();
  //检查是否初始化成功
  CHECK(kRegistry != nullptr) << "Global layer register init failed!";
  return *kRegistry;
}
} // namespace kuiper_infer
