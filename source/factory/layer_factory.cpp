#include "factory/layer_factory.hpp"
#include <glog/logging.h>

namespace kuiper_infer {
void LayerRegisterer::RegisterCreator(OpType op_type, const Creator &creator) {
  CHECK(creator != nullptr) << "Layer creator is empty";
  CreateRegistry &registry = Registry();
  CHECK_EQ(registry.count(op_type), 0)
      << "Layer type: " << int(op_type) << " has already registered!";
  registry.insert({op_type, creator});
}

std::shared_ptr<Layer>
LayerRegisterer::CreateLayer(const std::shared_ptr<Operator> &op) {
  CreateRegistry &registry = Registry();
  //获取Operator的类型
  const OpType op_type = op->op_type_;

  LOG_IF(FATAL, registry.count(op_type) <= 0)
      << "Can not find the layer type: " << int(op_type);
  //根据算子Type查找对应的layer创建函数的指针。
  const auto &creator = registry.find(op_type)->second;

  LOG_IF(FATAL, !creator) << "Layer creator is empty!";
  //构建！
  std::shared_ptr<Layer> layer = creator(op);
  LOG_IF(FATAL, !layer) << "Layer init failed!";
  return layer;
}

LayerRegisterer::CreateRegistry &LayerRegisterer::Registry() {
  //单例模式的关键，只初始化一次的注册表
  static CreateRegistry *kRegistry = new CreateRegistry();
  //检查是否初始化成功
  CHECK(kRegistry != nullptr) << "Global layer register init failed!";
  return *kRegistry;
}
} // namespace kuiper_infer