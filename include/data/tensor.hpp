/**
 * @file tensor.hpp
 * @brief 张量
 * @date 2023-07-31
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef KUIPER_COURSE_INCLUDE_TENSOR_HPP_
#define KUIPER_COURSE_INCLUDE_TENSOR_HPP_
#include <armadillo>
#include <memory>
#include <vector>

namespace kuiper_infer {

template <typename T> class Tensor {};

template <> class Tensor<uint8_t> {
  // 待实现，量化一个张量
};

template <> class Tensor<float> {
  // 元素都是float
public:
  explicit Tensor() = default;

  explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);

  explicit Tensor(const std::vector<uint32_t> &shapes);

  static std::shared_ptr<Tensor<float>> Create(uint32_t channels, uint32_t rows,
                                               uint32_t cols);

  Tensor(const Tensor &tensor);

  Tensor(Tensor &&tensor) noexcept;

  Tensor<float> &operator=(Tensor &&tensor) noexcept;

  Tensor<float> &operator=(const Tensor &tensor);

  uint32_t rows() const;

  uint32_t cols() const;

  uint32_t channels() const;

  uint32_t size() const;

  void set_data(const arma::fcube &data);

  bool empty() const;

  float index(uint32_t offset) const;

  float &index(uint32_t offset);

  std::vector<uint32_t> shapes() const;

  const std::vector<uint32_t> &raw_shapes() const;

  arma::fcube &data();

  const arma::fcube &data() const;

  arma::fmat &at(uint32_t channel);

  const arma::fmat &at(uint32_t channel) const;

  float at(uint32_t channel, uint32_t row, uint32_t col) const;

  float &at(uint32_t channel, uint32_t row, uint32_t col);

  void Padding(const std::vector<uint32_t> &pads, float padding_value);

  void Fill(float value);

  void Fill(const std::vector<float> &values);

  void Ones();

  void Rand();

  void Show();

  void ReRawshape(const std::vector<uint32_t> &shapes);

  void ReRawView(const std::vector<uint32_t> &shapes);

  static std::shared_ptr<Tensor<float>>
  ElementAdd(const std::shared_ptr<Tensor<float>> &tensor1,
             const std::shared_ptr<Tensor<float>> &tensor2);

  static std::shared_ptr<Tensor<float>>
  ElementMultiply(const std::shared_ptr<Tensor<float>> &tensor1,
                  const std::shared_ptr<Tensor<float>> &tensor2);

  static std::shared_ptr<Tensor<float>>
  ElementDivision(const std::shared_ptr<Tensor<float>> &tensor1,
                  const std::shared_ptr<Tensor<float>> &tensor2);

  void Flatten();

  void Transform(const std::function<float(float)> &filter);

  std::shared_ptr<Tensor> Clone();

  const float *raw_ptr() const;

private:
  void ReView(const std::vector<uint32_t> &shapes);
  std::vector<uint32_t>
      raw_shapes_; //张量数据的实际尺寸大小,会省略channel或row为一的维度。
  arma::fcube data_; //张量数据
};
using ftensor = Tensor<float>;
using sftensor = std::shared_ptr<Tensor<float>>;
} // namespace kuiper_infer
#endif // KUIPER_COURSE_INCLUDE_TENSOR_HPP_
