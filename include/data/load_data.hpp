#ifndef KUIPER_COURSE_INCLUDE_DATA_LOAD_DATA_HPP_
#define KUIPER_COURSE_INCLUDE_DATA_LOAD_DATA_HPP_
#include "data/tensor.hpp"
#include <armadillo>

namespace kuiper_infer {
class CSVDataLoader {
public:
  /**
   * 从csv文件中初始化张量
   * @param file_path csv文件的路径
   * @param split_char 分隔符号
   * @return 根据csv文件得到的张量
   */
  static std::shared_ptr<Tensor<float>> LoadData(const std::string &file_path,
                                                 char split_char = ',');
  /**
   * 从csv文件中初始化张量,第一行是描述符
   * @param file_path csv文件的路径
   * @param split_char 分隔符号
   * @return 根据csv文件得到的张量
   */
  static std::shared_ptr<Tensor<float>>
  LoadDataWithHeader(const std::string &file_path,
                     std::vector<std::string> &headers, char split_char = ',');

private:
  /**
   * 得到csv文件的尺寸大小，LoadData中根据这里返回的尺寸大小初始化返回的fmat
   * @param file csv文件的路径
   * @param split_char 分割符号
   * @return 根据csv文件的尺寸大小
   */
  static std::pair<size_t, size_t> GetMatrixSize(std::ifstream &file,
                                                 char split_char);
};
} // namespace kuiper_infer
#endif // KUIPER_COURSE_INCLUDE_DATA_LOAD_DATA_HPP_
