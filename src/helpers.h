#include <fstream>
#include <vector>
#include <string>
#include <filesystem>

#include <Eigen/Dense>

#include "classifier.h"

namespace mnist
{

    Eigen::MatrixXf read_mat_from_file(size_t rows, size_t cols, const std::filesystem::path &);

    void read_features(std::vector<std::string> &, Classifier::features_t &);

    std::vector<std::vector<std::string>> read_csv(const std::filesystem::path &);

}

namespace utilities
{

    std::vector<std::string> Split(const std::string &, char);

}
