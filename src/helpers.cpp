#include "helpers.h"

namespace mnist
{
    Eigen::MatrixXf read_mat_from_file(std::size_t rows, std::size_t cols, const std::filesystem::path &matPath)
    {
        std::ifstream matFile(matPath.native());
        Eigen::MatrixXf matRes(rows, cols);

        for (std::size_t i = 0; i < rows; ++i)
        {
            for (std::size_t j = 0; j < cols; ++j)
            {
                std::string tmp;
                matFile >> tmp;
                double val = std::stod(tmp);
                matRes(i, j) = (float)val;
            }
        }

        return matRes;
    }

    void read_features(std::vector<std::string> &pixels, Classifier::features_t &features)
    {
        features.clear();
        for (const auto &pixel : pixels)
        {
            features.push_back(std::stof(pixel));
        }
    }

    std::vector<std::vector<std::string>> read_csv(const std::filesystem::path &csvPath)
    {
        std::vector<std::vector<std::string>> result;
        std::fstream csv(csvPath);

        for (std::string line; std::getline(csv, line);)
        {
            result.push_back(utilities::Split(line, ','));
        }

        return result;
    }

}

namespace utilities
{
    std::vector<std::string> Split(const std::string &str, char d)
    {
        std::vector<std::string> v;
        if (!str.empty()) //Если строка не пустая - анализируем
        {
            size_t pos = 0, pos_last = 0;
            while ((pos = str.find(d, pos_last)) != std::string::npos)
            {
                if (pos != 0)
                {
                    if ((pos - pos_last) > 0)
                    {
                        v.emplace_back(str.substr(pos_last, pos - pos_last));
                    }
                    pos_last = ++pos;
                }
                else
                {
                    ++pos_last;
                }
            }

            if (pos == std::string::npos)
            {
                size_t sz = str.size();
                if (pos_last < sz)
                {
                    v.emplace_back(str.substr(pos_last, sz - pos_last));
                }
            }
        }
        return v;
    }
}
