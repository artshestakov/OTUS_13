#pragma once
//-----------------------------------------------------------------------------
#include "helpers.h"
#include "mlp_classifier.h"
//-----------------------------------------------------------------------------
class AccuracyFinder
{
public:
    AccuracyFinder(std::size_t input, std::size_t hidden, std::size_t output, const std::string &classfilier_path);
    ~AccuracyFinder();

    float GetAccuracy(const std::string& csv_path);

private:
    std::size_t m_Input;
    std::size_t m_Hidden;
    std::size_t m_Output;
    mnist::MlpClassifier* m_Classifier;
};
//-----------------------------------------------------------------------------
