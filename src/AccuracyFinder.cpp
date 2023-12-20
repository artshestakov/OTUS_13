#include "AccuracyFinder.h"
//-----------------------------------------------------------------------------
AccuracyFinder::AccuracyFinder(std::size_t input, std::size_t hidden, std::size_t output, const std::string& classfilier_path)
    : m_Input(input),
    m_Hidden(hidden),
    m_Output(output),
    m_Classifier(nullptr)
{
#ifdef WIN32
    char path_sep = '\\';
#else
    char path_sep = '/';
#endif

    auto l1 = mnist::read_mat_from_file(m_Input, m_Hidden, classfilier_path + path_sep + "w1.txt");
    auto l2 = mnist::read_mat_from_file(m_Hidden, m_Output, classfilier_path + path_sep + "w2.txt");

    m_Classifier = new mnist::MlpClassifier(l1.transpose(), l2.transpose());
}
//-----------------------------------------------------------------------------
AccuracyFinder::~AccuracyFinder()
{

}
//-----------------------------------------------------------------------------
float AccuracyFinder::GetAccuracy(const std::string& csv_path)
{
    mnist::MlpClassifier::features_t features;

    auto file = mnist::read_csv(csv_path);
    float all = (float)file.size();
    float right = 0;

    for (auto &fileString : file)
    {
        std::size_t expected = std::stoi(fileString.front());
        fileString.erase(fileString.begin());

        mnist::read_features(fileString, features);
        auto pred = m_Classifier->predict(features);

        if (expected = pred)
        {
            right++;
        }
    }

    float accuracy = right / all;
    return accuracy;
}
//-----------------------------------------------------------------------------
