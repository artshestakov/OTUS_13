#include <iostream>
#include <stdexcept>
#include <filesystem>
#include "AccuracyFinder.h"
#include <boost/program_options.hpp>
//-----------------------------------------------------------------------------
struct InputStruct
{
    std::filesystem::path model_src;
    std::filesystem::path test_src;
    std::size_t input_dim;
    std::size_t hidden_dim;
    std::size_t output_dim;
};
//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    boost::program_options::options_description opt_descr("Available commands");
    opt_descr.add_options()
        ("help", "Print this message")
        ("model_path", boost::program_options::value<std::string>()->required(), "Path to the W files")
        ("test_path", boost::program_options::value<std::string>()->required(), "Path to the csv file")
        ("input", boost::program_options::value<std::size_t>()->required()->default_value(784), "columns")
        ("hidden", boost::program_options::value<std::size_t>()->required()->default_value(128), "rows")
        ("output", boost::program_options::value<std::size_t>()->required()->default_value(10), "class amount")
        ;

    boost::program_options::variables_map varMap;
    try
    {
        auto res = boost::program_options::command_line_parser(argc, argv)
            .options(opt_descr).run();

        boost::program_options::store(res, varMap);
        if (varMap.count("help") != 0)
        {
            std::cout << opt_descr << "\n";
            return EXIT_SUCCESS;
        }
        boost::program_options::notify(varMap);
    }
    catch (const boost::program_options::error& error)
    {
        std::cerr << "Error while parsing command-line arguments: "
            << error.what() << "\nPlease use --help to see help message\n";
        return EXIT_FAILURE;
    }

    std::string model_path = varMap["model_path"].as<std::string>();
    std::string test_path = varMap["test_path"].as<std::string>();
    std::size_t input = varMap["input"].as<std::size_t>();
    std::size_t hidden = varMap["hidden"].as<std::size_t>();
    std::size_t output = varMap["output"].as<std::size_t>();

    AccuracyFinder finder(input, hidden, output, model_path);
    std::cout << finder.GetAccuracy(test_path) << std::endl;

    return EXIT_SUCCESS;
}
//-----------------------------------------------------------------------------
