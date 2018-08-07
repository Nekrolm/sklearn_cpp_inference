#include "transformer.h"

#include <fstream>

std::vector<Transformer::feature_t> Transformer::transform(const std::vector<Transformer::feature_t> &features)
{
    std::vector<feature_t> ret(0);
    for (auto&& f : features)
        ret.emplace_back(transform(f));
    return ret;
}

bool Transformer::init(const std::string& filename)
{
    std::ifstream input(filename, std::ios_base::in | std::ios_base::binary);
    std::vector<char> data(std::istreambuf_iterator<char>(input), (std::istreambuf_iterator<char>()));

    return init(data.data(), data.size()) > 0;

}
