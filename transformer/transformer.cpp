#include "transformer.h"

std::vector<Transformer::feature_t> Transformer::transform(const std::vector<Transformer::feature_t> &features)
{
    std::vector<feature_t> ret(0);
    for (auto&& f : features)
        ret.emplace_back(transform(f));
    return ret;
}
