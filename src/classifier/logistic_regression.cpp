#include <sklearn/classifier/logistic_regression.h>

#include <utils/binary_read_helper.h>

namespace sklearn::classifier {

LogisticRegression::LogisticRegression()
{}

void LogisticRegression::predict_proba(const feature_t& features, proba_t* proba)
{
    using namespace boost::numeric::ublas;
    vector<feature_t::value_type> input(features.size());
    std::copy(begin(features), end(features), input.begin());
    vector<feature_t::value_type> res = prod(weights_, trans(input)) + bias_;

    proba->clear();
    proba_t::value_type p_sum = 0;

    for (auto x : res)
    {
        proba->push_back(1 / (1 + exp(-x)));
        p_sum += proba->back();
    }

    if (proba->size() == 1)
    {
        proba->push_back(1 - proba->back());
        std::swap(proba->front(), proba->back());
    }
    else
    {
        for (auto& x : *proba)
            x /= p_sum;
    }
}

int LogisticRegression::init(const char* raw_data, int raw_data_size)
{
    std::stringstream s(std::string(raw_data, raw_data_size));

    using namespace boost::numeric::ublas;


    int ofs = 0;

    int32_t n_features, n_classes;

    ofs += utils::binary::readVal<int32_t>(s, n_classes);
    ofs += utils::binary::readVal<int32_t>(s, n_features);

    std::vector<double> matr_raw(n_features * n_classes);
    ofs += utils::binary::readVec(s, matr_raw);


    weights_.resize(n_classes, n_features);
    std::copy(matr_raw.begin(), matr_raw.end(), weights_.data().begin());

    std::vector<double> bias_raw(n_classes);
    ofs += utils::binary::readVec(s, bias_raw);

    bias_.resize(n_classes);
    std::copy(bias_raw.begin(), bias_raw.end(), bias_.begin());

    return ofs;
}

}
