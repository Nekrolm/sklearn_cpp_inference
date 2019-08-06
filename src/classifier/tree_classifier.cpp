#include <sklearn/classifier/tree_classifier.h>

#include <algorithm>
#include <numeric>
#include <string>
#include <sstream>

#include <utils/binary_read_helper.h>

namespace sklearn::classifier {

TreeClassifier::TreeClassifier()
{}

TreeClassifier::~TreeClassifier()
{}

void TreeClassifier::predict_proba(const feature_t& features, proba_t* proba)
{
    predict_internal(features, proba);

    double sum_p = std::accumulate(proba->begin(), proba->end(), 0.);

    for (size_t i = 0; i < proba->size(); ++i)
    {
        proba->at(i) /= sum_p;
    }
}

void TreeClassifier::predict_internal(const feature_t& features, proba_t* proba)
{
    int node = 0;

    while (left_children_[node] != NO_NODE)
    {
        if (features[feature_id_[node]] <= threshold_[node])
            node = left_children_[node];
        else
            node = right_children_[node];
    }

    *proba = value_[node];
}

int TreeClassifier::init(const char* raw_data, int raw_data_size)
{
    std::stringstream s(std::string(raw_data, raw_data_size));

    int ofs = 0;

    int nodes_cnt, class_cnt;

    ofs += utils::binary::readVal(s, nodes_cnt);
    ofs += utils::binary::readVal(s, class_cnt);


    left_children_.resize(nodes_cnt);
    right_children_.resize(nodes_cnt);
    feature_id_.resize(nodes_cnt);
    threshold_.resize(nodes_cnt);
    value_.resize(nodes_cnt, proba_t(class_cnt));


    ofs += utils::binary::readVec(s, left_children_);
    ofs += utils::binary::readVec(s, right_children_);
    ofs += utils::binary::readVec(s, feature_id_);

    ofs += utils::binary::readVec(s, threshold_);

    for (int i = 0; i < nodes_cnt; ++i)
    {
        ofs += utils::binary::readVec(s, value_[i]);
    }

    return ofs;
}

}
