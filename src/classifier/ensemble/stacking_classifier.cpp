#include <sklearn/classifier/ensemble/stacking_classifier.h>

#include <sklearn/classifier/logistic_regression.h>
#include <sklearn/classifier/ensemble/random_forest_classifier.h>

#include <utils/binary_read_helper.h>

namespace sklearn::classifier::ensemble {

StackingClassifier::StackingClassifier()
{}

void StackingClassifier::predict_proba(const feature_t& features, proba_t* proba)
{
    feature_t Xfeat(0);

    for (auto& clf : estimators_)
    {
        proba_t p;
        clf->predict_proba(features, &p);
        Xfeat.push_back(p.back());
    }
    stacker_->predict_proba(Xfeat, proba);
}

int StackingClassifier::init(const char* raw_data, int raw_data_size)
{
    // TODO: different models

    stacker_ = std::make_shared<LogisticRegression>();
    int ofs = stacker_->init(raw_data, raw_data_size);

    int n = *reinterpret_cast<const int32_t*>(raw_data + ofs);
    ofs += sizeof(uint32_t);

    for (int i = 0; i < n; ++i)
    {
        auto clf = std::make_shared<RandomForestClassifier>(RandomForestClassifier::WEIGHTED_MEAN);
        ofs += clf->init(raw_data + ofs, raw_data_size - ofs);
        addClassifier(clf);
    }
    return ofs;
}

}
