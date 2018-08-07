#include "classifier.h"



void Classifier::predict_proba(const std::vector<Classifier::features_t> &features, std::vector<Classifier::proba_t> *proba)
{
    proba->clear();
    for (auto&& f : features){
        proba->emplace_back();
        predict_proba(f, &(proba->back()));
    }
}
