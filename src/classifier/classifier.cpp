#include <sklearn/classifier/classifier.h>

namespace sklearn::classifier {

void Classifier::predict_proba(const std::vector<feature_t>& features, std::vector<proba_t>* proba)
{
    proba->clear();

    for (auto&& f : features)
    {
        proba->emplace_back();
        predict_proba(f, &(proba->back()));
    }
}

}
