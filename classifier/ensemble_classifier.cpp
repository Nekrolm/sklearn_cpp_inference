#include "ensemble_classifier.h"

#include <algorithm>
#include <numeric>
#include <cmath>


void EnsembleClassifier::predict_proba(const features_t& features, proba_t* proba)
{
    std::vector<proba_t> res(estimators_.size());



    for (size_t i = 0; i < estimators_.size(); ++i){
        estimators_[i]->predict_proba(features, &res[i]);
    }


    proba->resize(res[0].size());
    std::fill(proba->begin(), proba->end(), 0.);

    double w_sum = std::accumulate(begin(estimator_weights_), end(estimator_weights_), 0.);

    if (algo_ == WEIGHTED_MEAN)
    {
        for (size_t i = 0; i < estimators_.size(); ++i)
        {
            for (size_t j = 0; j < proba->size(); ++j)
            {
                proba->at(j) += res[i][j] * estimator_weights_[i];
            }
        }

        for (size_t j = 0; j < proba->size(); ++j)
            proba->at(j) /= w_sum;

    }else{

        for (size_t i = 0 ; i < estimators_.size(); ++i)
        {
            double lp_sum = 0;
            for (size_t j = 0; j < proba->size(); ++j)
            {
                lp_sum += (res[i][j] = log(std::max(2.22e-16, res[i][j])));
            }

            lp_sum /= proba->size();

            for (size_t j = 0; j < proba->size(); ++j)
            {
                proba->at(j) += (res[i][j] - lp_sum ) * (proba->size() - 1);
            }
        }

        double p_sum = 0.;

        for (size_t j = 0; j < proba->size(); ++j){
            p_sum += (proba->at(j) = exp(proba->at(j) / w_sum));
        }


        for (size_t j = 0; j < proba->size(); ++j){
            proba->at(j) /= p_sum;
        }

    }


}


