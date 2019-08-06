#pragma once
#include <sklearn/classifier/classifier.h>
#include <memory>


namespace sklearn::classifier::ensemble {

class SKLEARN_API EnsembleClassifier : public Classifier
{
public:
    enum AlgoType
    {
        WEIGHTED_MEAN,
        SAMME_R
    };


    EnsembleClassifier(AlgoType type = WEIGHTED_MEAN)
    {
        algo_ = type;
    }

    virtual ~EnsembleClassifier() override {}

    virtual void predict_proba(const feature_t& features, proba_t* proba) override;

    virtual int init(const char* raw_data, int raw_data_size) override { return 0; }

    virtual bool init(const std::string& filename) override { return false; }

    void addClassifier(std::shared_ptr<Classifier> clf, double w)
    {
        estimators_.push_back(clf);
        estimator_weights_.push_back(w);
    }

private:
    std::vector<std::shared_ptr<Classifier>> estimators_;
    std::vector<double> estimator_weights_;

    AlgoType algo_;
};

}
