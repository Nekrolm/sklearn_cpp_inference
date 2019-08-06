#ifndef STACKING_CLASSIFIER_H
#define STACKING_CLASSIFIER_H

#include <sklearn/classifier/classifier.h>
#include <memory>

namespace sklearn::classifier::ensemble {


class StackingClassifier : public Classifier
{
public:
    StackingClassifier();
    virtual ~StackingClassifier() override {}

    virtual void predict_proba (const feature_t& features, proba_t* proba) override;

    virtual int init (const char* raw_data, int raw_data_size) override;


    void addClassifier (std::shared_ptr<Classifier> clf)
    {
        estimators_.push_back (clf);
    }

    void setStacker(std::shared_ptr<Classifier> clf){
        stacker_ = clf;
    }


private:
    std::vector<std::shared_ptr<Classifier>> estimators_;
    std::shared_ptr<Classifier> stacker_;
};


}

#endif // STACKING_CLASSIFIER_H
