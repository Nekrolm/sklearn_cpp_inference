#ifndef STDSCALER_H
#define STDSCALER_H


#include "transformer.h"

class StdScaler : public Transformer
{
public:
    StdScaler();
    virtual ~StdScaler() override {}

    virtual void init(const std::string& filename) override;
    virtual void transform(const feature_t& features, feature_t* transformed_features) override;
    virtual feature_t transform(const feature_t& features) override;

private:
    feature_t mean_;
    feature_t vars_;
};

#endif // STDSCALER_H
