#include "LinearRegression.h"
#include <cmath>
#include <stdexcept>
LinearRegression::LinearRegression(double rate, std::size_t iterations)
    : learningRate(rate), epochs(iterations) {
    if (!std::isfinite(rate) || rate <= 0 || iterations == 0)
        throw std::invalid_argument("Parametros de entrenamiento invalidos");
}
void LinearRegression::validate(const Matrix& X) const {
    if (X.empty() || X[0].empty()) throw std::invalid_argument("Matriz vacia");
    for (const auto& row : X) {
        if (row.size() != X[0].size()) throw std::invalid_argument("Matriz irregular");
        for (double v : row) if (!std::isfinite(v))
            throw std::invalid_argument("Dato no finito");
    }
}
Matrix LinearRegression::data_scaling(const Matrix& X, bool learn) {
    validate(X);
    const auto p = X[0].size();
    if (learn) {
        // Reaprender el escalado invalida los pesos anteriores.
        trained = false;
        means.assign(p, 0); scales.assign(p, 0);
        for (const auto& row : X) for (std::size_t j=0;j<p;++j) means[j]+=row[j];
        for (double& v : means) v/=X.size();
        for (const auto& row : X) for (std::size_t j=0;j<p;++j)
            scales[j]+=(row[j]-means[j])*(row[j]-means[j]);
        for (double& v : scales) { v=std::sqrt(v/X.size()); if (v==0) v=1; }
    }
    if (means.size()!=p) throw std::invalid_argument("Escalado no disponible o dimension incorrecta");
    Matrix Z=X;
    for (auto& row : Z) for (std::size_t j=0;j<p;++j) row[j]=(row[j]-means[j])/scales[j];
    return Z;
}
void LinearRegression::fit(const Matrix& X, const std::vector<double>& y) {
    validate(X);
    if (X.size()!=y.size()) throw std::invalid_argument("X e y incompatibles");
    for (double v:y) if (!std::isfinite(v)) throw std::invalid_argument("Salida no finita");
    Matrix Z=data_scaling(X,true);
    weights.assign(X[0].size(),0); bias=0;
    for (std::size_t e=0;e<epochs;++e) {
        std::vector<double> gradient(weights.size(),0); double gb=0;
        for (std::size_t i=0;i<Z.size();++i) {
            double error=bias-y[i];
            for (std::size_t j=0;j<weights.size();++j) error+=weights[j]*Z[i][j];
            gb+=error;
            for (std::size_t j=0;j<weights.size();++j) gradient[j]+=error*Z[i][j];
        }
        bias-=learningRate*gb/Z.size();
        for (std::size_t j=0;j<weights.size();++j) weights[j]-=learningRate*gradient[j]/Z.size();
        if (!std::isfinite(bias)) throw std::runtime_error("Entrenamiento divergente");
        for (double v:weights) if (!std::isfinite(v)) throw std::runtime_error("Entrenamiento divergente");
    }
    trained=true;
}
std::vector<double> LinearRegression::predict(const Matrix& X) const {
    if (!trained) throw std::logic_error("Modelo no entrenado");
    validate(X);
    if (X[0].size()!=weights.size()) throw std::invalid_argument("Dimension de entrada incorrecta");
    std::vector<double> out(X.size(),bias);
    for (std::size_t i=0;i<X.size();++i) for (std::size_t j=0;j<weights.size();++j)
        out[i]+=weights[j]*(X[i][j]-means[j])/scales[j];
    return out;
}
double LinearRegression::score(const Matrix& X, const std::vector<double>& y) const {
    auto out=predict(X);
    if (y.size()!=out.size()) throw std::invalid_argument("X e y incompatibles");
    double mse=0;
    for (std::size_t i=0;i<y.size();++i) {
        if (!std::isfinite(y[i])) throw std::invalid_argument("Salida no finita");
        mse+=(out[i]-y[i])*(out[i]-y[i]);
    }
    return mse/y.size();
}
std::vector<double> LinearRegression::getWeights() const {
    if (!trained) throw std::logic_error("Modelo no entrenado");
    auto raw=weights;
    for (std::size_t j=0;j<raw.size();++j) raw[j]/=scales[j];
    return raw;
}
double LinearRegression::getBias() const {
    if (!trained) throw std::logic_error("Modelo no entrenado");
    double raw=bias;
    for (std::size_t j=0;j<weights.size();++j) raw-=weights[j]*means[j]/scales[j];
    return raw;
}
std::size_t LinearRegression::dynamicBytes() const {
    return (weights.capacity()+means.capacity()+scales.capacity())*sizeof(double);
}
