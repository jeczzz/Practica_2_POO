#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H
#include <vector>
#include <cstddef>
using Matrix = std::vector<std::vector<double>>;
// Modelo con escalado aprendido exclusivamente del conjunto de entrenamiento.
class LinearRegression {
    std::vector<double> weights, means, scales;
    double bias = 0.0;
    double learningRate;
    std::size_t epochs;
    bool trained = false;
    void validate(const Matrix& X) const;
public:
    explicit LinearRegression(double rate = 0.05, std::size_t iterations = 10000);
    Matrix data_scaling(const Matrix& X, bool learn = false);
    void fit(const Matrix& X, const std::vector<double>& y);
    std::vector<double> predict(const Matrix& X) const;
    double score(const Matrix& X, const std::vector<double>& y) const;
    std::vector<double> getWeights() const;
    double getBias() const;
    std::size_t dynamicBytes() const;
};
#endif
