#include "LinearRegression.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
void require(bool ok) { if (!ok) throw std::runtime_error("Prueba fallida"); }
template<class F> void rejects(F f) { bool caught=false; try { f(); } catch(const std::exception&) { caught=true; } require(caught); }
int main() {
    LinearRegression m;
    rejects([&]{m.predict({{1}});});
    Matrix X{{0,0},{1,0},{0,1},{1,1},{2,1},{1,2}};
    std::vector<double> y; for(auto r:X)y.push_back(2*r[0]+3*r[1]+4);
    m.fit(X,y); auto w=m.getWeights();
    require(std::abs(w[0]-2)<1e-6 && std::abs(w[1]-3)<1e-6 && std::abs(m.getBias()-4)<1e-6);
    require(std::abs(m.predict({{3,4}})[0]-22)<1e-6);
    require(m.score(X,y)<1e-12);
    rejects([&]{m.predict({{1}});});
    rejects([&]{m.fit({{1},{2,3}},{1,2});});
    rejects([&]{m.fit({{1},{2}},{1});});
    rejects([&]{m.fit({},{});});
    m.fit({{2},{2},{2}},{7,7,7}); require(std::abs(m.predict({{2}})[0]-7)<1e-6);
    m.fit({{1},{2},{3}},{3,5,7}); require(std::abs(m.predict({{4}})[0]-9)<1e-6);
    std::cout<<"OK: coeficientes conocidos, prediccion externa, MSE, validaciones, columna constante y reentrenamiento\n";
}
