#include "LinearRegression.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
LinearRegression globalModel;
struct Dataset { Matrix X; std::vector<double> y; };
// CSV numerico: ultima columna es la respuesta; estos archivos no usan campos entre comillas.
Dataset readCSV(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("No se pudo abrir: "+path);
    std::string line; std::getline(in,line); Dataset d;
    std::size_t columns=0, lineNumber=1;
    while (std::getline(in,line)) {
        ++lineNumber;
        if (line.find_first_not_of(" \t\r")==std::string::npos) continue;
        std::stringstream ss(line); std::string cell; std::vector<double> row;
        if (!line.empty() && line.back()==',') throw std::runtime_error("Celda vacia");
        while (std::getline(ss,cell,',')) {
            std::size_t used=0; double v=std::stod(cell,&used);
            if (cell.find_first_not_of(" \t\r",used)!=std::string::npos)
                throw std::runtime_error("Celda invalida en linea "+std::to_string(lineNumber));
            row.push_back(v);
        }
        if (row.size()<2 || (columns && columns!=row.size()))
            throw std::runtime_error("Columnas incorrectas");
        columns=row.size(); d.y.push_back(row.back()); row.pop_back(); d.X.push_back(row);
    }
    if (d.X.size()<5) throw std::runtime_error("Datos insuficientes");
    return d;
}
void runDataset(const std::string& path, const std::string& label) {
    Dataset d=readCSV(path), train, test;
    std::vector<std::size_t> indices(d.X.size()); std::iota(indices.begin(),indices.end(),0);
    // Fisher-Yates con generador fijo: mismo orden entre bibliotecas estandar.
    std::mt19937 rng(42);
    for (std::size_t i=indices.size()-1;i>0;--i) std::swap(indices[i],indices[rng()%(i+1)]);
    const std::size_t ntrain=indices.size()*4/5;
    for (std::size_t k=0;k<indices.size();++k) {
        auto i=indices[k]; Dataset& part=k<ntrain?train:test;
        part.X.push_back(d.X[i]); part.y.push_back(d.y[i]);
    }
    LinearRegression model; model.fit(train.X,train.y);
    auto predicted=model.predict(test.X);
    std::cout<<"\n"<<label<<" | entrenamiento="<<train.X.size()<<" prueba="<<test.X.size()<<"\n";
    auto w=model.getWeights();
    for (std::size_t j=0;j<w.size();++j) std::cout<<"w"<<j+1<<"="<<w[j]<<"\n";
    std::cout<<"bias="<<model.getBias()<<"\nMSE train="<<model.score(train.X,train.y)
             <<"\nMSE test="<<model.score(test.X,test.y)<<"\n";
    std::ofstream csv("resultados/"+label+"_predicciones.csv");
    if (!csv) throw std::runtime_error("Cree la carpeta resultados antes de ejecutar");
    csv<<std::setprecision(12);
    for (std::size_t j=0;j<w.size();++j) csv<<"x"<<j+1<<",";
    csv<<"real,prediccion,error\n";
    for (std::size_t i=0;i<predicted.size();++i) {
        for (double x:test.X[i]) csv<<x<<",";
        csv<<test.y[i]<<","<<predicted[i]<<","<<predicted[i]-test.y[i]<<"\n";
        std::cout<<"prueba "<<i<<": real="<<test.y[i]<<" pred="<<predicted[i]<<"\n";
    }
}
void memoryAnalysis() {
    LinearRegression local;
    // Propiedad exclusiva: liberacion automatica, incluso si se lanza una excepcion.
    auto dynamic=std::make_unique<LinearRegression>();
    const std::size_t n=3;
    auto array=std::make_unique<LinearRegression[]>(n);
    Matrix X{{1,2},{2,1},{3,4},{4,3}}; std::vector<double> y{5,4,11,10};
    globalModel.fit(X,y); local.fit(X,y); dynamic->fit(X,y);
    for (std::size_t i=0;i<n;++i) array[i].fit(X,y);
    std::cout<<"\nMEMORIA (bytes)\nsizeof(double)="<<sizeof(double)
             <<" sizeof(vector<double>)="<<sizeof(std::vector<double>)
             <<" sizeof(size_t)="<<sizeof(std::size_t)<<" sizeof(bool)="<<sizeof(bool)
             <<"\nalignof(modelo)="<<alignof(LinearRegression)<<"\n";
    std::cout<<"global="<<&globalModel<<" local="<<&local<<" dinamico="<<dynamic.get()<<"\n";
    std::cout<<"sizeof(modelo)="<<sizeof(local)<<" buffers por modelo="<<local.dynamicBytes()<<"\n";
    std::cout<<"arreglo="<<array.get()<<" sizeof(puntero crudo)="<<sizeof(array.get())
             <<" sizeof(unique_ptr)="<<sizeof(array)<<"\n";
    std::size_t buffers=0;
    for (std::size_t i=0;i<n;++i) {
        buffers+=array[i].dynamicBytes();
        std::cout<<"elemento "<<i<<" direccion="<<&array[i]<<"\n";
    }
    std::cout<<"arreglo objetos="<<n*sizeof(LinearRegression)<<" buffers="<<buffers
             <<" total contabilizado="<<n*sizeof(LinearRegression)+buffers<<"\n";
}
int main(int argc,char** argv) {
    try {
        std::cout<<std::fixed<<std::setprecision(8);
        const std::string root=argc>1?argv[1]:"datos";
        runDataset(root+"/Ice_cream_selling_data.csv","helados");
        runDataset(root+"/student_exam_scores.csv","estudiantes");
        memoryAnalysis();
    } catch (const std::exception& e) { std::cerr<<"Error: "<<e.what()<<"\n"; return 1; }
}
