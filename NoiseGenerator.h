#pragma once
#include <armadillo>
#include <random>

class noiseGenerator{
    public:
            noiseGenerator(const arma::vec& mean,const arma::vec& stddev,unsigned int seed=0);
            arma::vec sample();
            void setStddev(const arma::vec& new_stddev);
            arma::vec Disturbance(const arma::vec& d_inti,double omega,double current_time);

    private:
            arma::vec mean_;
            arma::vec stddev_;
            std::default_random_engine generator_;
            std::normal_distribution<double> dist_;
};