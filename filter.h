#pragma once
#include <armadillo>
#include <cmath>

enum class FilterType{
    None,
    LowPass,
    Kalman
};

class kalmanFilter{
    public:
            kalmanFilter(const arma::mat&Q,const arma::mat& R,const arma::mat& H,double L);
            void init(const arma::vec& x0,const arma::mat& P0);
            void predict(const arma::vec& u,double T);
            void update(const arma::vec& z);
            arma::vec getState() const;
            arma::mat getCovariance() const;
            bool isInitialized() const;

    private:
            arma::vec x_hat_; 
            arma::mat P_;    
            arma::mat Q_;     // (n_ x n_) 过程噪声
            arma::mat R_;     // (m_ x m_) 测量噪声
            arma::mat H_;     // (m_ x n_) 测量矩阵
            int n_; 
            int m_; 
            bool initialized_;
            double L_;
            arma::mat calculate_F_jacobian(const arma::vec&x, double T);
            arma::vec non_linear_predict(arma::vec x_in, const arma::vec& u, double T, double L);

};

class LowPassFilter{
    public:
            LowPassFilter(double alpha, int num_states);
            void init(const arma::vec& x0);
            arma::vec update(const arma::vec& noisy_full_state);
            arma::vec getState() const;
    private:
            double alpha_;
            int n_;
            bool initialized_;
            arma::vec y_prev_;
};