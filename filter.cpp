#include "filter.h"
#include <iostream>

kalmanFilter::kalmanFilter(const arma::mat&Q,const arma::mat& R,const arma::mat& H, double L)
:Q_(Q),R_(R),H_(H),L_(L),n_(Q.n_rows),m_(R.n_rows),initialized_(false){
    x_hat_.zeros(n_);
    P_ = arma::eye<arma::mat>(n_,n_);
}

void kalmanFilter::init(const arma::vec& x0,const arma::mat& P0){
    x_hat_ = x0;
    P_ = P0;
    initialized_ = true;
}
arma::mat kalmanFilter::calculate_F_jacobian(const arma::vec&x, double T){
    double phi_hat = x(2);
    double v_r_hat = x(3);
    double delta_f_hat = x(4);
    arma::mat F = arma::eye<arma::mat>(n_,n_);
    F(0, 2) = -T * v_r_hat * sin(phi_hat);
    F(0, 3) = T * cos(phi_hat);

    F(1, 2) = T * v_r_hat * cos(phi_hat);
    F(1, 3) = T * sin(phi_hat);
    
    F(2, 3) = T * tan(delta_f_hat) / L_;
    F(2, 4) = T * v_r_hat / (L_ * cos(delta_f_hat) * cos(delta_f_hat));
    if (n_ == 6){
        F(1,5) = T;
    }

    return F;

}

void kalmanFilter::predict(const arma::vec& u,double T){
    if (!initialized_) return;
    arma::mat F = calculate_F_jacobian(x_hat_,T);
    x_hat_ = non_linear_predict(x_hat_, u, T,L_);
    P_ = F * P_ * F.t() + Q_;
}

void kalmanFilter::update(const arma::vec& z){
    if (!initialized_) return;
    arma::vec y = z-H_*x_hat_;
    arma::mat S = H_*P_*H_.t()+R_;
    arma::mat K = P_*H_.t()*arma::inv(S);
    x_hat_ = x_hat_ + K*y;
    arma::mat I = arma::eye<arma::mat>(n_,n_);
    P_ = (I - K*H_)*P_;
}
arma::vec kalmanFilter::getState() const {return x_hat_;}
arma::mat kalmanFilter::getCovariance() const {return P_;}
bool kalmanFilter::isInitialized() const {return initialized_;}

arma::vec kalmanFilter::non_linear_predict(arma::vec x_in, const arma::vec& u, double T, double L) {
    int sub_steps =10;
    double dt =T/sub_steps;
    arma::vec x_curr = x_in;    
    double a = u(0);
    double w_delta = u(1);
    double disturbance = 0.0;

    if (x_in.n_elem == 6){
        disturbance =x_in(5);
    }
    for (int i =0;i<sub_steps;++i){
        double phi = x_curr(2);
        double v_r = x_curr(3);
        double delta_f = x_curr(4);

        x_curr(0) += dt * v_r * cos(phi);
        x_curr(1) += dt * v_r * sin(phi) + dt * disturbance;
        x_curr(2) += dt * v_r * tan(delta_f) / L;
        x_curr(3) += dt * a;
        x_curr(4) += dt * w_delta;
    }

    return x_curr;
}

LowPassFilter::LowPassFilter(double alpha, int num_states)
:alpha_(alpha),n_(num_states),initialized_(false){
    y_prev_.zeros(n_);
}

void LowPassFilter::init(const arma::vec& x0){
    y_prev_=x0;
    initialized_ = true;
}

arma::vec LowPassFilter::update(const arma::vec& noisy_full_state){
    if(!initialized_){
        init(noisy_full_state);
    }
    arma::vec y_k = (1.0-alpha_)*noisy_full_state+alpha_*y_prev_;
    y_prev_ = y_k;
    return y_k;
}

arma::vec LowPassFilter::getState() const {
    return y_prev_;
}