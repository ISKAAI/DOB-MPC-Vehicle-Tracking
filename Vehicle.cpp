#include "Vehicle.h"

Vehicle::Vehicle(const arma::vec& initial_state,const arma::vec& physical_limits,double wheelbase){
            x_ = initial_state(0);
            y_ = initial_state(1);
            phi_ = initial_state(2);

            v_r_ = initial_state(3);
            delta_f_ = initial_state(4);

            v_max_=physical_limits(0);
            v_min_ = physical_limits(1);
            delta_f_max_ = physical_limits(2);
            delta_f_min_ = physical_limits(3);
            L_ = wheelbase;
        }

void Vehicle::update_state(double a,double w_delta,double T){
    x_+=T*v_r_*cos(phi_);
    y_+=T*v_r_*sin(phi_);
    phi_+=T*v_r_*tan(delta_f_)/L_;
    v_r_ += T*a;
    delta_f_ += T*w_delta;
    v_r_ = std::clamp(v_r_, v_min_, v_max_);
    delta_f_ = std::clamp(delta_f_, delta_f_min_, delta_f_max_);

}

arma::mat Vehicle::A_matrix_c(double v_r_ref,double phi_ref, double delta_f_ref){
            arma::mat Ac = arma::zeros<arma::mat>(5,5);
            Ac(0, 2) = -v_r_ref* sin(phi_ref);
            Ac(0, 3) = cos(phi_ref);

            Ac(1, 2) = v_r_ref * cos(phi_ref);
            Ac(1, 3) = sin(phi_ref);

            Ac(2,3) = tan(delta_f_ref)/L_;
            Ac(2,4) = v_r_ref/(L_*pow(cos(delta_f_ref),2));

            return Ac;
        }

arma::mat Vehicle::A_matrix(double v_r_ref,double phi_ref,double delta_f_ref,double T){
    arma::mat Ac = A_matrix_c(v_r_ref, phi_ref, delta_f_ref);
    arma::mat A = arma::eye(5,5)+ T * Ac;
    return A;
}

arma::mat Vehicle::B_matrix_c(){
    arma::mat Bc = arma::zeros<arma::mat>(5, 2);
    Bc(3, 0) = 1;
    Bc(4, 1) = 1;
    return Bc;
}

arma::mat Vehicle::B_matrix(double T){
    arma::mat Bc = B_matrix_c();
    arma::mat B = T*Bc;
    return B;
}
arma::mat Vehicle::C_matrix(){
    arma::mat C = arma::eye<arma::mat>(5,5);
    return C;
}

arma::vec Vehicle::gd(const arma::vec& xr, const arma::vec& ur,double T){
    double phi_ref = xr(2);
    double v_r_ref = xr(3);
    double delta_f_ref = xr(4);
    double a_ref = ur(0);
    double w_delta_ref = ur(1);
    arma::vec g = arma::zeros<arma::vec>(5);

    arma::vec f = {
    v_r_ref*cos(phi_ref),
    v_r_ref*sin(phi_ref),
    v_r_ref*tan(delta_f_ref)/L_,
    a_ref,
    w_delta_ref

    };
    arma::mat Ac = A_matrix_c(v_r_ref,phi_ref,delta_f_ref);
    arma::mat Bc = B_matrix_c();
    arma::vec gc = f-Ac*xr-Bc*ur;
    arma::vec gd = T*gc;
    return gd;

}


double Vehicle::get_x() const{ return x_;}
double Vehicle::get_y() const { return y_;}
double Vehicle::get_phi() const { return phi_; }
double Vehicle::get_v_r() const {return v_r_;}
double Vehicle::get_delta_f() const{return delta_f_;}
double Vehicle::get_v_max() const {return v_max_;}
double Vehicle::get_v_min() const {return v_min_;}
double Vehicle::get_delta_f_max() const {return delta_f_max_;}
double Vehicle::get_delta_f_min() const {return delta_f_min_;}
double Vehicle::get_L() const {return L_;}

void Vehicle::add_noise(const arma::vec& noise){
    x_+=noise(0);
    y_+=noise(1);
    phi_+=noise(2);
    v_r_+=noise(3);
    delta_f_+=noise(4);
}
void Vehicle::set_state(const arma::vec& state){
    x_ = state(0);
    y_ = state(1);
    phi_ = state(2);
    v_r_ = state(3);
    delta_f_ = state(4);
}

void Vehicle::add_disturbance(const arma::vec& d,double dt){
    x_ += d(0)*dt;
    y_ += d(1)*dt;
    phi_ += d(2)*dt;
    v_r_ += d(3)*dt;
    delta_f_ += d(4)*dt;
}