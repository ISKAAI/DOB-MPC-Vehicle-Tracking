#pragma once
#include <armadillo>
#include <algorithm>

class Vehicle{
    private:
        double x_;
        double y_;
        double phi_;
        double v_r_;
        double delta_f_;
        double L_;
        double v_max_;
        double v_min_;
        double delta_f_max_;
        double delta_f_min_;
    public:
        Vehicle(const arma::vec& initial_state,const arma::vec& physical_limits,double wheelbase);
        void update_state(double a,double w_delta,double T);
        arma::mat A_matrix_c(double v_r_ref,double phi_ref, double delta_f_ref);
        arma::mat A_matrix(double v_r_ref,double phi_ref,double delta_f_ref,double T);
        arma::mat B_matrix_c();
        arma::mat B_matrix(double T);
        arma::mat C_matrix();
        arma::vec gd(const arma::vec& xr, const arma::vec& ur,double T);
        
        double get_x() const;
        double get_y() const;
        double get_phi() const;
        double get_v_r() const;
        double get_delta_f() const;
        double get_v_max() const;
        double get_v_min() const;
        double get_delta_f_max() const;
        double get_delta_f_min() const;
        double get_L() const;

        void add_noise(const arma::vec& noise);
        void set_state(const arma::vec& state);
        void add_disturbance(const arma::vec& d,double dt);

    };