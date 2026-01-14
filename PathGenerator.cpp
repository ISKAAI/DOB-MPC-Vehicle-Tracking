#include "PathGenerator.h"
#include <cmath>

PathGenerator::PathGenerator(double T, int steps, double v_r_ref) :
    T_(T), steps_(steps), v_r_ref_(v_r_ref) {}

arma::mat PathGenerator::straightLine(double y_ref) const {
            arma::mat ref_total(3,steps_,arma::fill::zeros);
            for (int i =0;i<steps_;++i){
                ref_total(0,i)=v_r_ref_*i*T_;
                ref_total(1,i)=y_ref;
                ref_total(2,i)=0;
            }
            return ref_total;
}

arma::mat PathGenerator::uturn(double straight_len, double turn_radius) const {
    arma::mat ref_total(3, steps_, arma::fill::zeros);
    double t_straight1_end = straight_len / v_r_ref_;
    double t_turn_duration = (M_PI * turn_radius) / v_r_ref_;
    double t_turn_end = t_straight1_end + t_turn_duration;
    double cx = straight_len;
    double cy = turn_radius;

    for (int i = 0; i < steps_; ++i) {
        double t = i * T_;

        if (t < t_straight1_end) {
            // --- Phase 1: 初始直行 ---
            ref_total(0, i) = v_r_ref_ * t;
            ref_total(1, i) = 0.0;
            ref_total(2, i) = 0.0; // 航向为0度

        } else if (t < t_turn_end) {
            double s = v_r_ref_ * (t - t_straight1_end);
            double theta = s / turn_radius;
            double current_angle_on_circle = -M_PI_2 + theta;
            
            ref_total(0, i) = cx + turn_radius * cos(current_angle_on_circle);
            ref_total(1, i) = cy + turn_radius * sin(current_angle_on_circle);
            ref_total(2, i) = theta;

        } else {
            double x_at_turn_end = straight_len;
            double y_at_turn_end = 2 * turn_radius;
            double extra_t = t - t_turn_end;

            ref_total(0, i) = x_at_turn_end - v_r_ref_ * extra_t;
            ref_total(1, i) = y_at_turn_end;
            ref_total(2, i) = M_PI; 
        }
    }
    return ref_total;
}

arma::mat PathGenerator::pathChange(double lane_width,double lane_length){
    arma::mat ref_total(3,steps_,arma::fill::zeros);
    double t_start_change = lane_length / v_r_ref_;
    double t_change_duration = lane_length / v_r_ref_;
    double t_end_change = t_start_change + t_change_duration;

    for (int i = 0; i < steps_; ++i) {
        double t = i * T_;
        ref_total(0, i) = v_r_ref_ * t; 
        if (t < t_start_change) {

            ref_total(1, i) = 0;
            ref_total(2, i) = 0;

        } else if (t < t_end_change) {
            double t_local = t - t_start_change;
            ref_total(1, i) = lane_width / 2.0 * (1 - cos(M_PI * t_local / t_change_duration));
            double y_dot = (lane_width / 2.0) * (M_PI / t_change_duration) * sin(M_PI * t_local / t_change_duration);
            double x_dot = v_r_ref_;
            ref_total(2, i) = atan2(y_dot, x_dot);
        } else {
            ref_total(1, i) = lane_width;
            ref_total(2, i) = 0;
        }
    }
    return ref_total;
}

arma::mat PathGenerator::turn_quintic(double straightLen, double radius) const {
    arma::mat ref_total(3, steps_, arma::fill::zeros);

    double t1 = straightLen / v_r_ref_;          
    double T_turn = (M_PI_2 * radius) / v_r_ref_; // 转弯持续时间
    double t2 = t1 + T_turn;                      // 转弯结束时间

    for (int i = 0; i < steps_; ++i) {
        double t = i * T_;

        if (t < t1) {
            ref_total(0, i) = v_r_ref_ * t;
            ref_total(1, i) = 0.0;
            ref_total(2, i) = 0.0;
        } 
        else if (t < t2) {
            double tau = (t - t1) / T_turn;
            double phi = M_PI_2 * (10*pow(tau,3) - 15*pow(tau,4) + 6*pow(tau,5));
            double s = v_r_ref_ * (t - t1);
            double x = straightLen + radius * sin(phi);
            double y = radius * (1 - cos(phi));

            ref_total(0, i) = x;
            ref_total(1, i) = y;
            ref_total(2, i) = phi;
        } 
        else {
            double extra_t = t - t2;
            ref_total(0, i) = straightLen + radius;
            ref_total(1, i) = radius + v_r_ref_ * extra_t;
            ref_total(2, i) = M_PI_2;
        }
    }

    return ref_total;
}