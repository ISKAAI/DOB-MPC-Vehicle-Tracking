#pragma once
#include <armadillo>

class PathGenerator{
    public:
        PathGenerator(double T,int steps,double v_r_ref);
        arma::mat straightLine(double y_ref) const;
        arma::mat uturn(double straight_len, double turn_radius) const;
        arma::mat pathChange(double lane_width,double lane_length);
        arma::mat turn_quintic(double straightLen, double radius) const;
    private:
        double T_;
        int steps_;
        double v_r_ref_;
};