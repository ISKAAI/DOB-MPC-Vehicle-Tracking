#pragma once
#include <armadillo>
#include <memory>
#include "osqp.h"
#include "Vehicle.h"

class MpcController {
    public:
            MpcController(arma::mat A,arma::mat B,arma::mat C,
                    unsigned int np, unsigned int nc,
                    arma::mat R, arma::mat Q,arma::mat P,
                    arma::mat x0,
                    arma::mat x_ref,
                    arma::mat S);
            arma::vec computeControl(const arma::vec& input_limits,Vehicle&car,double T,const arma::vec& u_prev);
            void setDisturbance(const arma::vec& d);
            void setWeights(const arma::mat&Q_new,const arma::mat&R_new);
            void setModel(const arma::mat&A,const arma::mat&B,const arma::mat&C);
            void setState(const arma::vec& x0);
            void setReference(const arma::mat& ref_traj);
            int get_np();
            int get_nc();
            arma::mat get_Q();
            arma::mat get_R();

            
        
    private:
            unsigned int j;
            //m 输入维度 ,n 状态维度,r 输出维度 
            unsigned int m,n,r;
            arma::mat A_,B_,C_;
            arma::vec d_; //扰动
            arma::mat Q_,R_;
            arma::mat P_;
            arma::mat Q_lifted_,R_lifted_;
            arma::mat x0_;
            arma::mat x_ref_;
            unsigned int np_,nc_;
            //O 预测矩阵 M 控制矩阵 输入增益矩阵
            arma::mat M_,O_,K_;
            arma::mat S_;
            arma::vec stackRef(const arma::mat& ref_traj);
            arma::vec solveMpc(const arma::mat& H, const arma::vec&F,const arma::mat&G,const arma::vec&h);
            std::unique_ptr<OSQPCscMatrix, decltype(&free)>makeCscMatrix(const arma::sp_mat& M);

};