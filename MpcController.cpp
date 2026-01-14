#include "MpcController.h"
#include <iostream>

MpcController::MpcController(arma::mat A,arma::mat B,arma::mat C,
                unsigned int np, unsigned int nc,
                arma::mat R, arma::mat Q,arma::mat P,
                arma::mat x0,
                arma::mat x_ref,
                arma::mat S)
                : A_(A),B_(B),C_(C),np_(np),nc_(nc),R_(R),Q_(Q),P_(P),x0_(x0),x_ref_(x_ref),S_(S)
                {
                    n = A_.n_rows;
                    m = B_.n_cols;
                    r = C_.n_rows;
                    Q_lifted_ = arma::kron(arma::eye(np_, np_), Q_);
                    R_lifted_ = arma::kron(arma::eye(nc_, nc_),R_);
                    d_=arma::zeros<arma::vec>(r);
}

arma::vec MpcController::computeControl(const arma::vec& input_limits,Vehicle&car,double T,const arma::vec& u_prev){
        
        O_.set_size(n*np_,n);
        O_.zeros();
        M_.set_size(n*np_,m*nc_);
        M_.zeros();
        arma::mat G_vector(n*np_,1,arma::fill::zeros);

        arma::mat Ad_power_i = arma::eye(n, n);

        arma::mat A_last = arma::eye(n, n);

    for (int i = 0; i < np_; ++i) {
        arma::vec xr_i = x_ref_.rows(i * n, (i + 1) * n - 1);
        arma::vec ur_i = {0.0, 0.0};
        double phi_ref = xr_i(2);
        double v_r_ref   = xr_i(3);
        double delta_f_ref = xr_i(4);

        arma::mat Ad_i = car.A_matrix(v_r_ref, phi_ref, delta_f_ref,T);
        arma::mat Bd_i = car.B_matrix(T);
        arma::vec gd_i = car.gd(xr_i, ur_i, T);
        arma::vec gd_total = gd_i+(d_*T);


        if (i == 0)
            A_last = Ad_i;
        else
            A_last = Ad_i * A_last;
        O_.rows(i * n, (i + 1) * n - 1) = A_last;

        if (i == 0)
            G_vector.rows(0, n - 1) = gd_total;
        else
            G_vector.rows(i * n, (i + 1) * n - 1) =
                Ad_i * G_vector.rows((i - 1) * n, i * n - 1) + gd_total;

        for (int j = 0; j < nc_; ++j) {
            if (j > i) continue;
            if (j == i) {
                M_.submat(i*n, j*m, (i+1)*n-1, (j+1)*m-1) = Bd_i;
            } else {
                M_.submat(i*n, j*m, (i+1)*n-1, (j+1)*m-1) =
                    Ad_i * M_.submat((i-1)*n, j*m, i*n-1, (j+1)*m-1);
            }
        }
    }
    // ---------- 终端 cost ----------
    arma::mat Q_mod = Q_lifted_;
    Q_mod.submat((np_-1)*n,(np_-1)*n,np_*n-1,np_*n-1)=P_; 
    
    // ---------- 构造代价函数 ----------
    arma::vec u_ref = {0.0, 0.0}; // u_ref = [v_ref, delta_f_ref=0]
    arma::vec U_ref = arma::repmat(u_ref, nc_, 1);
    //arma::vec d_lifted = arma::repmat(d_,np_,1);
    //arma::vec error_prediction = (O_ * x0_ + G_vector - x_ref_) + d_lifted;
    arma::vec error_prediction = (O_ * x0_ + G_vector - x_ref_);
    arma::mat F = 2*M_.t()*Q_mod*error_prediction - 2*R_lifted_*U_ref;
    arma::mat H = 2*(M_.t()*Q_mod*M_+R_lifted_);

    // ----------  舒适度  ---------
    int mn = m*nc_;
    arma::mat D1 = arma::zeros(mn,mn);
    for (int k =0;k<nc_; ++k){
        D1.submat(k*m,k*m,(k+1)*m-1,(k+1)*m-1) = arma::eye(m,m);
        if(k>0){
            D1.submat(k*m,(k-1)*m,(k+1)*m-1,k*m-1) = -arma::eye(m,m);
        }
    }

    arma::mat E = arma::zeros(mn,m);
    E.submat(0,0,m-1,m-1) = arma::eye(m,m);

    H += 2.0*(D1.t()*S_*D1);
    F +=-2.0*(D1.t()*S_*(E*u_prev));

    // ---------- 输入约束 ----------
    arma::vec u_min = {input_limits(1), input_limits(3)}; // [a_min, w_delta_min] 
    arma::vec u_max = {input_limits(0), input_limits(2)}; // [a_max, w_delta_max] 

    arma::vec umin_rep =arma::repmat(u_min,nc_,1);
    arma::vec umax_rep =arma::repmat(u_max,nc_,1);
    arma::mat G = arma::join_cols(arma::eye(m*nc_,m*nc_),-arma::eye(m*nc_,m*nc_));
    arma::vec h = arma::join_cols(umax_rep,-umin_rep);
    
    // ---------- 调用 OSQP 求解 ----------
    arma::vec u_opt = solveMpc(H, F, G, h);
    arma::vec u_k = u_opt.rows(0, m - 1);
    return u_k;
}

int MpcController::get_np(){return np_;}
int MpcController::get_nc(){return nc_;}
arma::mat MpcController::get_Q(){return Q_;}
arma::mat MpcController::get_R(){return R_;}
void MpcController::setModel(const arma::mat&A,const arma::mat&B,const arma::mat&C){A_=A; B_=B; C_=C;}
void MpcController::setState(const arma::vec& x0){x0_=x0;}
void MpcController::setReference(const arma::mat& ref_traj){x_ref_ = stackRef(ref_traj);}
void MpcController::setDisturbance(const arma::vec& d){d_ = d;}
// ----------  权重设置  ---------
void MpcController::setWeights(const arma::mat&Q_new,const arma::mat&R_mew){
    Q_ = Q_new;
    R_ = R_mew;
    Q_lifted_ = arma::kron(arma::eye(np_, np_), Q_);
    R_lifted_ = arma::kron(arma::eye(nc_, nc_),R_);
}

std::unique_ptr<OSQPCscMatrix, decltype(&free)>
    MpcController::makeCscMatrix(const arma::sp_mat& M){
    auto* mat =(OSQPCscMatrix*)malloc(sizeof(OSQPCscMatrix));
    mat->m = M.n_rows;
    mat->n = M.n_cols;
    mat->nzmax = M.n_nonzero;
    mat->x = (OSQPFloat*)M.values;
    mat->i = (OSQPInt*)M.row_indices;
    mat->p = (OSQPInt*)M.col_ptrs;
    mat->nz =-1;
    mat->owned = 0;
    return std::unique_ptr<OSQPCscMatrix, decltype(&free)>(mat, &free);
}

arma::vec MpcController::solveMpc(const arma::mat& H, const arma::vec&F,
    const arma::mat&G,const arma::vec&h){
        arma::mat H_sym = 0.5 * (H + H.t());
        arma::sp_mat P_sparse = arma::sp_mat(arma::trimatu(H_sym));
        arma::sp_mat A_sparse = arma::sp_mat(G);

        auto P_csc = makeCscMatrix(P_sparse);
        auto A_csc = makeCscMatrix(A_sparse);

        arma::vec l(G.n_rows, arma::fill::value(-OSQP_INFTY));
        arma::vec u = h;

        OSQPSettings settings;
        osqp_set_default_settings(&settings);
        settings.verbose = false;  // 不打印求解过程
        settings.alpha = 1.6;      // 松弛参数
        OSQPSolver* solver = nullptr;
        OSQPInt exitflag = osqp_setup(
            &solver,
            P_csc.get(),                  // Hessian 矩阵 P
            (const OSQPFloat*)F.memptr(), // 线性项 q
            A_csc.get(),                  // 约束矩阵 A
            (const OSQPFloat*)l.memptr(), // 约束下界 l
            (const OSQPFloat*)u.memptr(), // 约束上界 u
            G.n_rows,                     // m: 约束数量
            H.n_cols,                     // n: 优化变量数量
            &settings                    
        );

        if (exitflag) {
            std::cerr << "ERROR: osqp_setup failed!" << std::endl;
            return arma::vec(H.n_cols, arma::fill::zeros); 
        }
        exitflag = osqp_solve(solver);
        arma::vec u_opt;
        if (exitflag == 0 && solver->info->status_val == OSQP_SOLVED) {
            u_opt = arma::vec(solver->solution->x, H.n_cols);
        } else {
            std::cerr << "WARNING: OSQP did not solve the problem! Status: " << solver->info->status << std::endl;
            u_opt.zeros(H.n_cols); 
        }

        osqp_cleanup(solver);
        return u_opt;
    }

    arma::vec MpcController::stackRef(const arma::mat& ref_traj){
        arma::vec xr(n*np_, arma::fill::zeros);
        for (unsigned i=0;i<np_;++i){
            xr.subvec(i*n, (i+1)*n-1) = ref_traj.col(i);
      }
        return xr;
  }