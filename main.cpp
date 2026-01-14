#include <iostream>
#include <armadillo>
#include <string>
#include <fstream>
#include <cmath>
#include "osqp.h"
#include <deque>
#include <algorithm>
#include <chrono>

#include "Vehicle.h"
#include "filter.h"
#include "MpcController.h"
#include "PathGenerator.h"
#include "NoiseGenerator.h"


using namespace std;
// ----------  参考点  ---------
int findClosestRefIndex(const arma::vec& currentState, const arma::mat& ref_total) {
    double min_dist = 1e9; 
    int closest_idx = 0;
    for (int i = 0; i < ref_total.n_cols; ++i) {
        double dx = currentState(0) - ref_total(0, i);
        double dy = currentState(1) - ref_total(1, i);
        double dist = std::sqrt(dx*dx + dy*dy);
        
        if (dist < min_dist) {
            min_dist = dist;
            closest_idx = i;
        }
    }
    return closest_idx;
}

// ----------  延迟补偿  ---------
arma::vec compensateDelay(
    const arma::vec& x0_delayed_filter,
    const std::deque<arma::vec>& historyInput,
    double T,
    double delay,
    Vehicle& car
){
    int tau = static_cast<int>(std::ceil(delay / T));
    if (tau == 0) {
        return x0_delayed_filter;
    }
    auto non_linear_predict_step = [](arma::vec x_in,const arma::vec& u,double T_step,double L){
        double phi = x_in(2);
        double v_r = x_in(3);
        double delta_f = x_in(4);
        double a = u(0);
        double w_delta = u(1);

        arma::vec x_out = x_in;
        x_out(0) += T_step * v_r * cos(phi);
        x_out(1) += T_step * v_r * sin(phi);
        x_out(2) += T_step * v_r * tan(delta_f) / L;
        x_out(3) += T_step * a;
        x_out(4) += T_step * w_delta;
        return x_out;
    };
    arma::vec x_pred = x0_delayed_filter;
    double L = car.get_L();
    for (int i = 0;i<tau;++i){
        arma::vec u_i = historyInput[i];
        x_pred = non_linear_predict_step(x_pred,u_i,T,L);
    }
    return x_pred;
}


// ----------  路径模拟  ---------
arma::vec path_simulation(
    Vehicle& car,
    const arma::mat& ref_total,
    MpcController& mpc,
    kalmanFilter& kalmanFilter,
    string output,
    arma::vec input_limits,
    arma::vec speed_limit,
    arma::vec d_init,
    double v_r_ref,
    int steps,
    double T_mpc,
    double T_model,
    double delay,
    const arma::mat& Q_base, 
    const arma::mat& R_base,
    noiseGenerator& process_noise,        
    noiseGenerator& measurement_noise,
    bool enable_compensation
) {
    int ratio = static_cast<int>(round(T_mpc / T_model));
    double total_time = steps * T_mpc;
    int total_model_steps = static_cast<int>(total_time / T_model);
    int tau = static_cast<int>(std::ceil(delay / T_mpc));

    // ----------  历史输入 ----------  
    std::deque<arma::vec> input_history;
    for (int i = 0; i < tau; ++i) {
        input_history.push_back(arma::zeros<arma::vec>(2)); 
    }
    arma::vec ekf_prev_input = arma::zeros<arma::vec>(2);
    std::deque<arma::vec> state_history;
    arma::vec initial_state = {car.get_x(), car.get_y(), car.get_phi(), car.get_v_r(), car.get_delta_f()};
    for (int i = 0; i < tau; ++i) {
        state_history.push_back(initial_state);
    }
    arma::vec compute_times(total_model_steps / ratio, arma::fill::zeros);
    arma::vec disturbance = arma::zeros<arma::vec>(5);
    arma::vec prevInput = {0, 0};
    arma::vec current_u = {0, 0};
    int mpc_counter = 0; 

    arma::vec noisy_measurement = initial_state;
    arma::vec measured_state = initial_state;
    arma::vec filtered_state = initial_state;

    // ----------  输出文件 ----------  
    ofstream fout(output);
    fout << "t,x,x_ref,y,y_ref,phi,phi_ref,v_r,v_ref,delta_f,delta_ref,a,w_delta,"
        << "est_dist_y,real_dist_y"
        << endl;
    
    // ----------   仿真主循环 ----------  
    for (int k_model = 0; k_model < total_model_steps; ++k_model) {
        double t = k_model * T_model;
        double dist_time = (t > 2.0) ? (t - 2.0) : 0.0;
        arma::vec d_dis = process_noise.Disturbance(d_init,0.5,dist_time);
        if (k_model%ratio == 0){
            int mpc_step = mpc_counter++;

            measured_state = state_history.front();
            arma::vec v_meas = measurement_noise.sample();
            noisy_measurement = measured_state +v_meas;

            kalmanFilter.predict(ekf_prev_input,T_mpc);
            kalmanFilter.update(noisy_measurement);

            arma::vec x_hat_6d = kalmanFilter.getState();
            ekf_prev_input = input_history.front();

            arma::vec x_hat_5d = x_hat_6d.head(5);
            double est_dist_y = x_hat_6d(5);

            filtered_state = x_hat_5d;

            arma::vec compensated_state = compensateDelay(
                x_hat_5d,
                input_history,T_mpc,delay,car
            );

        
        // ----------   生成未来参考轨迹 ----------  
        arma::mat ref_traj(5, mpc.get_np());
        for (int j = 0; j < mpc.get_np(); ++j) {
            int ref_idx = mpc_step + 1 + j;
            if (ref_idx >= (int)ref_total.n_cols) ref_idx = ref_total.n_cols - 1;
            ref_traj.col(j).rows(0, 2) = ref_total.col(ref_idx);
            ref_traj(3, j) = v_r_ref;
            ref_traj(4, j) = 0.0;
        }
        // ----------  根据车速调整权重 ----------  
        double v= car.get_v_r();
        double v_low = speed_limit(1);
        double v_high = speed_limit(0);
        double w = (v - v_low)/(v_high-v_low);
        w = std::clamp(w,0.0,1.0);
        double scale = 1.0+3.0*w*w;
        arma::mat Q_now = Q_base;
        arma::mat R_now = R_base;
        Q_now(2,2) *= scale;
        Q_now(4,4) *= scale;
        R_now(1,1) *= scale;
        mpc.setWeights(Q_now,R_now);
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // ----------   MPC 设置与求解 ----------  
        mpc.setState(compensated_state); 
        mpc.setReference(ref_traj);
        arma::vec dist_vec = arma::zeros(5);
        if(enable_compensation){
            dist_vec(1) = est_dist_y;
        }
        mpc.setDisturbance(dist_vec);
        current_u = mpc.computeControl(input_limits, car, T_mpc, prevInput);
        auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed_ms = end_time - start_time;
            
            if (mpc_step < compute_times.n_elem) {
                compute_times(mpc_step) = elapsed_ms.count();
            }

        prevInput = current_u;

        // ----------  状态更新 ----------  
        input_history.pop_front();
        input_history.push_back(current_u);
        int ref_idx = std::min((int)(t / T_mpc), (int)ref_total.n_cols - 1);
            arma::vec ref_point = ref_total.col(ref_idx);

            fout << t << ","
                << car.get_x() << "," << ref_point(0) << ","  
                << car.get_y() << "," << ref_point(1) << ","  
                << car.get_phi() << "," << ref_point(2) << "," 
                << car.get_v_r() << "," << v_r_ref << ","      
                << car.get_delta_f() << "," << 0.0 << ","       
                << current_u(0) << "," << current_u(1) << ","
                << est_dist_y << "," << d_dis(1) 
                << endl;
    }
        car.update_state(current_u(0), current_u(1), T_model);
        arma::vec w_proc = process_noise.sample();
        car.add_noise(w_proc);
        car.add_disturbance(d_dis,T_model);

        state_history.pop_front();
        state_history.push_back({
            car.get_x(), car.get_y(), car.get_phi(),
            car.get_v_r(), car.get_delta_f()
            });
        
    }
    fout.close();
    return compute_times;
}

int main(){
    double v_r_initial =2;
    double v_r_ref = 2.5;
    arma::vec initial_state = {0, 0, 0,v_r_initial,0};
    arma::vec physical_limits = {25, 0, M_PI/2.5, -M_PI/2.5};
    arma::vec input_limits ={3, -5, 0.5, -0.5};
    arma::vec speed_limits = {15, 0};
    
    //----------- 参数设置 -----------
    double T_mpc=0.1;
    double T_model=0.005;
    double steps=800;
    double delay = 0.1;
    double car_L=2.9;

    //----------- 路径 -----------
    //arma::mat ref_total = path_1.straigntLine(y_ref);
    PathGenerator path_1(T_mpc,steps,v_r_ref);
    arma::mat ref_total = path_1.uturn(50,20);
    //arma::mat ref_total = path_1.pathChange(2,50);
    //arma::mat ref_total= path_1.turn_quintic(50,10);
    
    //----------- 噪声 -----------
    unsigned int fixed_seed = 42;
    arma::vec d_init = {0,0.3,0,0,0};
    arma::vec mean_proc = arma::zeros<arma::vec>(5);
    arma::vec studdev_proc = arma::vec({0.001,0.001,0.0001,0.05,0.005});
    noiseGenerator process_noise(mean_proc,studdev_proc,fixed_seed);

    arma::vec mean_meas = arma::zeros<arma::vec>(5);
    arma::vec studdev_meas = arma::vec{0.105, 0.105, 0.002, 0.003, 0.002};
    //arma::vec studdev_meas = arma::zeros<arma::vec>(5) + 1e-6;
    noiseGenerator measurement_noise(mean_meas,studdev_meas,fixed_seed);
    
    //----------- MPC -----------
    int np = 60;
    int nc = 15;
    arma::mat Q_mpc = arma::diagmat(arma::vec({25, 25, 45.0,5,50})); //MPC
    arma::mat R_mpc = arma::diagmat(arma::vec({1, 45}));//MPC
    arma::mat P_mpc =10*Q_mpc; 
    arma::mat S_blk = arma::diagmat(arma::vec({5.0,20.0}));
    arma::mat S_mpc = arma::kron(arma::eye(nc,nc),S_blk);
    arma::mat A_init = arma::eye(5, 5);
    arma::mat B_init = arma::zeros(5, 2);
    arma::mat C_init = arma::eye(5, 5);
    arma::vec x_ref = arma::zeros(5 * np, 1);
    MpcController mpc_template(A_init, B_init, C_init, np, nc, R_mpc, Q_mpc, P_mpc, initial_state, x_ref, S_mpc);

    //----------- Kalman -----------
    int n_state =6;
    int m_meas = 5;
    arma::mat H_ka = arma::eye<arma::mat>(m_meas, n_state);
    H_ka.submat(0,0,4,4) = arma::eye(5,5);

    arma::mat Q_ka = arma::diagmat(arma::vec({1e-4, 1e-4, 1e-4, 1e-2, 1e-3,1e-3}));

    arma::vec r_diag = arma::pow(studdev_meas,2);
    arma::mat R_ka = arma::diagmat(r_diag);
    
    kalmanFilter ekf_template(Q_ka,R_ka,H_ka,car_L);

    arma::vec x0_aug(6);
    x0_aug.rows(0,4) = initial_state;
    x0_aug(5) = 0.0;  
    ekf_template.init(x0_aug,arma::eye(6,6));

   cout << "--- Running: 1. Disturbed MPC (No Compensation) ---" << endl;
    Vehicle car_no_comp(initial_state, physical_limits, 2.9);
    MpcController mpc_no_comp = mpc_template;
    kalmanFilter ekf_no_comp = ekf_template; 
    noiseGenerator pn1(mean_proc, studdev_proc, fixed_seed);
    noiseGenerator mn1(mean_meas, studdev_meas, fixed_seed);
    
    path_simulation(
        car_no_comp, ref_total, mpc_no_comp, ekf_no_comp, 
        "traj_no_comp.csv", 
        input_limits, speed_limits, d_init, v_r_ref, steps, T_mpc, T_model,
        delay, Q_mpc, R_mpc, pn1, mn1,
        false 
    );

    // --- 实验 2: 有干扰，且有补偿 (DOB-MPC) ---
    cout << "--- Running: 2. DOB-MPC (With Compensation) ---" << endl;
    
    Vehicle car_with_comp(initial_state, physical_limits, 2.9);
    MpcController mpc_with_comp = mpc_template;
    kalmanFilter ekf_with_comp = ekf_template;
    noiseGenerator pn2(mean_proc, studdev_proc, fixed_seed);
    noiseGenerator mn2(mean_meas, studdev_meas, fixed_seed);
    
    path_simulation(
        car_with_comp, ref_total, mpc_with_comp, ekf_with_comp, 
        "traj_with_comp.csv",
        input_limits, speed_limits, d_init, v_r_ref, steps, T_mpc, T_model,
        delay, Q_mpc, R_mpc, pn2, mn2,
        true 
    );

    cout << "--- All simulations complete! ---" << endl;
    return 0;
}