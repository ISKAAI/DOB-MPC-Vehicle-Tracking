# DOB-MPC-Vehicle-Tracking
A Model Predictive Control (MPC) based vehicle trajectory tracking algorithm integrated with Disturbance Observer (DOB) compensation. It supports typical scenarios such as straight lines, U-turns, lane changes, and quintic polynomial turns. The system incorporates Kalman filtering/low-pass filtering, control delay compensation, and noise/disturbance simulation, with comparative validation of the robustness improvement brought by disturbance compensation for trajectory tracking.

## Key Technicla Features
- **Core Algorithm**: DOB-MPC (MPC with Disturbance Observer compensation), addressing the tracking accuracy degradation of traditional MPC under disturbances/noise;
  
- **Multi-Scenario Support**: Built-in reference path generators for straight lines, U-turns, lane changes, quintic polynomial turns, etc.;
  
- **Engineering Optimizations**: Integrated state filtering (Kalman/low-pass), control delay compensation, and input constraint handling;
  
- **Comparative Validation**: Provides two sets of experiments ("MPC without Compensation" vs. "DOB-MPC") with visualization of trajectories, lateral errors, and disturbance estimation results;
  
- **Numerical Solver**: Fast solution of constrained MPC optimization problems based on the OSQP solver.

## Dependencies
### Build Dependencies
- C++17 or higher
- Armadillo (Linear Algebra Library)
- OSQP Solver
- CMake

### Visualization Dependencies
- Python 3.7+
- Third-party Libraries: Pandas numpy matplotlib

## Quick Start
Follow these steps to build and execute the simulation using the simplified `make run` command.
1. Build the Project
```
mkdir build
cmake ..
make
```
2. Run Simulation

  You don't need to manually find the binary. Use the custom target defined in `CMakeLists.txt`.
```
make run
```
  This command will automatically check for updates,compile and execute the simulation.

3. Visualization
  After the simulation finishes (generating `.csv` files), run the provided Python script to see the performance comparison:
```
cd ..
python3 plot.py
```

## Project Structure
```
DOB-MPC-Vehicle-Tracking/
├── MpcController.h/.cpp       # Core MPC controller implementation (OSQP solving, cost function construction, constraint handling)
├── Vehicle.h/.cpp             # Vehicle dynamics model (state update, A/B/C matrix calculation)
├── PathGenerator.h/.cpp       # Reference path generator (straight line, U-turn, lane change, quintic polynomial turn)
├── filter.h/.cpp              # Filtering module (Kalman filter, low-pass filter)
├── NoiseGenerator.h/.cpp      # Noise/disturbance generator (Gaussian noise, sinusoidal disturbance)
├── main.cpp                   # Main simulation program (experiment configuration, loop logic, result output)
├── plot.py                    # Result visualization script (reads CSV from build directory)
├── CMakeLists.txt             # CMake configuration (includes add_custom_target(run) for one-click run)
└── README.md                  # Project documentation
```

## Module Explanation
1. Vehicle Model
- Simplified front-wheel steering vehicle dynamics model with state variables including: x/y position, heading angle, longitudinal velocity, and front wheel angle;
- Supports state update, noise addition, disturbance injection, and provides continuous/discretized A/B matrix calculation.
2. MPC Controller
- Configurable prediction horizon (np) and control horizon (nc), supporting dynamic adjustment of cost function weights;
- Integrates input constraints (acceleration, front wheel angle rate) and terminal cost;
- Supports disturbance injection and compensation, with disturbance values estimated by DOB accessed via the setDisturbance interface.
3. Disturbance Observer (DOB)
- Extended state estimation based on Kalman filtering, treating lateral disturbance as an extended state to achieve real-time disturbance estimation and compensation.
4. Path Generator

The path generator provides multiple built-in functions to generate typical reference paths for trajectory tracking validation, with details as follows:
- ‘straightLine’ : Generates a straight-line reference path, with the key configurable parameter being the reference y-coordinate (to define the lateral position of the straight path).
- ‘uturn’ : Generates a U-turn reference path, with core configurable parameters including the straight segment length (before and after the turn) and the turning radius (to control the curvature of the U-turn).
- ‘pathChange’ : Generates a lane change reference path, with key configurable parameters including the lane width (to define the lateral offset of the lane change) and the lane change length (to control the smoothness of the lane change process).
- ‘turn_quintic’ : Generates a quintic polynomial turn reference path, with core configurable parameters including the straight segment length (before the polynomial turn) and the turning radius (to define the curvature of the polynomial turn).
5. Delay Compensation
- Predicts and compensates for delayed states using historical control inputs, addressing tracking errors caused by control execution delays (implemented in the `compensateDelay` function).


