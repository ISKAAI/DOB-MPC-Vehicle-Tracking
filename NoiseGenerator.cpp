#include "NoiseGenerator.h"

noiseGenerator::noiseGenerator(const arma::vec& mean,const arma::vec& stddev,unsigned int seed)
        :mean_(mean),stddev_(stddev),dist_(0.0,1.0){
        if(seed == 0){
            std::random_device rd;
            generator_.seed(rd());
        }else{
            generator_.seed(seed);
        }
    }
arma::vec noiseGenerator::sample(){
            arma::vec noise(mean_.n_elem);
            
            for (arma::uword i =0; i<mean_.n_elem; ++i){
                noise(i) = mean_(i)+stddev_(i)*dist_(generator_);
            }
            return noise;
        }
void noiseGenerator::setStddev(const arma::vec& new_stddev) {
    stddev_ = new_stddev;
}

arma::vec noiseGenerator::Disturbance(const arma::vec& d_init,double omega,double current_time){
    arma::vec d = d_init;
    for (arma::uword i = 0;i<d.n_elem;++i){
        d(i) = d_init(i)*std::sin(omega*current_time);
    }
    return d;
}