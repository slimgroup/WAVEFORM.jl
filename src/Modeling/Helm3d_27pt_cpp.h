#include "math.h"
#include <thread>
#include <vector>
#include <complex>

using namespace std;

/**
   Enum definitions
 */
enum Mult_Mode {
    FORW_MULT,
    ADJ_MULT
};

enum Deriv_Mode {
    DERIV_MODE,
    NO_DERIV_MODE
};
    
template <class T,Mult_Mode m, Deriv_Mode d>
void do_Hmvp( const complex<T> * wn, const T * h, const int * n, const int * npml, complex<T> * y, const complex<T> * x, int zmin, int zmax);


template <class T,Mult_Mode m, Deriv_Mode d>
void do_Hmvp_mt( const complex<T> * wn, const T * h, const int * n, const int * npml, complex<T> * y, const complex<T> * x, int n_threads){
    vector<thread> th(n_threads);
    int zmin, zmax;    
    
    int dz = (int)(n[2])/n_threads;
   
    for (int i=0; i<n_threads; i++){
        zmin = i*dz;
        if (i<n_threads-1)
            zmax = (i+1)*dz;
        else
            zmax = n[2];
        th[i] = thread {do_Hmvp<T,m,d>,wn,h,n,npml,y,x,zmin,zmax};
    }

    for (auto &t : th) {
        t.join();
    }
    
}

