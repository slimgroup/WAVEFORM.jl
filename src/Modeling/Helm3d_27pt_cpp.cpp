#include "Helm3d_27pt_cpp.h"
#include <stdio.h>
#include <complex.h>
#include <math.h>


/*
  Numerical Constants
 */
template <class T>
constexpr T PI = M_PI;
template <class T>
constexpr T W1 = 1.8395262e-5;
template <class T>
constexpr T W2 = 0.296692333333333;
template <class T>
constexpr T W3 = (0.027476150000000);
template <class T>
constexpr T WM1 = (0.49649658);
template <class T>
constexpr T WM2 = (0.075168750000000);
template <class T>
constexpr T WM3 = (0.004373916666667);
template <class T>
constexpr T WM4 = (5.690375e-07);


/*
  Enum declarations
 */


//PML function type
enum class pml_t {
    LOWER, UPPER
};

// Used for referencing in to a 3-length array
// M - one point behind current pt
// N - current pt
// P - one point ahead current pt
enum class offset_idx : int {
    M = 0,
    N = 1,
    P = 2    
};

// Used for referencing in to a 3d, 3x3x3 array
enum offset3d {
    MMM = 0,
    NMM = 1,
    PMM = 2,
    MNM = 3,
    NNM = 4,
    PNM = 5,
    MPM = 6,
    NPM = 7,
    PPM = 8,
    MMN = 9,
    NMN = 10,
    PMN = 11,
    MNN = 12,
    NNN = 13,
    PNN = 14,
    MPN = 15,
    NPN = 16,
    PPN = 17,
    MMP = 18,
    NMP = 19,
    PMP = 20,
    MNP = 21,
    NNP = 22,
    PNP = 23,
    MPP = 24,
    NPP = 25,
    PPP = 26    
};

// Alias templates, need a C++14 compiler
template <class T>
using C = complex<T>;

template <class T>
using Cbuf_1d = std::array<C<T>,3>;

template <class T>
using Cbuf_3d = std::array<C<T>,27>;


/*
  
 */

inline int IDX1D3(int i, int j, int k, int nx, int ny, int nz)
{
    return i + j*ny + k*nx*ny;
}

inline int IDX1D3_disp(int i, int j, int k, int nx, int ny, int nz)
{
    return j + i*ny + k*nx*ny;
}

template <class T>
inline void print_nbrhd(const Cbuf_3d<T> & v)
{
    for(int k=0; k<3; k++ )
    {
        for(int j=0; j<3; j++)
        {
            for (int i=0; i<3; i++)
            {
                printf("%3.3e + i%3.3e ",real(v[IDX1D3(j,i,k,3,3,3)]),imag(v[IDX1D3(j,i,k,3,3,3)]));
            }
            printf("\n");
        }
        printf("\n");
    }
}


//Buffered pml functions in a window + neighbourhood information
template <class T>
struct pml_info {
    Cbuf_1d<T> pzlo_buf,pzhi_buf,pz_buf;
    Cbuf_1d<T> pylo_buf,pyhi_buf,py_buf;
    Cbuf_1d<T> pxlo_buf,pxhi_buf,px_buf;
    int x_hasL,x_hasR,y_hasL,y_hasR,z_hasL,z_hasR;
};


/** Gross macros to help GCC out
 **/
#define MMM_BDRY(...) __builtin_expect(p.x_hasL && p.y_hasL && p.z_hasL,1) ? ( __VA_ARGS__ ) : 0
#define NMM_BDRY(...) __builtin_expect(p.y_hasL && p.z_hasL,1) ? ( __VA_ARGS__ ) : 0
#define PMM_BDRY(...) __builtin_expect(p.x_hasR && p.y_hasL && p.z_hasL,1) ? ( __VA_ARGS__ ) : 0
#define MNM_BDRY(...) __builtin_expect(p.x_hasL && p.z_hasL,1) ? ( __VA_ARGS__ ) : 0
#define NNM_BDRY(...) __builtin_expect(p.z_hasL,1) ? ( __VA_ARGS__) : 0
#define PNM_BDRY(...) __builtin_expect(p.x_hasR && p.z_hasL,1) ? ( __VA_ARGS__) : 0

#define MPM_BDRY(...) __builtin_expect(p.x_hasL && p.y_hasR && p.z_hasL,1) ? ( __VA_ARGS__) : 0
#define NPM_BDRY(...) __builtin_expect(p.y_hasR &&  p.z_hasL,1) ? ( __VA_ARGS__) : 0
#define PPM_BDRY(...) __builtin_expect(p.x_hasR && p.y_hasR && p.z_hasL,1) ? ( __VA_ARGS__) : 0
#define MMN_BDRY(...) __builtin_expect(p.x_hasL && p.y_hasL,1) ? ( __VA_ARGS__) : 0
#define NMN_BDRY(...) __builtin_expect(p.y_hasL,1) ? ( __VA_ARGS__) : 0
#define PMN_BDRY(...) __builtin_expect(p.x_hasR && p.y_hasL,1) ? ( __VA_ARGS__) : 0
#define MNN_BDRY(...) __builtin_expect(p.x_hasL,1) ? ( __VA_ARGS__) : 0
#define PNN_BDRY(...) __builtin_expect(p.x_hasR,1) ? ( __VA_ARGS__) : 0
#define MPN_BDRY(...) __builtin_expect(p.x_hasL && p.y_hasR,1) ? ( __VA_ARGS__) : 0
#define NPN_BDRY(...) __builtin_expect(p.y_hasR,1) ? ( __VA_ARGS__) : 0
#define PPN_BDRY(...) __builtin_expect(p.x_hasR && p.y_hasR,1) ? ( __VA_ARGS__) : 0
#define MMP_BDRY(...) __builtin_expect(p.x_hasL && p.y_hasL && p.z_hasR,1) ? ( __VA_ARGS__) : 0
#define NMP_BDRY(...) __builtin_expect(p.y_hasL && p.z_hasR,1) ? ( __VA_ARGS__) : 0
#define PMP_BDRY(...) __builtin_expect(p.x_hasR && p.y_hasL &&  p.z_hasR,1) ? ( __VA_ARGS__) : 0
#define MNP_BDRY(...) __builtin_expect(p.x_hasL && p.z_hasR,1) ? ( __VA_ARGS__) : 0
#define NNP_BDRY(...) __builtin_expect(p.z_hasR,1) ? ( __VA_ARGS__) : 0
#define PNP_BDRY(...) __builtin_expect(p.x_hasR && p.z_hasR,1) ? ( __VA_ARGS__) : 0
#define MPP_BDRY(...) __builtin_expect(p.x_hasL && p.y_hasR && p.z_hasR,1) ? ( __VA_ARGS__) : 0
#define NPP_BDRY(...) __builtin_expect(p.y_hasR && p.z_hasR,1) ? ( __VA_ARGS__) : 0
#define PPP_BDRY(...) __builtin_expect(p.x_hasR && p.y_hasR && p.z_hasR,1) ? ( __VA_ARGS__) : 0

/**
   Neighbourhood functions
 */
template <class T>
inline void load_nbrhoodc( std::array<complex<T>,27> & x, const T * xr, const T * xi, int i, int j, int k, int nx, int ny, int nz, pml_info<T> p )
{
    using C = complex<T>;
    x[MMM] = MMM_BDRY(C( xr[ IDX1D3(i-1,j-1,k-1,nx,ny,nz) ], xi[ IDX1D3(i-1,j-1,k-1,nx,ny,nz) ] ));
    x[NMM] = NMM_BDRY(C( xr[ IDX1D3(i  ,j-1,k-1,nx,ny,nz) ], xi[ IDX1D3(i  ,j-1,k-1,nx,ny,nz) ] ));
    x[PMM] = PMM_BDRY(C( xr[ IDX1D3(i+1,j-1,k-1,nx,ny,nz) ], xi[ IDX1D3(i+1,j-1,k-1,nx,ny,nz) ] ));
    
    x[MNM] = MNM_BDRY(C( xr[ IDX1D3(i-1,j  ,k-1,nx,ny,nz) ], xi[ IDX1D3(i-1,j  ,k-1,nx,ny,nz) ] ));
    x[NNM] = NNM_BDRY(C( xr[ IDX1D3(i  ,j  ,k-1,nx,ny,nz) ], xi[ IDX1D3(i  ,j  ,k-1,nx,ny,nz) ] ));
    x[PNM] = PNM_BDRY(C( xr[ IDX1D3(i+1,j  ,k-1,nx,ny,nz) ], xi[ IDX1D3(i+1,j  ,k-1,nx,ny,nz) ] ));
    
    x[MPM] = MPM_BDRY(C( xr[ IDX1D3(i-1,j+1,k-1,nx,ny,nz) ], xi[ IDX1D3(i-1,j+1,k-1,nx,ny,nz) ] ));
    x[NPM] = NPM_BDRY(C( xr[ IDX1D3(i  ,j+1,k-1,nx,ny,nz) ], xi[ IDX1D3(i  ,j+1,k-1,nx,ny,nz) ] ));
    x[PPM] = PPM_BDRY(C( xr[ IDX1D3(i+1,j+1,k-1,nx,ny,nz) ], xi[ IDX1D3(i+1,j+1,k-1,nx,ny,nz) ] ));
    
    x[MMN] = MMN_BDRY(C( xr[ IDX1D3(i-1,j-1,k  ,nx,ny,nz) ], xi[ IDX1D3(i-1,j-1,k  ,nx,ny,nz) ] ));
    x[NMN] = NMN_BDRY(C( xr[ IDX1D3(i  ,j-1,k  ,nx,ny,nz) ], xi[ IDX1D3(i  ,j-1,k  ,nx,ny,nz) ] ));
    x[PMN] = PMN_BDRY(C( xr[ IDX1D3(i+1,j-1,k  ,nx,ny,nz) ], xi[ IDX1D3(i+1,j-1,k  ,nx,ny,nz) ] ));
    
    x[MNN] = MNN_BDRY(C( xr[ IDX1D3(i-1,j  ,k  ,nx,ny,nz) ], xi[ IDX1D3(i-1,j  ,k  ,nx,ny,nz) ] ));
    x[NNN] =          C( xr[ IDX1D3(i  ,j  ,k  ,nx,ny,nz) ], xi[ IDX1D3(i  ,j  ,k  ,nx,ny,nz) ] );
    x[PNN] = PNN_BDRY(C( xr[ IDX1D3(i+1,j  ,k  ,nx,ny,nz) ], xi[ IDX1D3(i+1,j  ,k  ,nx,ny,nz) ] ));
    
    x[MPN] = MPN_BDRY(C( xr[ IDX1D3(i-1,j+1,k  ,nx,ny,nz) ], xi[ IDX1D3(i-1,j+1,k  ,nx,ny,nz) ] ));
    x[NPN] = NPN_BDRY(C( xr[ IDX1D3(i  ,j+1,k  ,nx,ny,nz) ], xi[ IDX1D3(i  ,j+1,k  ,nx,ny,nz) ] ));
    x[PPN] = PPN_BDRY(C( xr[ IDX1D3(i+1,j+1,k  ,nx,ny,nz) ], xi[ IDX1D3(i+1,j+1,k  ,nx,ny,nz) ] ));
    
    x[MMP] = MMP_BDRY(C( xr[ IDX1D3(i-1,j-1,k+1,nx,ny,nz) ], xi[ IDX1D3(i-1,j-1,k+1,nx,ny,nz) ] ));
    x[NMP] = NMP_BDRY(C( xr[ IDX1D3(i  ,j-1,k+1,nx,ny,nz) ], xi[ IDX1D3(i  ,j-1,k+1,nx,ny,nz) ] ));
    x[PMP] = PMP_BDRY(C( xr[ IDX1D3(i+1,j-1,k+1,nx,ny,nz) ], xi[ IDX1D3(i+1,j-1,k+1,nx,ny,nz) ] ));
    
    x[MNP] = MNP_BDRY(C( xr[ IDX1D3(i-1,j  ,k+1,nx,ny,nz) ], xi[ IDX1D3(i-1,j  ,k+1,nx,ny,nz) ] ));
    x[NNP] = NNP_BDRY(C( xr[ IDX1D3(i  ,j  ,k+1,nx,ny,nz) ], xi[ IDX1D3(i  ,j  ,k+1,nx,ny,nz) ] ));
    x[PNP] = PNP_BDRY(C( xr[ IDX1D3(i+1,j  ,k+1,nx,ny,nz) ], xi[ IDX1D3(i+1,j  ,k+1,nx,ny,nz) ] ));
    
    x[MPP] = MPP_BDRY(C( xr[ IDX1D3(i-1,j+1,k+1,nx,ny,nz) ], xi[ IDX1D3(i-1,j+1,k+1,nx,ny,nz) ] ));
    x[NPP] = NPP_BDRY(C( xr[ IDX1D3(i  ,j+1,k+1,nx,ny,nz) ], xi[ IDX1D3(i  ,j+1,k+1,nx,ny,nz) ] ));
    x[PPP] = PPP_BDRY(C( xr[ IDX1D3(i+1,j+1,k+1,nx,ny,nz) ], xi[ IDX1D3(i+1,j+1,k+1,nx,ny,nz) ] ));
}

template <class T>
inline void load_nbrhoodr( std::array<complex<T>,27> & x, const T * xr, int i, int j, int k, int nx, int ny, int nz, pml_info<T> p )
{
    using C = complex<T>;
    x[MMM] = MMM_BDRY(C( xr[ IDX1D3(i-1,j-1,k-1,nx,ny,nz) ], 0 ));
    x[NMM] = NMM_BDRY(C( xr[ IDX1D3(i  ,j-1,k-1,nx,ny,nz) ], 0 ));    
    x[PMM] = PMM_BDRY(C( xr[ IDX1D3(i+1,j-1,k-1,nx,ny,nz) ], 0 ));
    
    x[MNM] = MNM_BDRY(C( xr[ IDX1D3(i-1,j  ,k-1,nx,ny,nz) ], 0 ));
    x[NNM] = NNM_BDRY(C( xr[ IDX1D3(i  ,j  ,k-1,nx,ny,nz) ], 0 ));
    x[PNM] = PNM_BDRY(C( xr[ IDX1D3(i+1,j  ,k-1,nx,ny,nz) ], 0 ));
    
    x[MPM] = MPM_BDRY(C( xr[ IDX1D3(i-1,j+1,k-1,nx,ny,nz) ], 0 ));
    x[NPM] = NPM_BDRY(C( xr[ IDX1D3(i  ,j+1,k-1,nx,ny,nz) ], 0 ));
    x[PPM] = PPM_BDRY(C( xr[ IDX1D3(i+1,j+1,k-1,nx,ny,nz) ], 0 ));
    
    x[MMN] = MMN_BDRY(C( xr[ IDX1D3(i-1,j-1,k  ,nx,ny,nz) ], 0 ));
    x[NMN] = NMN_BDRY(C( xr[ IDX1D3(i  ,j-1,k  ,nx,ny,nz) ], 0 ));
    x[PMN] = PMN_BDRY(C( xr[ IDX1D3(i+1,j-1,k  ,nx,ny,nz) ], 0 ));
    
    x[MNN] = MNN_BDRY(C( xr[ IDX1D3(i-1,j  ,k  ,nx,ny,nz) ], 0 ));
    x[NNN] =          C( xr[ IDX1D3(i  ,j  ,k  ,nx,ny,nz) ], 0 );

    x[PNN] = PNN_BDRY(C( xr[ IDX1D3(i+1,j  ,k  ,nx,ny,nz) ], 0 ));
    
    x[MPN] = MPN_BDRY(C( xr[ IDX1D3(i-1,j+1,k  ,nx,ny,nz) ], 0 ));
    x[NPN] = NPN_BDRY(C( xr[ IDX1D3(i  ,j+1,k  ,nx,ny,nz) ], 0 ));
    x[PPN] = PPN_BDRY(C( xr[ IDX1D3(i+1,j+1,k  ,nx,ny,nz) ], 0 ));
    
    x[MMP] = MMP_BDRY(C( xr[ IDX1D3(i-1,j-1,k+1,nx,ny,nz) ], 0 ));
    x[NMP] = NMP_BDRY(C( xr[ IDX1D3(i  ,j-1,k+1,nx,ny,nz) ], 0 ));
    x[PMP] = PMP_BDRY(C( xr[ IDX1D3(i+1,j-1,k+1,nx,ny,nz) ], 0 ));
    
    x[MNP] = MNP_BDRY(C( xr[ IDX1D3(i-1,j  ,k+1,nx,ny,nz) ], 0 ));
    x[NNP] = NNP_BDRY(C( xr[ IDX1D3(i  ,j  ,k+1,nx,ny,nz) ], 0 ));
    x[PNP] = PNP_BDRY(C( xr[ IDX1D3(i+1,j  ,k+1,nx,ny,nz) ], 0 ));
    
    x[MPP] = MPP_BDRY(C( xr[ IDX1D3(i-1,j+1,k+1,nx,ny,nz) ], 0 ));
    x[NPP] = NPP_BDRY(C( xr[ IDX1D3(i  ,j+1,k+1,nx,ny,nz) ], 0 ));
    x[PPP] = PPP_BDRY(C( xr[ IDX1D3(i+1,j+1,k+1,nx,ny,nz) ], 0 ));
}


/**
   Pml functions
 */
template <class T>
inline T gamma_func_lower( T x, T nx, T npml_bot)
{
    return cos(PI<T>*((x)*nx/(2*(nx+1)*npml_bot)));
}
template <class T>
inline T gamma_func_upper( T x, T nx, T npml_top)
{
    return cos(PI<T>*((1-(x)/(nx+1)) * nx/(2*npml_top)));
}


template <class T, pml_t p>
class pml_func {
    using C = complex <T>;   
public:
    pml_func(int npts,int npml_t, int npml_b): nz(npts),npml_bot(npml_b),npml_top(npml_t){iz = -1;};
    template <std::size_t N>
    void operator()(int ix, std::array<C,N> & values) {
        constexpr int P = (N-1)/2;
        for (int i=-P; i<=P; i++)
        {
            values[i+P] = (ix+i >=0 && ix+i<nz) ? pml_func_single(ix+i) : 0.0+0.0*1i;
        }	    
    };
    C pml_func_single(int iz){
        T gammaP, gammaN, gammaM;
        // Lower part of the interval, 0, 1, ..., npml_bot-1
        if( iz+1 < npml_bot ){            
            gammaM = gamma_func_lower<T>(iz,nz,npml_bot);
            gammaN = gamma_func_lower<T>(iz+1,nz,npml_bot);
            gammaP = (iz+2 < npml_bot) ? gamma_func_lower<T>(iz+2,nz,npml_bot) : 0;
        }
        else{
            // Interior of the interval, npml_bot, npml_bot+1,...,nz-npml_top-1
            if( (iz+1 >= npml_bot) && (iz+1 < nz+2-npml_top) )
            {
                gammaP = (iz+2 >= nz+2-npml_top) ? gamma_func_upper<T>(iz+2,nz,npml_top) : 0;
                gammaN = 0;
                gammaM = (iz < npml_bot) ? gamma_func_lower<T>(iz,nz,npml_top) : 0;
            }
            else
            {
            //End of the interval, nz-npml_top,...,nz-1
                gammaP = gamma_func_upper<T>(iz+2,nz,npml_top);
                gammaN = gamma_func_upper<T>(iz+1,nz,npml_top);
                gammaM = (iz < nz+2-npml_top) ? 0 : gamma_func_upper<T>(iz,nz,npml_top);
            }
        }
        return  (p==pml_t::UPPER) 
            ? (2.0+0.0*1i)/(2.0-gammaN*(gammaN+gammaP)+(3.0*gammaN+gammaP)*1i)
            : (2.0+0.0*1i)/((2.0-gammaN*(gammaN+gammaM))+(3.0*gammaN+gammaM)*1i);
    };
	   
private:
    C values[3];
    int iz;
    int nz;
    int npml_bot;
    int npml_top;
};


//Constants associated to the stencil
template <class T>
struct coef_consts {
    const T wn_coef; 
    const T wn_xcoef;
    const T wn_ycoef;
    const T wn_zcoef;

    const T pmlx_coef;
    const T pmly_coef;
    const T pmlz_coef;

    const T xz_coef;
    const T xy_coef;
    const T yz_coef;

    const T W3A_2;    
};

/*
  Compute constants associated to stencil computations
 */
template <class T>
inline coef_consts<T> compute_coef_consts(const T * h)
{
    T hx = h[0]; T hy = h[1]; T hz = h[2]; 
    T hx2 = hx*hx, hy2 = hy*hy, hz2 = hz*hz;
    T hxy = hx2 +hy2; 
    T hxz = hx2 + hz2;
    T hyz = hy2 + hz2;
    T hxyz = hx2 + hy2 + hz2;

    T W3A = (W3<T>)*3/(4*hxyz);
    T W3A_2 = 2*W3A;
    T wn_coef = -(W1<T> + 3*W2<T> + 16*W3A*hxyz/3 + WM1<T>-1);
    T wn_xcoef = (W1<T>/hx2 + W2<T>/hx2 + W2<T>/hxz + W2<T>/hxy + 8*W3A);
    T wn_ycoef = (W1<T>/hy2 + W2<T>/hy2 + W2<T>/hyz + W2<T>/hxy + 8*W3A);
    T wn_zcoef = (W1<T>/hz2 + W2<T>/hz2 + W2<T>/hxz + W2<T>/hyz + 8*W3A);
    T pmlx_coef = -(W1<T>/hx2 + W2<T>/hx2 + W2<T>/hxz + W2<T>/hxy + 8*W3A);
    T pmly_coef = -(W1<T>/hy2 + W2<T>/hy2 + W2<T>/hyz + W2<T>/hxy + 8*W3A);
    T pmlz_coef = -(W1<T>/hz2 + W2<T>/hz2 + W2<T>/hxz + W2<T>/hyz + 8*W3A);
    T xz_coef   = W2<T>/(2*hxz);
    T xy_coef   = W2<T>/(2*hxy);
    T yz_coef   = W2<T>/(2*hyz);
    
    coef_consts<T> c = {
        wn_coef,wn_xcoef,wn_ycoef,wn_zcoef,
        pmlx_coef,pmly_coef,pmlz_coef
        ,xz_coef,xy_coef,yz_coef,
        W3A_2   
    };
        
    return c;
}

template <class T,Mult_Mode m, Deriv_Mode d>
inline void compute_coefs(Cbuf_3d<T> & coef, const Cbuf_3d<T> wn_window, const coef_consts<T> c, const pml_info<T> p)
{
    constexpr int M = static_cast<int>(offset_idx::M);
    constexpr int N = static_cast<int>(offset_idx::N);
    constexpr int P = static_cast<int>(offset_idx::P);
    constexpr T deriv_flag = (d!=DERIV_MODE) ? 1 : 0;
    constexpr T w2 = WM2<T>, w3 = WM3<T>, w4 = WM4<T>;
    
    
    if(m==FORW_MULT){
        /* Compute coefficients - forward mode */
        coef[MMM] = MMM_BDRY(- w4*wn_window[MMM] + deriv_flag*(-c.W3A_2*( p.pxlo_buf[N] + p.pylo_buf[N] + p.pzlo_buf[N] )));
        coef[NMM] = NMM_BDRY(- w3*wn_window[NMM] + deriv_flag*(-c.yz_coef * (p.pzlo_buf[N] + p.pylo_buf[N]) + c.W3A_2*p.px_buf[N]));
        coef[PMM] = PMM_BDRY(- w4*wn_window[PMM] + deriv_flag*(-c.W3A_2*( p.pxhi_buf[N] + p.pylo_buf[N] + p.pzlo_buf[N] )));
        
        coef[MNM] = MNM_BDRY(- w3*wn_window[MNM] + deriv_flag*(-c.xz_coef * (p.pzlo_buf[N] + p.pxlo_buf[N]) + c.W3A_2*p.py_buf[N]));
        coef[NNM] = NNM_BDRY(- w2*wn_window[NNM] + deriv_flag*(c.pmlz_coef*p.pzlo_buf[N] + c.yz_coef*p.py_buf[N] + c.xz_coef*p.px_buf[N]));
        coef[PNM] = PNM_BDRY(- w3*wn_window[PNM] + deriv_flag*(-c.xz_coef * (p.pzlo_buf[N] + p.pxhi_buf[N]) + c.W3A_2*p.py_buf[N]));
        
        coef[MPM] = MPM_BDRY(- w4*wn_window[MPM] + deriv_flag*(-c.W3A_2*( p.pxlo_buf[N] + p.pyhi_buf[N] + p.pzlo_buf[N] )));
        coef[NPM] = NPM_BDRY(- w3*wn_window[NPM] + deriv_flag*(-c.yz_coef * (p.pyhi_buf[N] + p.pzlo_buf[N]) + c.W3A_2*p.px_buf[N]));
        coef[PPM] = PPM_BDRY(- w4*wn_window[PPM] + deriv_flag*(-c.W3A_2*( p.pxhi_buf[N] + p.pyhi_buf[N] + p.pzlo_buf[N] )));		
        
        coef[MMN] = MMN_BDRY(- w3*wn_window[MMN] + deriv_flag*(-c.xy_coef * (p.pxlo_buf[N] + p.pylo_buf[N]) + c.W3A_2*p.pz_buf[N]));
        coef[NMN] = NMN_BDRY(- w2*wn_window[NMN] + deriv_flag*(c.pmly_coef*p.pylo_buf[N] + c.yz_coef*p.pz_buf[N] + c.xy_coef*p.px_buf[N]));	
        coef[PMN] = PMN_BDRY(- w3*wn_window[PMN] + deriv_flag*(-c.xy_coef * (p.pxhi_buf[N] + p.pylo_buf[N]) + c.W3A_2*p.pz_buf[N]));	
        
        coef[MNN] = MNN_BDRY(- w2*wn_window[MNN] + deriv_flag*(c.pmlx_coef*p.pxlo_buf[N] + c.xz_coef*p.pz_buf[N] + c.xy_coef*p.py_buf[N]));
        coef[NNN] = c.wn_coef*wn_window[NNN] + deriv_flag*(c.wn_xcoef*p.px_buf[N] + c.wn_ycoef*p.py_buf[N] + c.wn_zcoef*p.pz_buf[N]);
        coef[PNN] = PNN_BDRY(- w2*wn_window[PNN] + deriv_flag*(c.pmlx_coef*p.pxhi_buf[N] + c.xz_coef*p.pz_buf[N] + c.xy_coef*p.py_buf[N]));
        
        coef[MPN] = MPN_BDRY(- w3*wn_window[MPN] + deriv_flag*(-c.xy_coef * (p.pxlo_buf[N] + p.pyhi_buf[N]) + c.W3A_2*p.pz_buf[N]));		
        coef[NPN] = NPN_BDRY(- w2*wn_window[NPN] + deriv_flag*(c.pmly_coef*p.pyhi_buf[N] + c.yz_coef*p.pz_buf[N] + c.xy_coef*p.px_buf[N]));
        coef[PPN] = PPN_BDRY(- w3*wn_window[PPN] + deriv_flag*(-c.xy_coef * (p.pxhi_buf[N] + p.pyhi_buf[N]) + c.W3A_2*p.pz_buf[N]));
        
        coef[MMP] = MMP_BDRY(- w4*wn_window[MMP] + deriv_flag*(-c.W3A_2*( p.pxlo_buf[N] + p.pylo_buf[N] + p.pzhi_buf[N] )));
        coef[NMP] = NMP_BDRY(- w3*wn_window[NMP] + deriv_flag*(-c.yz_coef * (p.pylo_buf[N] + p.pzhi_buf[N]) + c.W3A_2*p.px_buf[N]));
        coef[PMP] = PMP_BDRY(- w4*wn_window[PMP] + deriv_flag*(-c.W3A_2*( p.pxhi_buf[N] + p.pylo_buf[N] + p.pzhi_buf[N] )));
        
        coef[MNP] = MNP_BDRY(- w3*wn_window[MNP] + deriv_flag*(-c.xz_coef * (p.pzhi_buf[N] + p.pxlo_buf[N]) + c.W3A_2*p.py_buf[N]));
        coef[NNP] = NNP_BDRY(- w2*wn_window[NNP] + deriv_flag*(c.pmlz_coef*p.pzhi_buf[N] + c.yz_coef*p.py_buf[N] + c.xz_coef*p.px_buf[N]));
        coef[PNP] = PNP_BDRY(- w3*wn_window[PNP] + deriv_flag*(-c.xz_coef * (p.pzhi_buf[N] + p.pxhi_buf[N]) + c.W3A_2*p.py_buf[N]));
        
        coef[MPP] = MPP_BDRY(- w4*wn_window[MPP] + deriv_flag*(-c.W3A_2*( p.pxlo_buf[N] + p.pyhi_buf[N] + p.pzhi_buf[N] )));
        coef[NPP] = NPP_BDRY(- w3*wn_window[NPP] + deriv_flag*(-c.yz_coef * (p.pzhi_buf[N] + p.pyhi_buf[N]) + c.W3A_2*p.px_buf[N]));
        coef[PPP] = PPP_BDRY(- w4*wn_window[PPP] + deriv_flag*(-c.W3A_2*( p.pxhi_buf[N] + p.pyhi_buf[N] + p.pzhi_buf[N] )));
    }
    else{        
        coef[PPP] = - w4*wn_window[NNN] + deriv_flag*(-c.W3A_2*( p.pxlo_buf[P] + p.pylo_buf[P] + p.pzlo_buf[P] ));
        coef[NPP] = - w3*wn_window[NNN] + deriv_flag*(-c.yz_coef * (p.pzlo_buf[P] + p.pylo_buf[P]) + c.W3A_2*p.px_buf[N]);
        coef[MPP] = - w4*wn_window[NNN] + deriv_flag*(-c.W3A_2*( p.pxhi_buf[M] + p.pylo_buf[P] + p.pzlo_buf[P] ));
        
        coef[PNP] = - w3*wn_window[NNN] + deriv_flag*(-c.xz_coef * (p.pzlo_buf[P] + p.pxlo_buf[P]) + c.W3A_2*p.py_buf[N]);
        coef[NNP] = - w2*wn_window[NNN] + deriv_flag*(c.pmlz_coef*p.pzlo_buf[P] + c.yz_coef*p.py_buf[N] + c.xz_coef*p.px_buf[N]);
        coef[MNP] = - w3*wn_window[NNN] + deriv_flag*(-c.xz_coef * (p.pzlo_buf[P] + p.pxhi_buf[M]) + c.W3A_2*p.py_buf[N]);
        
        coef[PMP] = - w4*wn_window[NNN] + deriv_flag*(-c.W3A_2*( p.pxlo_buf[P] + p.pyhi_buf[M] + p.pzlo_buf[P] ));
        coef[NMP] = - w3*wn_window[NNN] + deriv_flag*(-c.yz_coef * (p.pyhi_buf[M] + p.pzlo_buf[P]) + c.W3A_2*p.px_buf[N]);
        coef[MMP] = - w4*wn_window[NNN] + deriv_flag*(-c.W3A_2*( p.pxhi_buf[M] + p.pyhi_buf[M] + p.pzlo_buf[P] ));		
        
        coef[PPN] = - w3*wn_window[NNN] + deriv_flag*(-c.xy_coef * (p.pxlo_buf[P] + p.pylo_buf[P]) + c.W3A_2*p.pz_buf[N]);
        coef[NPN] = - w2*wn_window[NNN] + deriv_flag*(c.pmly_coef*p.pylo_buf[P] + c.yz_coef*p.pz_buf[N] + c.xy_coef*p.px_buf[N]);	
        coef[MPN] = - w3*wn_window[NNN] + deriv_flag*(-c.xy_coef * (p.pxhi_buf[M] + p.pylo_buf[P]) + c.W3A_2*p.pz_buf[N]);	
        
        coef[PNN] = - w2*wn_window[NNN] + deriv_flag*(c.pmlx_coef*p.pxlo_buf[P] + c.xz_coef*p.pz_buf[N] + c.xy_coef*p.py_buf[N]);
        coef[NNN] = c.wn_coef*wn_window[NNN] + deriv_flag*(c.wn_xcoef*p.px_buf[N] + c.wn_ycoef*p.py_buf[N] + c.wn_zcoef*p.pz_buf[N]);
        coef[MNN] = - w2*wn_window[NNN] + deriv_flag*(c.pmlx_coef*p.pxhi_buf[M] + c.xz_coef*p.pz_buf[N] + c.xy_coef*p.py_buf[N]);
        
        coef[PMN] = - w3*wn_window[NNN] + deriv_flag*(-c.xy_coef * (p.pxlo_buf[P] + p.pyhi_buf[M]) + c.W3A_2*p.pz_buf[N]);		
        coef[NMN] = - w2*wn_window[NNN] + deriv_flag*(c.pmly_coef*p.pyhi_buf[M] + c.yz_coef*p.pz_buf[N] + c.xy_coef*p.px_buf[N]);
        coef[MMN] = - w3*wn_window[NNN] + deriv_flag*(-c.xy_coef * (p.pxhi_buf[M] + p.pyhi_buf[M]) + c.W3A_2*p.pz_buf[N]);
        
        coef[PPM] = - w4*wn_window[NNN] + deriv_flag*(-c.W3A_2*( p.pxlo_buf[P] + p.pylo_buf[P] + p.pzhi_buf[M] ));
        coef[NPM] = - w3*wn_window[NNN] + deriv_flag*(-c.yz_coef * (p.pylo_buf[P] + p.pzhi_buf[M]) + c.W3A_2*p.px_buf[N]);
        coef[MPM] = - w4*wn_window[NNN] + deriv_flag*(-c.W3A_2*( p.pxhi_buf[M] + p.pylo_buf[P] + p.pzhi_buf[M] ));
        
        coef[PNM] = - w3*wn_window[NNN] + deriv_flag*(-c.xz_coef * (p.pzhi_buf[M] + p.pxlo_buf[P]) + c.W3A_2*p.py_buf[N]);
        coef[NNM] = - w2*wn_window[NNN] + deriv_flag*(c.pmlz_coef*p.pzhi_buf[M] + c.yz_coef*p.py_buf[N] + c.xz_coef*p.px_buf[N]);
        coef[MNM] = - w3*wn_window[NNN] + deriv_flag*(-c.xz_coef * (p.pzhi_buf[M] + p.pxhi_buf[M]) + c.W3A_2*p.py_buf[N]);
        
        coef[PMM] = - w4*wn_window[NNN] + deriv_flag*(-c.W3A_2*( p.pxlo_buf[P] + p.pyhi_buf[M] + p.pzhi_buf[M] ));
        coef[NMM] = - w3*wn_window[NNN] + deriv_flag*(-c.yz_coef * (p.pzhi_buf[M] + p.pyhi_buf[M]) + c.W3A_2*p.px_buf[N]);
        coef[MMM] = - w4*wn_window[NNN] + deriv_flag*(-c.W3A_2*( p.pxhi_buf[M] + p.pyhi_buf[M] + p.pzhi_buf[M] ));
        for(int t=0; t<27; t++)
        {
            coef[t] = conj(coef[t]);
        }
    }
  
}


template <class T,Mult_Mode m, Deriv_Mode d, Wavenum_Cmplx c>
void do_Hmvp( const T * wnr, const T * wni, const T * h, const T * n, const T * npml, T * yr, T *yi, const T * xr, const T * xi, int zmin, int zmax) {
    using PMLlo = pml_func<T,pml_t::LOWER>;    
    using PMLhi = pml_func<T,pml_t::UPPER>;
    // Counter/index variables
    int i,j,k,t,kout;
    
    int nx = (int)n[0]; int ny = (int)n[1]; int nz = (int)n[2];
    
    int npmlx_lo = (int)npml[0]; int npmlx_hi = (int)npml[1]; 
    int npmly_lo = (int)npml[2]; int npmly_hi = (int)npml[3]; 
    int npmlz_lo = (int)npml[4]; int npmlz_hi = (int)npml[5];    

    PMLlo pzlo = PMLlo(nz,npmlz_hi,npmlz_lo);
    PMLhi pzhi = PMLhi(nz,npmlz_hi,npmlz_lo);
    PMLlo pylo = PMLlo(ny,npmly_hi,npmly_lo);
    PMLhi pyhi = PMLhi(ny,npmly_hi,npmly_lo);
    PMLlo pxlo = PMLlo(nx,npmlx_hi,npmlx_lo);
    PMLhi pxhi = PMLhi(nx,npmlx_hi,npmlx_lo);

    coef_consts<T> consts = compute_coef_consts(h);
    
    pml_info<T> p;

    Cbuf_3d<T> wn_window,x_window,coef;
    C<T> y_out;    

    for(k=zmin; k<zmax; k++)
    {
        pzlo(k,p.pzlo_buf); pzhi(k,p.pzhi_buf);
        for(t=0;t<3;t++){ p.pz_buf[t] = p.pzlo_buf[t] + p.pzhi_buf[t]; }
        p.z_hasL = k>0; p.z_hasR = k<nz-1;
        for(j=0; j<ny; j++)
        {
            pylo(j,p.pylo_buf); pyhi(j,p.pyhi_buf);
            for(t=0;t<3;t++){ p.py_buf[t] = p.pylo_buf[t] + p.pyhi_buf[t]; }
            p.y_hasL = j>0; p.y_hasR = j<ny-1;
            for(i=0; i<nx; i++)
            {
                 pxlo(i,p.pxlo_buf); pxhi(i,p.pxhi_buf);
                 for(t=0;t<3;t++){ p.px_buf[t] = p.pxlo_buf[t] + p.pxhi_buf[t]; }
                 p.x_hasL = i>0; p.x_hasR = i<nx-1;
                 //Load wavenumber window
                 if(c==WN_IS_REAL)
                     load_nbrhoodr( wn_window, wnr, i,j,k,nx,ny,nz,p );
                 else
                     load_nbrhoodc( wn_window, wnr,wni,i,j,k,nx,ny,nz,p );

                 //Compute coefficients                 
                 compute_coefs<T,m,d>(coef,wn_window,consts,p);
                 //Load wavefield window
                 load_nbrhoodc(x_window,xr,xi,i,j,k,nx,ny,nz,p);
               
                 y_out = 0.0+0.0*1i;
                 kout = IDX1D3(i,j,k,nx,ny,nz);
                 for(t=0;t<27; t++){ y_out += coef[t]*x_window[t]; }
                 yr[kout] = real(y_out);
                 yi[kout] = imag(y_out);                 
            }
        }
    }

}
