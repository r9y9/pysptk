### Definitions ###

cdef enum Window:
    BLACKMAN = 0
    HAMMING = 1
    HANNING = 2
    BARTLETT = 3
    TRAPEZOID = 4
    RECTANGULAR = 5

cdef enum Boolean:
    FA = 0
    TR = 1

cdef extern from "SPTK.h":

    # Library routines
    double _agexp "agexp"(double r, double x, double y)
    double _gexp "gexp"(const double r, const double x)
    double _glog "glog"(const double r, const double x)
    int _mseq "mseq"()


    # Adaptive mel-generalized cepstrum analysis
    double _acep "acep"(double x, double *c, const int m, const double lambda_coef,
                        const double step, const double tau, const int pd,
                        const double eps);
    double _agcep "agcep"(double x, double *c, const int m, const int stage,
                          const double lambda_coef, const double step, const double tau,
                          const double eps);
    double _amcep "amcep"(double x, double *b, const int m, const double a,
                          const double lambda_coef, const double step, const double tau,
                          const int pd, const double eps);


    # Mel-generalized cepstrum analysis
    int _mcep "mcep"(double *xw, const int flng, double *mc, const int m, const double a,
                     const int itr1, const int itr2, const double dd, const int etype,
                     const double e, const double f, const int itype)
    int _gcep "gcep"(double *xw, const int flng, double *gc, const int m, const double g,
                     const int itr1, const int itr2, const double d, const int etype,
                     const double e, const double f, const int itype)
    int _mgcep "mgcep"(double *xw, int flng, double *b, const int m, const double a,
                       const double g, const int n, const int itr1, const int itr2,
                       const double dd, const int etype, const double e, const double f,
                       const int itype)
    int _uels "uels"(double *xw, const int flng, double *c, const int m, const int itr1,
                     const int itr2, const double dd, const int etype, const double e,
                     const int itype)
    void _fftcep "fftcep"(double *sp, const int flng, double *c, const int m, int itr,
                          double ac)
    int _lpc "lpc"(double *x, const int flng, double *a, const int m, const double f)


    # MFCC
    void _mfcc "mfcc"(double *in_mfcc, double *mc, const double sampleFreq,
                      const double alpha, const double eps, const int wlng,
                      const int flng, const int m, const int n, const int ceplift,
                      const Boolean dftmode, const Boolean usehamming)

    # LPC, LSP and PARCOR conversions
    void _lpc2c "lpc2c"(double *a, int m1, double *c, const int m2)
    int _lpc2lsp "lpc2lsp"(double *lpc, double *lsp, const int order, const int numsp,
                           const int maxitr, const double eps)
    int _lpc2par "lpc2par"(double *a, double *k, const int m)
    void _par2lpc "par2lpc"(double *k, double *a, const int m)
    void _lsp2sp "lsp2sp"(double *lsp, const int m, double *x, const int l, const int gain)


    # Mel-generalized cepstrum conversions
    void _mc2b "mc2b"(double *mc, double *b, int m, const double a)
    void _b2mc "b2mc"(double *b, double *mc, int m, const double a)
    void _b2c "b2c"(double *b, int m1, double *c, int m2, double a)
    void _c2acr "c2acr"(double *c, const int m1, double *r, const int m2, const int flng)
    void _c2ir "c2ir"(double *c, const int nc, double *h, const int leng)
    void _ic2ir "ic2ir"(double *h, const int leng, double *c, const int nc)
    void _c2ndps "c2ndps"(double *c, const int m, double *n, const int l)
    void _ndps2c "ndps2c"(double *n, const int l, double *c, const int m)
    void _gc2gc "gc2gc"(double *c1, const int m1, const double g1, double *c2, const int m2,
                        const double g2)
    void _gnorm "gnorm"(double *c1, double *c2, int m, const double g)
    void _ignorm "ignorm"(double *c1, double *c2, int m, const double g)
    void _freqt "freqt"(double *c1, const int m1, double *c2, const int m2, const double a)
    void _frqtr "frqtr"(double *c1, int m1, double *c2, int m2, const double a)
    void _mgc2mgc "mgc2mgc"(double *c1, const int m1, const double a1, const double g1,
                            double *c2, const int m2, const double a2, const double g2)
    void _mgc2sp "mgc2sp"(double *mgc, const int m, const double a, const double g, double *x,
                          double *y, const int flng)
    void _mgclsp2sp "mgclsp2sp"(double a, double g, double *lsp, const int m, double *x,
                                const int l, const int gain)


    # F0 analysis
    void _swipe "swipe"(double *input, double *output, int length, int samplerate,
                        int frame_shift, double min, double max, double st, int otype)
    int _rapt "rapt"(float * input, float * output, int length, double sample_freq,
                     int frame_shift, double minF0, double maxF0, double voice_bias,
                     int otype)


    # Excitation
    void _excite "excite"(double *pitch, int n, double *out, int fprd, int iprd, Boolean gauss, int seed_i)

    # Waveform generation filters
    double _poledf "poledf"(double x, double *a, int m, double *d)
    double _lmadf "lmadf"(double x, double *c, const int m, const int pd, double *d)
    double _lspdf_even "lspdf_even"(double x, double *f, const int m, double *d)
    double _lspdf_odd "lspdf_odd"(double x, double *f, const int m, double *d)
    double _ltcdf "ltcdf"(double x, double *k, int m, double *d)
    double _glsadf "glsadf"(double x, double *c, const int m, const int n, double *d)
    double _mlsadf "mlsadf"(double x, double *b, const int m, const double a, const int pd,
                            double *d)
    double _mglsadf "mglsadf"(double x, double *b, const int m, const double a, const int n,
                              double *d)

    # Window functions
    double _window "window"(Window window_type, double *x, const int size, const int nflg)

    # Utils
    int _lspcheck "lspcheck"(double *lsp, const int ord)
    void _phidf "phidf"(const double x, const int m, double a, double *d)
