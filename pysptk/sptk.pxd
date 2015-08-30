### Definitions ###

# TODO: cdef extern from...?
cdef enum Window:
    BLACKMAN = 0
    HAMMING = 1
    HANNING = 2
    BARTLETT = 3
    TRAPEZOID = 4
    RECTANGULAR = 5

cdef extern from "../lib/SPTK/include/SPTK.h":

    # Library routines
    double agexp(double r, double x, double y)
    double gexp(const double r, const double x)
    double glog(const double r, const double x)
    int mseq()


    # Adaptive mel-generalized cepstrum analysis
    double acep(double x, double *c, const int m, const double lambda_coef,
                const double step, const double tau, const int pd,
                const double eps);
    double agcep(double x, double *c, const int m, const int stage,
                 const double lambda_coef, const double step, const double tau,
                 const double eps);
    double amcep(double x, double *b, const int m, const double a,
                 const double lambda_coef, const double step, const double tau,
                 const int pd, const double eps);


    # Mel-generalized cepstrum analysis
    int mcep(double *xw, const int flng, double *mc, const int m, const double a,
             const int itr1, const int itr2, const double dd, const int etype,
             const double e, const double f, const int itype)
    int gcep(double *xw, const int flng, double *gc, const int m, const double g,
             const int itr1, const int itr2, const double d, const int etype,
             const double e, const double f, const int itype)
    int mgcep(double *xw, int flng, double *b, const int m, const double a,
              const double g, const int n, const int itr1, const int itr2,
              const double dd, const int etype, const double e, const double f,
              const int itype)
    int uels(double *xw, const int flng, double *c, const int m, const int itr1,
             const int itr2, const double dd, const int etype, const double e,
             const int itype)
    void fftcep(double *sp, const int flng, double *c, const int m, int itr,
                double ac)
    int lpc(double *x, const int flng, double *a, const int m, const double f)


    # LPC, LSP and PARCOR conversions
    void lpc2c(double *a, int m1, double *c, const int m2)
    int lpc2lsp(double *lpc, double *lsp, const int order, const int numsp,
                const int maxitr, const double eps)
    int lpc2par(double *a, double *k, const int m)
    void par2lpc(double *k, double *a, const int m)
    void lsp2sp(double *lsp, const int m, double *x, const int l, const int gain)


    # Mel-generalized cepstrum conversions
    void mc2b(double *mc, double *b, int m, const double a)
    void b2mc(double *b, double *mc, int m, const double a)
    void b2c(double *b, int m1, double *c, int m2, double a)
    void c2acr(double *c, const int m1, double *r, const int m2, const int flng)
    void c2ir(double *c, const int nc, double *h, const int leng)
    void ic2ir(double *h, const int leng, double *c, const int nc)
    void c2ndps(double *c, const int m, double *n, const int l)
    void ndps2c(double *n, const int l, double *c, const int m)
    void gc2gc(double *c1, const int m1, const double g1, double *c2, const int m2,
               const double g2)
    void gnorm(double *c1, double *c2, int m, const double g)
    void ignorm(double *c1, double *c2, int m, const double g)
    void freqt(double *c1, const int m1, double *c2, const int m2, const double a)
    void frqtr(double *c1, int m1, double *c2, int m2, const double a)
    void mgc2mgc(double *c1, const int m1, const double a1, const double g1,
                 double *c2, const int m2, const double a2, const double g2)
    void mgc2sp(double *mgc, const int m, const double a, const double g, double *x,
                double *y, const int flng)
    void mgclsp2sp(double a, double g, double *lsp, const int m, double *x,
                   const int l, const int gain)


    # F0 analysis
    void swipe(double *input, double *output, int length, int samplerate,
               int frame_shift, double min, double max, double st, int otype)

    # Waveform generation filters
    double poledf(double x, double *a, int m, double *d)
    double lmadf(double x, double *c, const int m, const int pd, double *d)
    double lspdf_even(double x, double *f, const int m, double *d)
    double lspdf_odd(double x, double *f, const int m, double *d)
    double ltcdf(double x, double *k, int m, double *d)
    double glsadf(double x, double *c, const int m, const int n, double *d)
    double mlsadf(double x, double *b, const int m, const double a, const int pd,
                  double *d)
    double mglsadf(double x, double *b, const int m, const double a, const int n,
                   double *d)

    # Window functions
    double window(Window window_type, double *x, const int size, const int nflg)

    # Utils
    int lspcheck(double *lsp, const int ord)
    void phidf(const double x, const int m, double a, double *d)
