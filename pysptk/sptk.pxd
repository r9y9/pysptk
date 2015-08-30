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

    # library routines
    double agexp(double r, double x, double y)
    double gexp(const double r, const double x)
    double glog(const double r, const double x)
    int mseq()

    # f0
    void swipe(double *input, double *output, int length, int samplerate,
               int frame_shift, double min, double max, double st, int otype)

    # window functions
    double window(Window window_type, double *x, const int size, const int nflg)

    # mel-generalized cepstrums
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

    # Conversions
    void gnorm(double *c1, double *c2, int m, const double g)
    void ignorm(double *c1, double *c2, int m, const double g)

    void b2mc(double *b, double *mc, int m, const double a)
    void mc2b(double *mc, double *b, int m, const double a)
