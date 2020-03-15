#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cmath>
#include <iostream>
#include <string>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <cstdio>
#include <mkl.h>

// used to initialize numpy, otherwise it fails.
int init_(){
    mkl_set_dynamic(0);
    mkl_set_num_threads(1);
    import_array(); // PyError if not successful
    return 0;
}

const static double x_max = X_MAX, grid_size = GRID_SIZE; 
const static MKL_INT x_n = int(x_max/grid_size+0.5)*2+1;
const static double lambda = LAMBDA, mass = MASS;
const static int derivative_estimation_points = 4; // this parameter should influence the dimension and contents of matrix ab and p_hat, Hamiltonian etc.

const static struct matrix_descr descr = {SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT}; // notice that first order derivative and p_hat do not satisfy
static MKL_Complex16 ab[2*derivative_estimation_points+1][x_n]={{{0.,0.}}}, ab_LU[3*derivative_estimation_points+1][x_n]={{{0.,0.}}}; MKL_INT LU_ipiv[x_n];

class Set_World{
public:
    double x[x_n], x_2[x_n], quartic_V[x_n]; sparse_matrix_t identity;
    sparse_matrix_t delta_x, delta_2_x;
    sparse_matrix_t x_hat, p_hat, p_hat_2, xp_px_hat;
    sparse_matrix_t quartic_Hamil; // these two involve complex numbers
    // the following are substituted into imaginary parts of matrix ab
    double ab_upper[derivative_estimation_points][x_n]={{0.}}, ab_lower[derivative_estimation_points][x_n]={{0.}},\
           ab_center[x_n]={0.};

    int init_status = 0;

    sparse_matrix_t empty_matrix; // the "empty_matrix" is for simpler calculation writing
    MKL_INT* empty_rowindex_start, *empty_rowindex_end, *empty_column; // *** Somehow MKL does not allow copying and automatically allocating such a matrix,
    MKL_Complex16* empty_value;                                        // and we have to manage its memory manually.

    ///////////////////////////// *** initializer *** ///////////////////////////////////////////

    Set_World(){
    // compute the basic  x, x^2, V=\lambda*x^4  arrays
    for(int i=0;i < x_n;i++){x[i]=grid_size*((double)(i-int(x_max/grid_size+0.5)));}
    for(int i=0;i < x_n;i++){x_2[i]=x[i]*x[i];}
    for(int i=0;i < x_n;i++){quartic_V[i]=x_2[i]*x_2[i]*lambda;}

    // Fortran defaults to one-based indexing with column major format. zero-based indexing implies C format? it seems so.
    // If we simply use ```MKL_Complex16 delta_x_dense[x_n][x_n], delta_2_x_dense[x_n][x_n];```, it will result in a stack overflow when x_n is too large
    using arr2d = MKL_Complex16(*)[x_n];
    arr2d delta_x_dense = new MKL_Complex16[x_n][x_n]; arr2d delta_2_x_dense = new MKL_Complex16[x_n][x_n];
    //MKL_Complex16 delta_x_dense[x_n][x_n], delta_2_x_dense[x_n][x_n];
    for(int i=0;i<x_n;i++){for(int j=0;j<x_n;j++){delta_x_dense[i][j]={0.,0.};delta_2_x_dense[i][j]={0.,0.};}}

    for(int i=1;i < x_n-1;i++){
        delta_x_dense[i][i-1].real=-672./840. / grid_size;
        delta_x_dense[i-1][i].real=672./840. / grid_size;}
    for(int i=2;i < x_n-2;i++){
        delta_x_dense[i][i-2].real=168./840. / grid_size;
        delta_x_dense[i-2][i].real=-168./840. / grid_size;}
    for(int i=3;i < x_n-3;i++){
        delta_x_dense[i][i-3].real=-32./840. / grid_size;
        delta_x_dense[i-3][i].real=32./840. / grid_size;}
    for(int i=4;i < x_n-4;i++){
        delta_x_dense[i][i-4].real=3./840. / grid_size;
        delta_x_dense[i-4][i].real=-3./840. / grid_size;}//
    for(int i=0;i < x_n;i++){
        delta_2_x_dense[i][i].real=-14350./5040. / (grid_size*grid_size);
        ab_center[i]=(delta_2_x_dense[i][i].real*(-1.)/(2.*mass)+quartic_V[i])*0.5;}
    for(int i=1;i < x_n;i++){
        delta_2_x_dense[i][i-1].real=8064./5040. / (grid_size*grid_size);
        delta_2_x_dense[i-1][i].real=8064./5040. / (grid_size*grid_size);
        ab_upper[3][i]=0.5*delta_2_x_dense[i][i-1].real*(-1.)/(2.*mass);
        ab_lower[0][i-1]=0.5*delta_2_x_dense[i][i-1].real*(-1.)/(2.*mass);}
    for(int i=2;i < x_n;i++){
        delta_2_x_dense[i][i-2].real=-1008./5040. / (grid_size*grid_size);
        delta_2_x_dense[i-2][i].real=-1008./5040. / (grid_size*grid_size);
        ab_upper[2][i]=0.5*delta_2_x_dense[i][i-2].real*(-1.)/(2.*mass);
        ab_lower[1][i-2]=0.5*delta_2_x_dense[i][i-2].real*(-1.)/(2.*mass);}
    for(int i=3;i < x_n;i++){
        delta_2_x_dense[i][i-3].real=128./5040. / (grid_size*grid_size);
        delta_2_x_dense[i-3][i].real=128./5040. / (grid_size*grid_size);
        ab_upper[1][i]=0.5*delta_2_x_dense[i][i-3].real*(-1.)/(2.*mass);
        ab_lower[2][i-3]=0.5*delta_2_x_dense[i][i-3].real*(-1.)/(2.*mass);}
    for(int i=4;i < x_n;i++){
        delta_2_x_dense[i][i-4].real=-9./5040. / (grid_size*grid_size);
        delta_2_x_dense[i-4][i].real=-9./5040. / (grid_size*grid_size);
        ab_upper[0][i]=0.5*delta_2_x_dense[i][i-4].real*(-1.)/(2.*mass);
        ab_lower[3][i-4]=0.5*delta_2_x_dense[i][i-4].real*(-1.)/(2.*mass);}
    /* if we add lower order derivative estimations around the borders,
       it will become not hermitian */
    for(int i=0;i < x_n;i++){ab[derivative_estimation_points][i].real=1.;}

    // create an empty sparse matrix for convenience
    empty_rowindex_start = (MKL_INT*)mkl_malloc(sizeof(MKL_INT)*x_n, 64);
    empty_rowindex_end =   (MKL_INT*)mkl_malloc(sizeof(MKL_INT)*x_n, 64);
    empty_column =         (MKL_INT*)mkl_malloc(sizeof(MKL_INT)*1      , 64);
    empty_value =    (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16)*1, 64);
    for(int i=0;i<x_n;i++){
        empty_rowindex_start[i]=1; empty_rowindex_end[i]=1;
    }
    empty_rowindex_start[0]=0; empty_column[0]=0; empty_value[0]={0.,0.};

    if (SPARSE_STATUS_SUCCESS!= mkl_sparse_z_create_csr (&empty_matrix, SPARSE_INDEX_BASE_ZERO, x_n, x_n, empty_rowindex_start, empty_rowindex_end, empty_column, empty_value)&& \
         init_status==0){init_status = -1;}

    // prepare for mkl_?dnscsr(...), which transforms a dense matrix to a sparse
    MKL_INT job[6]={0,0,0,2,(2*derivative_estimation_points+1)*x_n,3}; MKL_INT info;
    MKL_Complex16 acsr_1[(2*derivative_estimation_points+1)*x_n]={{0.,0.}};
    MKL_INT ja_1[(2*derivative_estimation_points+1)*x_n]={0}, ia_1[x_n+1]={0};
    /* dense -> CSR, zero-based indexing in CSR, and dense,
       adns is a whole matrix, max non-zero values, require all outputs */

    mkl_zdnscsr (job, &x_n, &x_n, &(delta_x_dense[0][0]) , &x_n , acsr_1 , ja_1 , ia_1 , &info );
    if(info!=0)printf("CSR conversion-1 error info %d",(int)info);
    // create delta_x operator
    sparse_matrix_t temp;
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_z_create_csr (&temp, SPARSE_INDEX_BASE_ZERO, x_n, x_n, ia_1, ia_1+1, ja_1, acsr_1) && init_status==0){init_status = -2;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_copy (temp, {SPARSE_MATRIX_TYPE_GENERAL,SPARSE_FILL_MODE_UPPER,SPARSE_DIAG_NON_UNIT}, &delta_x) && init_status==0){
        init_status = -2;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_destroy (temp) && init_status==0){init_status = -2;}
    delete[] delta_x_dense;
    // store the CSR format of delta_2_x_dense
    MKL_Complex16 acsr_2[(2*derivative_estimation_points+1)*x_n]={{0.,0.}};
    MKL_INT ja_2[(2*derivative_estimation_points+1)*x_n]={0}, ia_2[x_n+1]={0};
    mkl_zdnscsr (job , &x_n , &x_n , delta_2_x_dense[0] , &x_n , acsr_2 , ja_2 , ia_2 , &info );
    if(info!=0)printf("CSR conversion-2 error info %d",(int)info);
    // create delta_2_x
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_z_create_csr (&temp, SPARSE_INDEX_BASE_ZERO, x_n, x_n, ia_2, ia_2+1, ja_2, acsr_2) && init_status==0){init_status = -3;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_copy (temp, {SPARSE_MATRIX_TYPE_SYMMETRIC,SPARSE_FILL_MODE_UPPER,SPARSE_DIAG_NON_UNIT}, &delta_2_x) && init_status==0){
        init_status = -3;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_destroy (temp) && init_status==0){init_status = -3;}
    delete[] delta_2_x_dense;
    // create x_hat CSR matrix
    // prepare for mkl_?csrdia(...)
    job[0]=1, job[3]=0, job[4]=0, job[5]=0;
    /* diagonal -> CSR, zero-based indexing in CSR, and diagonal,
       N/A, N/A, check zeroes and leave out them */
    // vairables for the CSR matrix, the "_rem"s are not used. There should be exactly x_n values in total. 
    MKL_Complex16 *acsr_rem=nullptr; MKL_INT *ja_rem=nullptr, *ia_rem=nullptr;
    MKL_Complex16 x_C[x_n]={{0.,0.}};
    for(int i=0;i<x_n;i++){x_C[i].real=x[i];}
    // variables for the diagonal format matrix.
    MKL_INT distance[1]={0}, idiag=1;
    MKL_Complex16 acsr_3[x_n]={{0.,0.}};
    MKL_INT ja_3[x_n]={0}, ia_3[x_n+1]={0};
    // change the format to CSR:
    mkl_zcsrdia (job, &x_n , acsr_3 , ja_3 , ia_3 , x_C , &x_n , distance , &idiag , acsr_rem , ja_rem , ia_rem , &info );
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_z_create_csr (&temp, SPARSE_INDEX_BASE_ZERO, x_n, x_n, ia_3, ia_3+1, ja_3, acsr_3) && init_status==0){init_status = -4;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_copy (temp, {SPARSE_MATRIX_TYPE_DIAGONAL,SPARSE_FILL_MODE_UPPER,SPARSE_DIAG_NON_UNIT}, &x_hat) && init_status==0){
        init_status = -4;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_destroy (temp) && init_status==0){init_status = -4;}

    // create quartic_V CSR matrix
    sparse_matrix_t V_hat;
    MKL_Complex16 quartic_V_C[x_n]={{0.,0.}};
    for(int i=0;i<x_n;i++){quartic_V_C[i].real=quartic_V[i];}
    MKL_Complex16 acsr_4[x_n]={{0.,0.}};
    MKL_INT ja_4[x_n]={0}, ia_4[x_n+1]={0};
    mkl_zcsrdia (job, &x_n , acsr_4 , ja_4 , ia_4 , quartic_V_C , &x_n , distance , &idiag , acsr_rem , ja_rem , ia_rem , &info );
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_z_create_csr (&temp, SPARSE_INDEX_BASE_ZERO, x_n, x_n, ia_4, ia_4+1, ja_4, acsr_4) && init_status==0){init_status = -5;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_copy (temp, {SPARSE_MATRIX_TYPE_DIAGONAL,SPARSE_FILL_MODE_UPPER,SPARSE_DIAG_NON_UNIT}, &V_hat) && init_status==0){
        init_status = -5;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_destroy (temp) && init_status==0){init_status = -5;}

    // create identity CSR matrix
    MKL_Complex16 identity_C[x_n]; for(int i=0;i<x_n;i++){identity_C[i]={1.,0.};} /* it does not initialize in the form of ={{1.,0.}}. */
    MKL_Complex16 acsr_5[x_n]={{0.,0.}};
    MKL_INT ja_5[x_n]={0}, ia_5[x_n+1]={0};
    mkl_zcsrdia (job, &x_n , acsr_5 , ja_5 , ia_5 , identity_C , &x_n , distance , &idiag , acsr_rem , ja_rem , ia_rem , &info );
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_z_create_csr (&temp, SPARSE_INDEX_BASE_ZERO, x_n, x_n, ia_5, ia_5+1, ja_5, acsr_5) && init_status==0){init_status = -6;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_copy (temp, {SPARSE_MATRIX_TYPE_HERMITIAN,SPARSE_FILL_MODE_UPPER,SPARSE_DIAG_UNIT}, &identity) && init_status==0){
        init_status = -6;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_destroy (temp) && init_status==0){init_status = -6;}

    // create p_hat, p_hat_2
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_z_add (SPARSE_OPERATION_NON_TRANSPOSE, delta_x, {0.,-1.}, empty_matrix, &p_hat) && init_status==0){init_status = -7;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_z_add (SPARSE_OPERATION_NON_TRANSPOSE, delta_2_x, {-1.,0.}, empty_matrix, &p_hat_2) && init_status==0){init_status = -7;}
    // create xp_px_hat
    sparse_matrix_t temp2;
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_spmm (SPARSE_OPERATION_NON_TRANSPOSE, x_hat, p_hat, &temp) && init_status==0){init_status = -8;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_copy (temp, {SPARSE_MATRIX_TYPE_GENERAL,SPARSE_FILL_MODE_UPPER,SPARSE_DIAG_NON_UNIT}, &temp2) && init_status==0){init_status = -8;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_z_add (SPARSE_OPERATION_CONJUGATE_TRANSPOSE, temp, {1.,0.}, temp2, &xp_px_hat) && init_status==0){init_status = -8;}
    if ((SPARSE_STATUS_SUCCESS!=mkl_sparse_destroy (temp)||SPARSE_STATUS_SUCCESS!=mkl_sparse_destroy (temp2)) && init_status==0){init_status = -9;}
 
    // create quartic_Hamiltonian
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_z_add (SPARSE_OPERATION_NON_TRANSPOSE, p_hat_2, {1./(2.*mass),0.}, V_hat, &quartic_Hamil) && init_status==0){init_status = -10;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_order (quartic_Hamil) && init_status==0){init_status = -11;}
    if ((SPARSE_STATUS_SUCCESS!=mkl_sparse_order (x_hat)||SPARSE_STATUS_SUCCESS!=mkl_sparse_order (p_hat)) && init_status==0){init_status = -12;}


    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_set_mv_hint (quartic_Hamil, SPARSE_OPERATION_NON_TRANSPOSE, descr, 10000000) && init_status==0){init_status = -13;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_optimize (quartic_Hamil) && init_status==0){init_status = -14;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_set_mv_hint (p_hat, SPARSE_OPERATION_NON_TRANSPOSE, {SPARSE_MATRIX_TYPE_HERMITIAN, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT}, 1000000) && init_status==0){init_status = -14;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_optimize (p_hat) && init_status==0){init_status = -15;}
    }
    
    // use of mkl sparse handler involves memory leak problems. Because mkl is based on C, not C++
    ~Set_World(){
        mkl_sparse_destroy (empty_matrix);
        mkl_sparse_destroy (delta_x); mkl_sparse_destroy (delta_2_x);
        mkl_sparse_destroy (x_hat); mkl_sparse_destroy (p_hat); mkl_sparse_destroy (p_hat_2); mkl_sparse_destroy (xp_px_hat);
        mkl_sparse_destroy (quartic_Hamil);
    }
};

const static Set_World world;
static int check_type(PyArrayObject* state);

static void compute_x_hat_state(const double &alpha, const MKL_Complex16* psi, const double &beta, MKL_Complex16* result){
    const double* psi_pointer;
    double* result_pointer; 
    psi_pointer = &(psi[0].real), result_pointer = &(result[0].real);
    if(beta==0.){ 
    for(int i=0; i<x_n; i++){
        result_pointer[2*i] = alpha * psi_pointer[2*i]*world.x[i];
        result_pointer[2*i+1] = alpha * psi_pointer[2*i+1]*world.x[i];}
    }else{
    for(int i=0; i<x_n; i++){
        result_pointer[2*i] *= beta;
        result_pointer[2*i] += alpha * psi_pointer[2*i]*world.x[i];
        result_pointer[2*i+1] *= beta;
        result_pointer[2*i+1] += alpha * psi_pointer[2*i+1]*world.x[i];}
    }
}
static double x_expct(const MKL_Complex16* psi){
    MKL_Complex16 x_hat_state[x_n]={{0.,0.}};
    compute_x_hat_state(1., psi, 0., x_hat_state);
    MKL_Complex16 temp;
    cblas_zdotc_sub (x_n, psi, 1, x_hat_state, 1, &temp);
    return temp.real*grid_size;
}
static double p_expct(const MKL_Complex16* psi){
    MKL_Complex16 p_hat_state[x_n]={{0.,0.}};
    mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, {1.,0.}, world.p_hat, {SPARSE_MATRIX_TYPE_HERMITIAN, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT}, psi, {0.,0.}, p_hat_state);
    MKL_Complex16 temp;
    cblas_zdotc_sub (x_n, psi, 1, p_hat_state, 1, &temp);
    return temp.real*grid_size;
}
static PyObject* x_expectation(PyObject *self, PyObject *args){
    PyObject* temp;
    PyArrayObject* state;
    if (!PyArg_ParseTuple(args, "O", &temp)){
        PyErr_SetString(PyExc_TypeError, "The input state is not a Numpy array");
        return NULL;
    }
    PyArray_OutputConverter(temp, &state);
    if (!state) {PyErr_SetString(PyExc_TypeError, "The input state cannot be identified as a Numpy array"); return NULL;}
    if (check_type(state)!=0) return NULL;
    // get the c array from numpy (complex128 is stored as successive float64 two by two)
    MKL_Complex16* psi;
    psi = (MKL_Complex16*) PyArray_GETPTR1(state, 0);
    return Py_BuildValue("d", x_expct(psi));
}
static void normalize(MKL_Complex16* psi){
    double norm;
    norm = cblas_dznrm2 (x_n, psi, 1);
    cblas_zdscal (x_n, 1./norm / std::sqrt(grid_size), psi, 1);
}

static double x_relative[x_n];
// the following two functions assume that "x_relative" is already prepared in advance
// the use of "x_relative" is only involved in the following functions that compute distribuion moments
static void compute_x_relative_state(const MKL_Complex16* psi, MKL_Complex16* result){
    const double* psi_pointer;
    double* result_pointer; psi_pointer = &(psi[0].real), result_pointer = &(result[0].real);
    for(int i=0; i<x_n; i++){
        result_pointer[2*i] =  psi_pointer[2*i]*x_relative[i];
        result_pointer[2*i+1] = psi_pointer[2*i+1]*x_relative[i];}
}
static void compute_x_relative_inplace(MKL_Complex16* state){
    double* state_pointer;
    state_pointer = &(state[0].real);
    for(int i=0; i<x_n; i++){
        state_pointer[2*i] *= x_relative[i];
        state_pointer[2*i+1] *= x_relative[i];}
}
static sparse_matrix_t p_hat_relative;
static inline void compute_x_or_p_relative_state(const MKL_Complex16* psi, char x_or_p, MKL_Complex16* result){
    if(x_or_p=='x'){compute_x_relative_state(psi,result);}
    if(x_or_p=='p'){mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, {1.,0.}, p_hat_relative, {SPARSE_MATRIX_TYPE_HERMITIAN, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT}, psi, {0.,0.}, result);}
}

static int check_type(PyArrayObject* state){
    if(PyArray_NDIM(state)!=1){
        PyErr_SetString(PyExc_ValueError, "The state array is not one-dimensional");
        return -1;
    }else{
    if(PyArray_SHAPE(state)[0]!=x_n){
        PyErr_SetString(PyExc_ValueError, ("The state array does not match the required size "+std::to_string(x_n)).c_str());
        return -1;
    }else{
        PyArray_Descr* descr = PyArray_DESCR(state);
        if(descr->type_num!=NPY_COMPLEX128){
            PyErr_SetString(PyExc_ValueError, "The state array does not match the required datatype: Complex128");
            return -1;
        }        
        return 0;
    }
    }
}
static int check_moment_data_array(PyArrayObject* data, int dimension){
    if(PyArray_NDIM(data)!=1){
        PyErr_SetString(PyExc_ValueError, "The moment data array is not one-dimensional");
        return -1;
    }else{
    if(PyArray_SHAPE(data)[0]!=dimension){
        PyErr_SetString(PyExc_ValueError, ("The moment data array does not match the required size "+std::to_string(dimension)).c_str());
        return -1;
    }else{
        PyArray_Descr* descr = PyArray_DESCR(data);
        if(descr->type_num!=NPY_FLOAT64){
            PyErr_SetString(PyExc_ValueError, "The moment data array does not match the required datatype: Float64");
            return -1;
        }        
        return 0;
    }
    }
}

static int moment_order = MOMENT;
static int compute_statistics(const MKL_Complex16* psi, double* data){
    // data are arranged in the order of:
    // <x>, <p>
    // centered moments -- <xx>, Re<xp>, <pp>, <xxx>, Re<xxp>, Re<xpp>, <ppp>, <xxxx>, Re<xxxp>, Re<xxpp>, Re<xppp>, <pppp>, 
    // <xxxxx>, Re<xxxxp>, Re<xxxpp>, Re<xxppp>, Re<xpppp>, Re<ppppp>

    // calculate <x>, <p> and prepare (x_hat-<x>) and (p_hat-<p>)
    data[0] = x_expct(psi), data[1] = p_expct(psi);
    for(int i=0;i<x_n;i++){x_relative[i]=world.x[i]-data[0];}
    sparse_status_t status=mkl_sparse_destroy (p_hat_relative);
    if(SPARSE_STATUS_SUCCESS!=status && SPARSE_STATUS_NOT_INITIALIZED!=status){return -1;}
    mkl_sparse_z_add (SPARSE_OPERATION_NON_TRANSPOSE, world.identity, {-data[1],0.}, world.p_hat, &p_hat_relative);
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_set_mv_hint (p_hat_relative, SPARSE_OPERATION_NON_TRANSPOSE, {SPARSE_MATRIX_TYPE_HERMITIAN, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT}, 5)){return -2;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_optimize (p_hat_relative)){return -2;}

    // arrange a temp array in the form of x|psi>, p|psi>, pp|psi>, ppp|psi>, pppp|psi>, ppppp|psi>, ...
    MKL_Complex16 temp[moment_order+1][x_n];
    compute_x_or_p_relative_state(psi, 'x', temp[0]);
    compute_x_or_p_relative_state(psi, 'p', temp[1]);
    for(int i=2;i<moment_order+1;i++){compute_x_or_p_relative_state(temp[i-1], 'p', temp[i]);}
    // prepare the moments
    MKL_Complex16 temp_scalar;
    int data_i = 2;
    for(int j=2;j<=moment_order;j++){
        for(int i=0;i<j;i++){
            // e.g. for moment 3, we change the temp array to the form of xxx|psi>, xxp|psi>, xpp|psi>, ppp|psi>, pppp|psi>, ppppp|psi>, ...,
            // and we collect the first four values
            compute_x_relative_inplace(temp[i]);}
        for(int i=0;i<j+1;i++){
            cblas_zdotc_sub (x_n, psi, 1, temp[i], 1, &temp_scalar);
            data[data_i] = temp_scalar.real*grid_size;
            data_i+=1;
        }
    }

    return 0;
}
static PyObject* get_moments(PyObject *self, PyObject *args){
    PyObject* temp1, *temp2;
    PyArrayObject* state, *data_np;
    if (!PyArg_ParseTuple(args, "OO", &temp1, &temp2)){
        PyErr_SetString(PyExc_TypeError, "The input does not match the required signature (state array (complex), moment data array (float))");
        return NULL;
    }
    PyArray_OutputConverter(temp1, &state);
    if (!state) {
        PyErr_SetString(PyExc_TypeError, "The input state (argument 1) cannot be identified as a Numpy array");
        return NULL;
    }
    PyArray_OutputConverter(temp2, &data_np);
    if (!state) {
        PyErr_SetString(PyExc_TypeError, "The input moment data array (argument 2) cannot be identified as a Numpy array");
        return NULL;
    }
    if (check_type(state)!=0) return NULL;
    if (check_moment_data_array(data_np, (2+moment_order+1)*moment_order/2)!=0) return NULL;
    MKL_Complex16* psi;
    psi = (MKL_Complex16*) PyArray_GETPTR1(state, 0);
    double* data;
    data = (double*) PyArray_GETPTR1(data_np, 0);
    if (compute_statistics(psi, data)!=0) return NULL;
    Py_RETURN_NONE;
}

static double _dt_cache=0., _force_cache=-1000000.;
static sparse_matrix_t Hamiltonian_addup_factor;
static sparse_matrix_t Hamil_temp[8];
// reset matrix ab
static int reset_ab(bool change_t){

    // compute the central diagonal
    cblas_dcopy (x_n, world.ab_center, 1, &(ab[derivative_estimation_points][0].imag), 2);
    cblas_dscal (x_n, _dt_cache, &(ab[derivative_estimation_points][0].imag), 2);
    cblas_daxpy (x_n, -_dt_cache*_force_cache*0.5*M_PI, world.x, 1, &(ab[derivative_estimation_points][0].imag), 2);
    if(change_t){
        // compute the upper and lower diagonals
        cblas_dcopy (derivative_estimation_points*x_n, world.ab_upper[0], 1, &(ab[0][0].imag), 2);
        cblas_dscal (derivative_estimation_points*x_n, _dt_cache, &(ab[0][0].imag), 2);
        cblas_dcopy (derivative_estimation_points*x_n, world.ab_lower[0], 1, &(ab[derivative_estimation_points+1][0].imag), 2);
        cblas_dscal (derivative_estimation_points*x_n, _dt_cache, &(ab[derivative_estimation_points+1][0].imag), 2);
    }
    sparse_status_t status=mkl_sparse_destroy (Hamiltonian_addup_factor);
    if(SPARSE_STATUS_SUCCESS!=status && SPARSE_STATUS_NOT_INITIALIZED!=status){return -1;}

    cblas_zcopy ((2*derivative_estimation_points+1)*x_n, &(ab[0][0]), 1, &(ab_LU[derivative_estimation_points][0]), 1);
    LAPACKE_zgbtrf (LAPACK_ROW_MAJOR, x_n , x_n , derivative_estimation_points , derivative_estimation_points , &(ab_LU[0][0]) , x_n , LU_ipiv );

    // compute the high order correction factors of the Hamiltonian evolution
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_z_add (SPARSE_OPERATION_NON_TRANSPOSE, world.x_hat, {-M_PI*_force_cache, 0.}, world.quartic_Hamil, &Hamil_temp[0])) return -1;
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_spmm (SPARSE_OPERATION_NON_TRANSPOSE, Hamil_temp[0], Hamil_temp[0], &Hamil_temp[1])) return -1; // H^2
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_spmm (SPARSE_OPERATION_NON_TRANSPOSE, Hamil_temp[1], Hamil_temp[0], &Hamil_temp[2])) return -1; // H^3
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_spmm (SPARSE_OPERATION_NON_TRANSPOSE, Hamil_temp[1], Hamil_temp[1], &Hamil_temp[3])) return -1; // H^4
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_spmm (SPARSE_OPERATION_NON_TRANSPOSE, Hamil_temp[1], Hamil_temp[2], &Hamil_temp[4])) return -1; // H^5
    // the following is the correction factor 3-4-5-6 for an implicit method
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_z_add (SPARSE_OPERATION_NON_TRANSPOSE, Hamil_temp[1], {_dt_cache*_dt_cache*_dt_cache / 12., 0.}, world.empty_matrix, &Hamil_temp[5])) return -1;
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_z_add (SPARSE_OPERATION_NON_TRANSPOSE, Hamil_temp[2], {0., - _dt_cache*_dt_cache*_dt_cache*_dt_cache / 24.}, Hamil_temp[5], &Hamil_temp[6])) return -1;
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_z_add (SPARSE_OPERATION_NON_TRANSPOSE, Hamil_temp[3], {- _dt_cache*_dt_cache*_dt_cache*_dt_cache*_dt_cache / 80.,0.} \
                                                                    , Hamil_temp[6], &Hamil_temp[7])) return -1;
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_z_add (SPARSE_OPERATION_NON_TRANSPOSE, Hamil_temp[4], {0., _dt_cache*_dt_cache*_dt_cache*_dt_cache*_dt_cache*_dt_cache / 360.} \
                                                                    , Hamil_temp[7], &Hamiltonian_addup_factor)) return -1;
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_set_mv_hint (Hamiltonian_addup_factor, SPARSE_OPERATION_NON_TRANSPOSE, descr, 80)) return -1;
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_order(Hamiltonian_addup_factor)) return -1; 
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_optimize (Hamiltonian_addup_factor)) return -1;
    // avoid memory leak
    for(int i=0;i<8;i++){if(SPARSE_STATUS_SUCCESS!=mkl_sparse_destroy (Hamil_temp[i])){return -1;}}
    return 0;
}
// carry out one step calculation
static void __attribute__((hot)) D1(MKL_Complex16* state, const double &force, const double &gamma, MKL_Complex16* result, MKL_Complex16* relative_state, double x_avg){
    MKL_Complex16 x_hat_state[x_n];
    compute_x_hat_state(1., state, 0., x_hat_state);

    // compute the relative_state 
    cblas_zcopy (x_n, x_hat_state, 1, relative_state, 1); // x_hat_state = relative_state = x|psi>
    cblas_daxpy (2*x_n, -x_avg, (double*)state, 1, (double*)relative_state, 1); // relative_state = (x-<x>)|psi>
    // compute the deterministic Hamiltonian term, stored in x_hat_state
    mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, {0.,-1.}, world.quartic_Hamil, descr, state, {0.,+1.*M_PI*force}, x_hat_state);
    cblas_zcopy (x_n, x_hat_state, 1, result, 1); // store the result: x_hat_state = result = -1.j(H|psi>-oemga*F*x|psi>)
    // compute the deterministic squeezing term , stored in x_hat_state
    cblas_zcopy (x_n, relative_state, 1, x_hat_state, 1); // x_hat_state = (x-<x>)|psi>
    compute_x_hat_state(1., relative_state, -x_avg, x_hat_state); // x_hat_state = (x(x-<x>)|psi>-<x>(x-<x>)|psi>) = (x-<x>)^2 |psi>
    // compute the result. The coefficient of the first term (currently in result) is already handled.
    cblas_daxpy (2*x_n, -gamma/4., (double*)x_hat_state, 1, (double*)result, 1); // result = -1.j(H-oemga*F*x)|psi> -gamma/4.(x-<x>)^2 |psi>
}

// The "ImRe" is a mistaken notation. The "Im" part refers to the Hamiltonian part that can be evaluated with implicit methods
static void __attribute__((hot)) D1ImRe(MKL_Complex16* state, const double &force, const double &gamma, MKL_Complex16* resultIm, MKL_Complex16* resultRe, MKL_Complex16* relative_state){
    // resultIm can replace the role of x_hat_state
    compute_x_hat_state(1., state, 0., resultIm);

        // compute x_avg
        double x_avg;
        MKL_Complex16 temp;
        cblas_zdotc_sub (x_n, state, 1, resultIm, 1, &temp);
        x_avg = temp.real*grid_size;

    // compute the relative_state
    cblas_zcopy (x_n, resultIm, 1, relative_state, 1);
    cblas_daxpy (2*x_n, -x_avg, (double*)state, 1, (double*)relative_state, 1);
    // compute the deterministic Hamiltonian term, stored in resultIm
    mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, {0.,-1,}, world.quartic_Hamil, descr, state,{0.,+1.*M_PI*force}, resultIm);
    // compute the deterministic squeezing term , stored in resultRe
    cblas_zcopy (x_n, relative_state, 1, resultRe, 1);
    compute_x_hat_state(-gamma/4., relative_state, x_avg*gamma/4., resultRe);
    // compute the result with a required coefficient gamma/-4
}

static void __attribute__((hot)) D2(MKL_Complex16* state, const double &gamma, MKL_Complex16* relative_state_and_result, bool precomputed_relative_state){
    if (! precomputed_relative_state){
        // compute the relative_state 
        // compute x|psi>
        compute_x_hat_state(1., state, 0., relative_state_and_result);
        // compute <x>
        double x_avg; 
        MKL_Complex16 temp;
        cblas_zdotc_sub (x_n, state, 1, relative_state_and_result, 1, &temp);
        x_avg = temp.real*grid_size;
        // compute (x-<x>)|psi>
        cblas_daxpy (2*x_n, -x_avg, (double*)state, 1, (double*)relative_state_and_result, 1);}
    cblas_zdscal (x_n, sqrt(gamma/2.), relative_state_and_result, 1);
}

static VSLStreamStatePtr stream;

static void go_one_step(MKL_Complex16* psi, double dt, double force, double _gamma, double* q_output, double* x_mean_output);
static void check_boundary_error(MKL_Complex16* psi, int* Fail);

static PyObject* step(PyObject *self, PyObject *args){
    double dt, force, _gamma;
    PyObject* temp;
    PyArrayObject* state;
    if (!PyArg_ParseTuple(args, "Oddd", &temp, &dt, &force, &_gamma)){
        PyErr_SetString(PyExc_TypeError, "The input does not match the required input signature (state (numpy array), dt (double), F (double), \\gamma (double))");
        return NULL;
    }
    PyArray_OutputConverter(temp, &state);
    if (!state) {
        PyErr_SetString(PyExc_TypeError, "The input object cannot be identified as a Numpy array");
        return NULL;
    }
    // check the shape and datatype of the array, so that an error will be raised if the shape is not consistent with this c module, avoiding a segmentation fault
    if (check_type(state)==-1) return NULL;

    // get the c array from numpy (complex128 is stored as successive float64 two by two)
    MKL_Complex16* psi;
    psi = (MKL_Complex16*) PyArray_GETPTR1(state, 0);
    // update the cache of dt and force
    bool change_t;
    change_t=(dt != _dt_cache);
    if (change_t||(force != _force_cache)){
        _dt_cache=dt;_force_cache=force;
        if(0!=reset_ab(change_t)){PyErr_SetString(PyExc_RuntimeError, "Failed to manage the sparse matrices when setting the implicit solver"); return NULL;}
    }
    double q=0., x_mean=0.;
    go_one_step(psi, dt, force, _gamma, &q, &x_mean);
    int Fail=0; // compute the norm of a single value in psi
    check_boundary_error(psi, &Fail);

    return Py_BuildValue("ddi", q, x_mean, Fail); 
}
static PyObject* simulate_10_steps(PyObject *self, PyObject *args){
    double dt, force, _gamma;
    PyObject* temp;
    PyArrayObject* state;
    if (!PyArg_ParseTuple(args, "Oddd", &temp, &dt, &force, &_gamma)){
        PyErr_SetString(PyExc_TypeError, "The input does not match the required input signature (state (numpy array), dt (double), F (double), \\gamma (double))");
        return NULL;
    }
    PyArray_OutputConverter(temp, &state);
    if (!state) {
        PyErr_SetString(PyExc_TypeError, "The input object cannot be identified as a Numpy array");
        return NULL;
    }
    // check the shape and datatype of the array, so that an error will be raised if the shape is not consistent with this c module, avoiding a segmentation fault
    if (check_type(state)==-1) return NULL;

    // get the c array from numpy (complex128 is stored as successive float64 two by two)
    MKL_Complex16* psi;
    psi = (MKL_Complex16*) PyArray_GETPTR1(state, 0);
    // update the cache of dt and force
    bool change_t;
    change_t=(dt != _dt_cache);
    if (change_t||(force != _force_cache)){
        _dt_cache=dt;_force_cache=force;
        if(0!=reset_ab(change_t)){PyErr_SetString(PyExc_RuntimeError, "Failed to manage the sparse matrices when setting the implicit solver"); return NULL;}
    }
    double q=0., x_mean=0.;
    for(int i=0; i<10; i++){go_one_step(psi, dt, force, _gamma, &q, &x_mean);}

    int Fail=0; // compute the norm of a single value in psi
    check_boundary_error(psi, &Fail);
    return Py_BuildValue("ddi", q, x_mean, Fail); 
}
static void check_boundary_error(MKL_Complex16* psi, int* Fail){
    int boundary_length = 6;
    double threshold = 5.e-3;
    if ( cblas_dznrm2(boundary_length, &(psi[x_n-boundary_length]), 1)>threshold || cblas_dznrm2(boundary_length, &(psi[0]), 1)>threshold){
        *Fail = 1; 
        }
}

static void simple_sum_up(double* state, double dt, double dW, double dZ, double* D2_state, double* D1_Y_plusIm_substract_D1_Y_minusIm, double* D1_Y_plusRe, \
    double* D1_Y_minusRe, double*  D1_state, double* D2_Y_plus, double* D2_Y_minus, double* D2_Phi_plus, double* D2_Phi_minus);
static void __attribute__((hot)) go_one_step(MKL_Complex16* psi, double dt, double force, double _gamma, double* q_output, double* x_mean_output){ 
    // sample random variables
    double r[2], dW, dZ;
    vdRngGaussian( VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, 2, r, 0., 1. );
    dW = r[0]*sqrt(dt), dZ = sqrt(dt)*dt*0.5*(r[0]+r[1]/sqrt(3.));
    double x_mean, q;
    x_mean = x_expct(psi);

    q = x_mean + dW/sqrt(2.*_gamma)/dt;
    *q_output=q; *x_mean_output=x_mean;
    // start calculation
    MKL_Complex16 D1_state[x_n], D2_state[x_n];
    MKL_Complex16 D2_state_drt[x_n]={{0.,0.}};
    D1(psi, force, _gamma, D1_state, D2_state, x_mean);
    D2(psi, _gamma, D2_state, true);

    cblas_daxpy (2*x_n, sqrt(dt), (double*)D2_state, 1, (double*)D2_state_drt, 1);

    // initialize Y as a 1st order step forward from psi
    MKL_Complex16 Y_plus[x_n], Y_minus[x_n];
    cblas_zcopy (x_n, psi, 1, Y_plus, 1); 

    cblas_daxpy (2*x_n, dt, (double*)D1_state, 1, (double*)Y_plus , 1); cblas_zcopy (x_n, Y_plus, 1, Y_minus, 1);

    cblas_daxpy (2*x_n, 1., (double*)D2_state_drt, 1, (double*)Y_plus , 1); 
    cblas_daxpy (2*x_n,-1., (double*)D2_state_drt, 1, (double*)Y_minus, 1);
    MKL_Complex16  D1_Y_plusIm[x_n],  D1_Y_plusRe[x_n],  D2_Y_plus[x_n];
    MKL_Complex16 D1_Y_minusIm[x_n], D1_Y_minusRe[x_n], D2_Y_minus[x_n];
    D1ImRe(Y_plus, force, _gamma, D1_Y_plusIm, D1_Y_plusRe, D2_Y_plus);
    D1ImRe(Y_minus, force, _gamma, D1_Y_minusIm, D1_Y_minusRe, D2_Y_minus);
    D2(Y_plus,_gamma, D2_Y_plus, true); D2(Y_minus,_gamma, D2_Y_minus, true);
    
    // use the storage of D1_Y_plusIm as another variable name:
    MKL_Complex16* D1_Y_plusIm_substract_D1_Y_minusIm; D1_Y_plusIm_substract_D1_Y_minusIm=D1_Y_plusIm;
    cblas_daxpy (2*x_n, -1., (double*)D1_Y_minusIm, 1, (double*)D1_Y_plusIm_substract_D1_Y_minusIm, 1);
    // use the storage of Y_minus as another variable name: Phi_minus
    MKL_Complex16* Phi_minus; Phi_minus=Y_minus;
    cblas_zcopy (x_n, Y_plus, 1, Phi_minus, 1); 
    // Phi_minus = Y_plus - sqrt(dt) D2(Y_plus)
    cblas_daxpy (2*x_n, - sqrt(dt), (double*)D2_Y_plus, 1, (double*)Phi_minus, 1);
    // use the storage of Y_plus as Phi_plus
    // Phi_plus = Y_plus + sqrt(dt) D2(Y_plus)
    MKL_Complex16* Phi_plus; Phi_plus=Y_plus;
    cblas_daxpy (2*x_n, sqrt(dt), (double*)D2_Y_plus, 1, (double*)Phi_plus, 1);

    MKL_Complex16 D2_Phi_plus[x_n], D2_Phi_minus[x_n];
    D2(Phi_plus, _gamma, D2_Phi_plus, false); D2(Phi_minus, _gamma, D2_Phi_minus, false);

    // sum them all
    simple_sum_up((double*)psi, dt, dW, dZ, (double*)D2_state, (double*)D1_Y_plusIm_substract_D1_Y_minusIm, (double*)D1_Y_plusRe, (double*)D1_Y_minusRe, \
                    (double*)D1_state, (double*)D2_Y_plus, (double*)D2_Y_minus, (double*)D2_Phi_plus, (double*)D2_Phi_minus);
   
    // implicitly solve
    LAPACKE_zgbtrs (LAPACK_ROW_MAJOR, 'N' , x_n , derivative_estimation_points , derivative_estimation_points , 1 , &(ab_LU[0][0]) , x_n , LU_ipiv , psi , 1 );
    normalize(psi);
}

static void __attribute__((hot)) simple_sum_up(double* state, double dt, double dW, double dZ, double* D2_state, double* D1_Y_plusIm_substract_D1_Y_minusIm, double* D1_Y_plusRe, \
    double* D1_Y_minusRe, double*  D1_state, double* D2_Y_plus, double* D2_Y_minus, double* D2_Phi_plus, double* D2_Phi_minus){

    MKL_Complex16 term7[x_n]={{0.,0.}};
    // note that D1_state has included a factor of -1.j ***
    mkl_sparse_z_mv (SPARSE_OPERATION_NON_TRANSPOSE, {1., 0.}, Hamiltonian_addup_factor, descr, (MKL_Complex16*)D1_state, {0.,0.}, term7);
    double* dterm7;
    dterm7=(double*)term7;
    // summation for the implicit method
    for(int i=0;i<2*x_n;i++){
        state[i]+=D2_state[i]*dW+0.5/sqrt(dt)*dZ*(D1_Y_plusIm_substract_D1_Y_minusIm[i]+D1_Y_plusRe[i]-D1_Y_minusRe[i])+\
                0.25*dt*(D1_Y_plusRe[i]+2*D1_state[i]+D1_Y_minusRe[i]) + \
                0.25/sqrt(dt)*(dW*dW-dt)*(D2_Y_plus[i] - D2_Y_minus[i]) + \
                0.5/dt*(dW*dt-dZ)*(D2_Y_plus[i] + D2_Y_minus[i] - 2*D2_state[i]) + \
                0.25/dt*(dW*dW/3-dt)*dW*(D2_Phi_plus[i] - D2_Phi_minus[i] - D2_Y_plus[i] + D2_Y_minus[i]) \
              - 0.25*sqrt(dt)*dW*(D1_Y_plusIm_substract_D1_Y_minusIm[i]) \
              + dterm7[i];
    }        
}
static PyObject* set_seed(PyObject* self, PyObject *args){
    int seed;
    if (!PyArg_ParseTuple(args, "i", &seed)){printf("Parse fail.\n"); return NULL;}
    vslNewStream( &stream, VSL_BRNG_MT19937, seed );
    Py_RETURN_NONE;
}

static PyObject* check_settings(PyObject* self, PyObject *args){
    return Py_BuildValue("(idddi)", x_n, grid_size, lambda, mass, moment_order);
}

static PyMethodDef methods[] = {
    {"step", (PyCFunction)step, METH_VARARGS,
     "Do one simulation step."},
    {"simulate_10_steps", (PyCFunction)simulate_10_steps, METH_VARARGS,
     "Do 10 simulation steps."},
    {"set_seed", (PyCFunction)set_seed, METH_VARARGS,
     "Initialize the random number generator with a seed."},
    {"check_settings", (PyCFunction)check_settings, METH_VARARGS,
     "test whether the imported C module responds and return (x_n,grid_size,\\lambda,mass)."},
    {"x_expectation", (PyCFunction)x_expectation,METH_VARARGS,""},
    {"get_moments", (PyCFunction)get_moments,METH_VARARGS,("get distribution moments up to the "+std::to_string(MOMENT)+"th").c_str()},
    {NULL, NULL, 0, NULL}   /* sentinel */
};

static struct PyModuleDef simulationmodule = {
    PyModuleDef_HEAD_INIT,
    "simulation",
    NULL,
    -1,
    methods
};

const int init = init_();

extern "C"{

PyMODINIT_FUNC
__attribute__((externally_visible)) PyInit_simulation(void)
{
    //mkl_set_memory_limit (MKL_MEM_MCDRAM, 256); // mbytes
    if(world.init_status!=0){
        printf("initialization status: %d\n",world.init_status);
        PyErr_SetString(PyExc_RuntimeError, "Initialization Failure");
    }
    //init_();
    return PyModule_Create(&simulationmodule);
}

}
