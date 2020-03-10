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
    import_array(); // PyError if not successful
    mkl_set_num_threads(1);
    return 0;
}

const MKL_INT n_max = N_MAX;
const double omega = OMEGA;

const static struct matrix_descr descr = {SPARSE_MATRIX_TYPE_HERMITIAN, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT};
static MKL_Complex16 ab[5][n_max+1]={{0.,0.}}, ab_LU[7][n_max+1]={{0.,0.}}; MKL_INT LU_ipiv[n_max+1];

class Set_World{
public:
    double n_phonon[n_max+1], sqrt_n[n_max];
    sparse_matrix_t annihilation, creation;
    sparse_matrix_t x_hat, x_hat_2, p_hat_2, harmonic_Hamil;
    sparse_matrix_t p_hat, xp_px_hat; // these two involve complex numbers
    double x_upper_diag[n_max+1], x_lower_diag[n_max+1];
    //double x_banded[2*(n_max+1)]={0.};
    double x_upper_diag_ab[n_max+1]={0.}, x_lower_diag_ab[n_max+1]={0.};
    double ab_upper2[n_max+1]={0.}, ab_lower2[n_max+1]={0.};


    int init_status = 0;

    sparse_matrix_t empty_matrix; // the "empty_matrix" is for simpler calculation writing
    MKL_INT* empty_rowindex_start, *empty_rowindex_end, *empty_column; // *** Somehow MKL does not allow copying and automatically allocating such a matrix,
    MKL_Complex16* empty_value;                                        // and we have to manage its memory manually.
    ///////////////////////////// *** initializer *** ///////////////////////////////////////////

    Set_World(){
    for(int i=0;i < n_max+1;i++){n_phonon[i]=(double)(i);}

    for(int i=1;i < n_max+1;i++){sqrt_n[i-1]=std::sqrt((double)(i));}
    MKL_INT n=n_max+1;

    sparse_status_t status; 
    empty_rowindex_start = (MKL_INT*)mkl_malloc(sizeof(MKL_INT)*n_max+1, 64);
    empty_rowindex_end =   (MKL_INT*)mkl_malloc(sizeof(MKL_INT)*n_max+1, 64);
    empty_column =         (MKL_INT*)mkl_malloc(sizeof(MKL_INT)*1      , 64);
    empty_value =    (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16)*1, 64);
    for(int i=0;i<n_max+1;i++){
        empty_rowindex_start[i]=1; empty_rowindex_end[i]=1;
    }
    empty_rowindex_start[0]=0; empty_column[0]=0; empty_value[0]={0.,0.};
    //MKL_INT empty_rowindex_start[n_max+1]={1}, empty_rowindex_end[n_max+1]={1}, empty_column[1]={0}; MKL_Complex16 empty_value[1]={{0.1,0.}};
    
    status= mkl_sparse_z_create_csr (&empty_matrix, SPARSE_INDEX_BASE_ZERO, n, n, empty_rowindex_start, empty_rowindex_end, empty_column, empty_value);
    if (status!=SPARSE_STATUS_SUCCESS && init_status==0){init_status = -1;}

    double sqrt_n_append1[n_max+1], sqrt_n_1append[n_max+1];
    for(int i=1;i < n_max+1;i++){ 
        sqrt_n_append1[i-1]=std::sqrt((double)(i)); 
        sqrt_n_1append[i]=std::sqrt((double)(i)); 
        } sqrt_n_append1[n_max]=0., sqrt_n_1append[0]=0.;
    for(int i=0;i < n_max+1;i++){
        x_upper_diag[i]=sqrt_n_1append[i]*sqrt(0.5); //x_banded[2*i]=x_upper_diag[i];
        x_lower_diag[i]=sqrt_n_append1[i]*sqrt(0.5);
        x_upper_diag_ab[i]=-x_upper_diag[i]*0.5*omega;
        x_lower_diag_ab[i]=-x_lower_diag[i]*0.5*omega;
        }

    // prepare for mkl_?csrdia(...)
    MKL_INT job[6]={1,0,0,0,0,0};
    /* diagonal -> CSR, zero-based indexing in CSR, and diagonal,
       N/A, N/A, check zeroes and leave out them */

    // vairables for the CSR matrix, the "_rem"s are not used. There should be exactly n_max values in total. 
    double acsr[n_max+1]; double* acsr_rem; MKL_INT ja[n_max+1], ia[n_max+1+1]; MKL_INT* ja_rem, *ia_rem;
    // variables for the diagonal format matrix. "adia" is "sqrt_n_append1".
    MKL_INT ndiag=n, distance[1]={1}, idiag=1, info;
    // change the format to CSR:
    mkl_dcsrdia (job, &n , acsr , ja , ia , sqrt_n_append1 , &ndiag , distance , &idiag , acsr_rem , ja_rem , ia_rem , &info );
    //transform the value array into a proper complex valued array, by filling zeroes.
    MKL_Complex16 acsr_z[n_max]={{0.,0.}};
    for (int i=0;i<n_max;i++){ acsr_z[i].real = acsr[i]; }

    //create annihilation and creation operator
    sparse_matrix_t temp, temp2, temp3;
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_z_create_csr (&temp, SPARSE_INDEX_BASE_ZERO, n, n, ia, ia+1, ja, acsr_z) && init_status==0){init_status = -2;}

    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_copy (temp, {SPARSE_MATRIX_TYPE_GENERAL,SPARSE_FILL_MODE_UPPER,SPARSE_DIAG_NON_UNIT}, &annihilation) && init_status==0){init_status = -2;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_destroy (temp) && init_status==0){init_status = -2;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_z_add (SPARSE_OPERATION_TRANSPOSE, annihilation, {1.,0.}, empty_matrix, &creation) && init_status==0){init_status = -3;}

    //create x_hat, p_hat, x_hat_2, p_hat_2, xp_px_hat
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_z_add (SPARSE_OPERATION_NON_TRANSPOSE, creation, {sqrt(0.5),0.}, empty_matrix, &temp) && init_status==0){init_status = -4;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_z_add (SPARSE_OPERATION_NON_TRANSPOSE, annihilation, {sqrt(0.5),0.}, temp, &x_hat) && init_status==0){init_status = -5;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_destroy (temp) && init_status==0){init_status = -6;}

    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_z_add (SPARSE_OPERATION_NON_TRANSPOSE, creation, {0.,sqrt(0.5)}, empty_matrix, &temp) && init_status==0){init_status = -6;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_z_add (SPARSE_OPERATION_NON_TRANSPOSE, annihilation, {0.,-sqrt(0.5)}, temp, &p_hat) && init_status==0){init_status = -7;}

    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_spmm (SPARSE_OPERATION_NON_TRANSPOSE, x_hat, x_hat, &x_hat_2) && init_status==0){init_status = -8;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_spmm (SPARSE_OPERATION_NON_TRANSPOSE, p_hat, p_hat, &p_hat_2) && init_status==0){init_status = -9;}

    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_destroy (temp) && init_status==0){init_status = -10;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_spmm (SPARSE_OPERATION_NON_TRANSPOSE, x_hat, p_hat, &temp) && init_status==0){init_status = -10;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_copy (temp, {SPARSE_MATRIX_TYPE_GENERAL,SPARSE_FILL_MODE_UPPER,SPARSE_DIAG_NON_UNIT}, &temp2) && init_status==0){init_status = -10;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_z_add (SPARSE_OPERATION_CONJUGATE_TRANSPOSE, temp, {1.,0.}, temp2, &xp_px_hat) && init_status==0){init_status = -11;}
    if ((SPARSE_STATUS_SUCCESS!=mkl_sparse_destroy (temp)||SPARSE_STATUS_SUCCESS!=mkl_sparse_destroy (temp2)) && init_status==0){init_status = -12;}

    //create Hamiltonian and upper/lower diagonals
    // take care!!! somehow, destroying an uninitialized sparse matrix results in a memory error in the constructor, without throwing out any error signal
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_spmm (SPARSE_OPERATION_NON_TRANSPOSE, creation, creation, &temp) && init_status==0){init_status = -13;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_spmm (SPARSE_OPERATION_NON_TRANSPOSE, annihilation, annihilation, &temp2) && init_status==0){init_status = -13;}

    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_z_add (SPARSE_OPERATION_NON_TRANSPOSE, temp, {-0.5*omega,0.}, empty_matrix, &temp3) && init_status==0){init_status = -14;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_z_add (SPARSE_OPERATION_NON_TRANSPOSE, temp2, {-0.5*omega,0.}, temp3, &harmonic_Hamil) && init_status==0){init_status = -15;}

    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_order (harmonic_Hamil) && init_status==0){init_status = -16;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_order (x_hat) && init_status==0){init_status = -17;}
    if ((SPARSE_STATUS_SUCCESS!=mkl_sparse_destroy (temp)||SPARSE_STATUS_SUCCESS!=mkl_sparse_destroy (temp2)||SPARSE_STATUS_SUCCESS!=mkl_sparse_destroy (temp3)) \
         && init_status==0){init_status = -21;}

    job[0]=0, job[5]=10;
    // **********IMPORTANT*********** Due to our explicit use of empty matrix in the form of stored 0-valued csr memory, it will be jugded as containing non-trivial values
    // and asks for more memory to proceed the calculation. Therefore, the required memory to store the diagonals is more than 2*(n_max+1), or otherwise it causes segmentation fault.
    MKL_INT distances[n_max+1], rows, cols, *rows_start, *rows_end, *col_indx;
    MKL_Complex16 diagonals[n_max+1][n_max+1], *values; // an array cannot be referenced to be passed as data_type ** xxx.
    sparse_index_base_t index_type;
    idiag = 2;
    MKL_Complex16 *z_acsr_rem;
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_z_export_csr (harmonic_Hamil, &index_type, &rows, &cols, &rows_start, &rows_end, &col_indx, &values) && init_status==0){init_status = -17;}
    MKL_INT rowindex[n_max+1+1]; for(int i=0;i<n_max+1;i++){rowindex[i]=rows_start[i];} rowindex[n_max+1]=rows_end[n_max];
    // ** it is not mentioned that compressed diagonal arrays are in column major format ! But they are! **
    mkl_zcsrdia (job , &n , values , col_indx , rowindex , &(diagonals[0][0]) , &ndiag , distances , &idiag , z_acsr_rem , ja_rem , ia_rem , &info);
    // initialize the matrix ab. *The storage of the extracted diagonals is of the reverse order of the banded matrix storage.
    for(int j=0;j<n_max+1;j++){
        if (distances[j]==-2){for(int i=0;i < n_max+1;i++){ab_lower2[i]=diagonals[j][i].real*0.5;}}
        if (distances[j]==+2){for(int i=0;i < n_max+1;i++){ab_upper2[i]=diagonals[j][i].real*0.5;}}
    }
    for(int i=0;i < n_max+1;i++){ab[2][i].real=1.;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_set_mv_hint (harmonic_Hamil, SPARSE_OPERATION_NON_TRANSPOSE, descr, 100000000) && init_status==0){init_status = -18;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_optimize (harmonic_Hamil) && init_status==0){init_status = -19;}
    if (SPARSE_STATUS_SUCCESS!=mkl_sparse_optimize (x_hat) && init_status==0){init_status = -20;}
    }
    
    // use of mkl sparse matrix handles can involve memory leaks. Because mkl is based on C, not C++
    ~Set_World(){
        mkl_sparse_destroy (empty_matrix);
        mkl_sparse_destroy (annihilation); mkl_sparse_destroy (creation);
        mkl_sparse_destroy (x_hat); mkl_sparse_destroy (x_hat_2); mkl_sparse_destroy (p_hat_2); mkl_sparse_destroy (harmonic_Hamil);
        mkl_sparse_destroy (p_hat); mkl_sparse_destroy (xp_px_hat);
    }
};

static const Set_World world;
/*
static void compute_x_hat_state(const double &alpha, MKL_Complex16* psi, double beta, MKL_Complex16* result){
    cblas_dsbmv(CblasColMajor,CblasUpper,n_max+1,1,alpha, world.x_banded, 2, &(psi->real), 2, beta, &(result->real),2);
    cblas_dsbmv(CblasColMajor,CblasUpper,n_max+1,1,alpha, world.x_banded, 2, &(psi->imag), 2, beta, &(result->imag),2);
}*/
static void compute_x_hat_state(const double &alpha, MKL_Complex16* psi, double beta, MKL_Complex16* result){
    double* psi_pointer, * result_pointer; psi_pointer = &(psi[0].real), result_pointer = &(result[0].real);
    if(beta==0.){ 
    result_pointer[0] = alpha * (psi_pointer[0+2]*world.x_lower_diag[0]);//world.x_lower_diag[0]
    result_pointer[1] = alpha * (psi_pointer[0+3]*world.x_lower_diag[0]);
    for(int i=1; i<n_max; i++){
        result_pointer[2*i] = alpha * (psi_pointer[2*i+2]*world.x_lower_diag[i]+psi_pointer[2*i-2]*world.x_lower_diag[i-1]);//world.x_lower_diag[i], world.x_lower_diag[i-1]
        result_pointer[2*i+1] = alpha * (psi_pointer[2*i+3]*world.x_lower_diag[i]+psi_pointer[2*i-1]*world.x_lower_diag[i-1]);
    }
    result_pointer[2*n_max] = alpha * (psi_pointer[2*n_max-2]*world.x_lower_diag[n_max-1]);//world.x_lower_diag[n_max-1]
    result_pointer[2*n_max+1] = alpha * (psi_pointer[2*n_max-1]*world.x_lower_diag[n_max-1]);
    }else{
        
    result_pointer[0] *= beta;
    result_pointer[0] += alpha * (psi_pointer[0+2]*world.x_lower_diag[0]);
    result_pointer[1] *= beta;
    result_pointer[1] += alpha * (psi_pointer[0+3]*world.x_lower_diag[0]);
    for(int i=1; i<n_max; i++){
        result_pointer[2*i] *= beta;
        result_pointer[2*i] += alpha * (psi_pointer[2*i+2]*world.x_lower_diag[i]+psi_pointer[2*i-2]*world.x_lower_diag[i-1]);
        result_pointer[2*i+1] *= beta;
        result_pointer[2*i+1] += alpha * (psi_pointer[2*i+3]*world.x_lower_diag[i]+psi_pointer[2*i-1]*world.x_lower_diag[i-1]);
    }
    result_pointer[2*n_max] *= beta;
    result_pointer[2*n_max] += alpha * (psi_pointer[2*n_max-2]*world.x_lower_diag[n_max-1]);
    result_pointer[2*n_max+1] *= beta;
    result_pointer[2*n_max+1] += alpha * (psi_pointer[2*n_max-1]*world.x_lower_diag[n_max-1]);
    }
}
static double x_expct(MKL_Complex16* psi){
    MKL_Complex16 x_hat_state[n_max+1];
    compute_x_hat_state(1., psi, 0., x_hat_state);
    MKL_Complex16 temp;
    cblas_zdotc_sub (n_max+1, psi, 1, x_hat_state, 1, &temp);
    return temp.real;
}
static PyObject* x_expectation(PyObject *self, PyObject *args){
    PyObject* temp;
    PyArrayObject* state;
    if (!PyArg_ParseTuple(args, "O", &temp))
        return NULL;
    PyArray_OutputConverter(temp, &state);
    if (!state) return NULL;
    // get the c array from numpy (complex128 is stored as successive float64 two by two)
    MKL_Complex16* psi;
    psi = (MKL_Complex16*) PyArray_GETPTR1(state, 0);
    return Py_BuildValue("d", x_expct(psi));
}
static void normalize(MKL_Complex16* psi){
    double norm;
    norm = cblas_dznrm2 (n_max+1, psi, 1);
    cblas_zdscal (n_max+1, 1./norm, psi, 1);
}

static double _dt_cache=0., _force_cache=0.;

// reset matrix ab
static sparse_matrix_t Hamiltonian_addup_factor;
static sparse_matrix_t Hamil_temp[8];
static int reset_ab(bool change_t){
    double* pointer1, *pointer2;
    pointer1=&(ab[1][1].imag), pointer2=&(ab[3][0].imag);
    double coefficent=_dt_cache*_force_cache;
    // compute the 1st upper diagonal
    cblas_dcopy (n_max, world.x_lower_diag_ab, 1, pointer2, 2);
    cblas_dscal (n_max, coefficent, pointer2, 2);
    // copy to the 1st lower diagonal
    cblas_dcopy (n_max, pointer2, 2, pointer1, 2);
    if(change_t){
        // compute the 2nd upper diagonal
        double* pointer3, *pointer4;
        pointer3=&(ab[0][2].imag), pointer4=&(ab[4][0].imag);
        cblas_dcopy (n_max-1, world.ab_upper2, 1, pointer4, 2);
        cblas_dscal (n_max-1, _dt_cache, pointer4, 2);
        // copy to the 2nd lower diagonal
        cblas_dcopy (n_max-1, pointer4, 2, pointer3, 2);
    }
    sparse_status_t status=mkl_sparse_destroy (Hamiltonian_addup_factor);
    if(SPARSE_STATUS_SUCCESS!=status && SPARSE_STATUS_NOT_INITIALIZED!=status){return -1;}
    /*printf("c update ab matrix\n");
    for(int j=0; j<5; j++){
    for(int i=0; i<5; i++){printf("(%.3f,%.3f)",ab[j][i].real, ab[j][i].imag);}printf("\n");}printf("\n");*/
    cblas_zcopy (5*(n_max+1), ab, 1, &(ab_LU[2][0]), 1);
    LAPACKE_zgbtrf (LAPACK_ROW_MAJOR, n_max+1 , n_max+1 , 2 , 2 , &(ab_LU[0][0]) , n_max+1 , LU_ipiv );

    mkl_sparse_z_add (SPARSE_OPERATION_NON_TRANSPOSE, world.x_hat, {-omega*_force_cache, 0.}, world.harmonic_Hamil, &Hamil_temp[0]);
    mkl_sparse_spmm (SPARSE_OPERATION_NON_TRANSPOSE, Hamil_temp[0], Hamil_temp[0], &Hamil_temp[1]); // H^2
    mkl_sparse_spmm (SPARSE_OPERATION_NON_TRANSPOSE, Hamil_temp[1], Hamil_temp[0], &Hamil_temp[2]); // H^3
    mkl_sparse_spmm (SPARSE_OPERATION_NON_TRANSPOSE, Hamil_temp[1], Hamil_temp[1], &Hamil_temp[3]); // H^4
    mkl_sparse_spmm (SPARSE_OPERATION_NON_TRANSPOSE, Hamil_temp[1], Hamil_temp[2], &Hamil_temp[4]); // H^5
    // the following is the correction factor 3-4-5-6 for an implicit method
    mkl_sparse_z_add (SPARSE_OPERATION_NON_TRANSPOSE, Hamil_temp[1], {_dt_cache*_dt_cache*_dt_cache / 12., 0.}, world.empty_matrix, &Hamil_temp[5]);
    mkl_sparse_z_add (SPARSE_OPERATION_NON_TRANSPOSE, Hamil_temp[2], {0., - _dt_cache*_dt_cache*_dt_cache*_dt_cache / 24.}, Hamil_temp[5], &Hamil_temp[6]);
    mkl_sparse_z_add (SPARSE_OPERATION_NON_TRANSPOSE, Hamil_temp[3], {- _dt_cache*_dt_cache*_dt_cache*_dt_cache*_dt_cache / 80.,0.} \
                                                                    , Hamil_temp[6], &Hamil_temp[7]);
    mkl_sparse_z_add (SPARSE_OPERATION_NON_TRANSPOSE, Hamil_temp[4], {0., _dt_cache*_dt_cache*_dt_cache*_dt_cache*_dt_cache*_dt_cache / 360.} \
                                                                    , Hamil_temp[7], &Hamiltonian_addup_factor);
    // the following is the correction factor 3-4-5 for an explicit method
    /*mkl_sparse_z_add (SPARSE_OPERATION_NON_TRANSPOSE, Hamil_temp[1], {-_dt_cache*_dt_cache*_dt_cache / 6., 0.}, world.empty_matrix, &Hamil_temp[4]);
    mkl_sparse_z_add (SPARSE_OPERATION_NON_TRANSPOSE, Hamil_temp[2], {0., + _dt_cache*_dt_cache*_dt_cache*_dt_cache / 24.}, Hamil_temp[4], &Hamil_temp[5]);
    mkl_sparse_z_add (SPARSE_OPERATION_NON_TRANSPOSE, Hamil_temp[3], {_dt_cache*_dt_cache*_dt_cache*_dt_cache*_dt_cache / 120.,0.} \
                                                                    , Hamil_temp[5], &Hamiltonian_addup_factor);*/

    mkl_sparse_set_mv_hint (Hamiltonian_addup_factor, SPARSE_OPERATION_NON_TRANSPOSE, {SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT}, 80);
    mkl_sparse_order(Hamiltonian_addup_factor); mkl_sparse_optimize (Hamiltonian_addup_factor);
    // avoid memory leak
    for(int i=0;i<8;i++){if(SPARSE_STATUS_SUCCESS!=mkl_sparse_destroy (Hamil_temp[i])){return -1;}} 
    
    return 0;
}
// carry out one step calculation
static void __attribute__((hot)) D1(MKL_Complex16* state, const double &force, const double &gamma, MKL_Complex16* result, MKL_Complex16* relative_state, double x_avg){
    MKL_Complex16 x_hat_state[n_max+1];
    compute_x_hat_state(1., state, 0., x_hat_state);
    /*if (x_avg==0.){    
        MKL_Complex16 temp;
        cblas_zdotc_sub (n_max+1, state, 1, x_hat_state, 1, &temp);
        x_avg = temp.real;
    }*/
    // compute the relative_state 
    cblas_zcopy (n_max+1, x_hat_state, 1, relative_state, 1); // x_hat_state = relative_state = x|psi>
    cblas_daxpy (2*n_max+2, -x_avg, (double*)state, 1, (double*)relative_state, 1); // relative_state = (x-<x>)|psi>
    // compute the deterministic Hamiltonian term, stored in x_hat_state
    mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, {0.,-1.}, world.harmonic_Hamil, descr, state, {0.,+1.*omega*force}, x_hat_state);
    cblas_zcopy (n_max+1, x_hat_state, 1, result, 1); // store the result: x_hat_state = result = -1.j(H|psi>-oemga*F*x|psi>)
    // compute the deterministic squeezing term , stored in x_hat_state
    cblas_zcopy (n_max+1, relative_state, 1, x_hat_state, 1); // x_hat_state = (x-<x>)|psi>
    compute_x_hat_state(1., relative_state, -x_avg, x_hat_state); // x_hat_state = (x(x-<x>)|psi>-<x>(x-<x>)|psi>) = (x-<x>)^2 |psi>
    // compute the result. The coefficient of the first term (currently in result) is already handled.
    cblas_daxpy (2*n_max+2, -gamma/4., (double*)x_hat_state, 1, (double*)result, 1); // result = -1.j(H-oemga*F*x)|psi> -gamma/4.(x-<x>)^2 |psi>
}

// The "ImRe" is a slight abuse of notation. The "Im" part refers to the Hamiltonian part that can be evaluated with implicit methods
static void __attribute__((hot)) D1ImRe(MKL_Complex16* state, const double &force, const double &gamma, MKL_Complex16* resultIm, MKL_Complex16* resultRe, MKL_Complex16* relative_state){
    // resultIm can replace the role of x_hat_state
    compute_x_hat_state(1., state, 0., resultIm);
    double x_avg;
    MKL_Complex16 temp;
    cblas_zdotc_sub (n_max+1, state, 1, resultIm, 1, &temp);
    x_avg = temp.real;

    // compute the relative_state
    cblas_zcopy (n_max+1, resultIm, 1, relative_state, 1);
    cblas_daxpy (2*n_max+2, -x_avg, (double*)state, 1, (double*)relative_state, 1);
    // compute the deterministic Hamiltonian term, stored in resultIm
    mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, {0.,-1,}, world.harmonic_Hamil, descr, state,{0.,+1.*omega*force}, resultIm);
    // compute the deterministic squeezing term , stored in resultRe
    cblas_zcopy (n_max+1, relative_state, 1, resultRe, 1);
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
        cblas_zdotc_sub (n_max+1, state, 1, relative_state_and_result, 1, &temp);
        x_avg = temp.real;
        // compute (x-<x>)|psi>
        cblas_daxpy (2*n_max+2, -x_avg, (double*)state, 1, (double*)relative_state_and_result, 1);}
    cblas_zdscal (n_max+1, sqrt(gamma/2.), relative_state_and_result, 1);
}

static VSLStreamStatePtr stream;

static void go_one_step(MKL_Complex16* psi, double dt, double force, double _gamma, double* q_output, double* x_mean_output);
static void check_boundary_error(MKL_Complex16* psi, int* Fail);

static int check_type(PyArrayObject* state){
    if(PyArray_NDIM(state)!=1){
        PyErr_SetString(PyExc_ValueError, "The input array is not one-dimensional");
        return -1;
    }else{
    if(PyArray_SHAPE(state)[0]!=n_max+1){
        PyErr_SetString(PyExc_ValueError, ("The input array does not match the required size "+std::to_string(n_max+1)).c_str());
        return -1;
    }else{
        PyArray_Descr* descr = PyArray_DESCR(state);
        if(descr->type_num!=NPY_COMPLEX128){
            PyErr_SetString(PyExc_ValueError, "The input array does not match the required datatype: Complex128");
            return -1;
        }        
        return 0;
    }
    }
}
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
        if(0!=reset_ab(change_t)){return NULL;}
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
    if (!PyArg_ParseTuple(args, "Oddd", &temp, &dt, &force, &_gamma))
        return NULL;
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
        if(0!=reset_ab(change_t)){return NULL;}
    }
    double q=0., x_mean=0.;
    for(int i=0; i<10; i++){go_one_step(psi, dt, force, _gamma, &q, &x_mean);}

    int Fail=0; // compute the norm of a single value in psi
    check_boundary_error(psi, &Fail);
    return Py_BuildValue("ddi", q, x_mean, Fail); 
}
static void check_boundary_error(MKL_Complex16* psi, int* Fail){
    if ( cblas_dznrm2(5, &(psi[n_max+1-5]), 1)>1.e-3){
        *Fail = 1; 
        }
}
/*
static void sum_them_all(MKL_Complex16* state, double dt, double dW, double dZ, MKL_Complex16* D2_state, MKL_Complex16* D1_Y_plusIm_substract_D1_Y_minusIm, MKL_Complex16* D1_Y_plusRe, \
    MKL_Complex16* D1_Y_minusRe, MKL_Complex16*  D1_state, MKL_Complex16* D2_Y_plus, MKL_Complex16* D2_Y_minus, MKL_Complex16* D2_Phi_plus, MKL_Complex16* D2_Phi_minus);*/
static void simple_sum_up(double* state, double dt, double dW, double dZ, double* D2_state, double* D1_Y_plusIm_substract_D1_Y_minusIm, double* D1_Y_plusRe, \
    double* D1_Y_minusRe, double*  D1_state, double* D2_Y_plus, double* D2_Y_minus, double* D2_Phi_plus, double* D2_Phi_minus, double* D1_Y_plusIm_sum_D1_Y_minusIm);
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
    MKL_Complex16 D1_state[n_max+1], D2_state[n_max+1];
    MKL_Complex16 D2_state_drt[n_max+1]={{0.,0.}};
    D1(psi, force, _gamma, D1_state, D2_state, x_mean);
    D2(psi, _gamma, D2_state, true);

    cblas_daxpy (2*n_max+2, sqrt(dt), (double*)D2_state, 1, (double*)D2_state_drt, 1);

    // initialize Y as a 1st order step forward from psi
    MKL_Complex16 Y_plus[n_max+1], Y_minus[n_max+1];
    cblas_zcopy (n_max+1, psi, 1, Y_plus, 1); 

    cblas_daxpy (2*n_max+2, dt, (double*)D1_state, 1, (double*)Y_plus , 1); cblas_zcopy (n_max+1, Y_plus, 1, Y_minus, 1);

    cblas_daxpy (2*n_max+2, 1., (double*)D2_state_drt, 1, (double*)Y_plus , 1); 
    cblas_daxpy (2*n_max+2,-1., (double*)D2_state_drt, 1, (double*)Y_minus, 1);
    MKL_Complex16  D1_Y_plusIm[n_max+1],  D1_Y_plusRe[n_max+1],  D2_Y_plus[n_max+1];
    MKL_Complex16 D1_Y_minusIm[n_max+1], D1_Y_minusRe[n_max+1], D2_Y_minus[n_max+1];
    D1ImRe(Y_plus, force, _gamma, D1_Y_plusIm, D1_Y_plusRe, D2_Y_plus);
    D1ImRe(Y_minus, force, _gamma, D1_Y_minusIm, D1_Y_minusRe, D2_Y_minus);
    D2(Y_plus,_gamma, D2_Y_plus, true); D2(Y_minus,_gamma, D2_Y_minus, true);
    
    // use the storage of D1_Y_plusIm as another variable name:
    MKL_Complex16* D1_Y_plusIm_substract_D1_Y_minusIm; D1_Y_plusIm_substract_D1_Y_minusIm=D1_Y_plusIm;
    MKL_Complex16 D1_Y_plusIm_sum_D1_Y_minusIm[n_max+1]; ////////////////////////////////////
    cblas_zcopy (n_max+1, (double*)D1_Y_plusIm, 1, (double*)D1_Y_plusIm_sum_D1_Y_minusIm, 1); ///////////////////////
    cblas_daxpy (2*n_max+2, 1., (double*)D1_Y_minusIm, 1, (double*)D1_Y_plusIm_sum_D1_Y_minusIm, 1); ///////////////////////
    cblas_daxpy (2*n_max+2, -1., (double*)D1_Y_minusIm, 1, (double*)D1_Y_plusIm_substract_D1_Y_minusIm, 1);
    // use the storage of Y_minus as another variable name: Phi_minus
    MKL_Complex16* Phi_minus; Phi_minus=Y_minus;
    cblas_zcopy (n_max+1, Y_plus, 1, Phi_minus, 1); 
    // Phi_minus = Y_plus - sqrt(dt) D2(Y_plus)
    cblas_daxpy (2*n_max+2, - sqrt(dt), (double*)D2_Y_plus, 1, (double*)Phi_minus, 1);
    // use the storage of Y_plus as Phi_plus
    // Phi_plus = Y_plus + sqrt(dt) D2(Y_plus)
    MKL_Complex16* Phi_plus; Phi_plus=Y_plus;
    cblas_daxpy (2*n_max+2, sqrt(dt), (double*)D2_Y_plus, 1, (double*)Phi_plus, 1);

    MKL_Complex16 D2_Phi_plus[n_max+1], D2_Phi_minus[n_max+1];
    D2(Phi_plus, _gamma, D2_Phi_plus, false); D2(Phi_minus, _gamma, D2_Phi_minus, false);

    // sum them all
    simple_sum_up((double*)psi, dt, dW, dZ, (double*)D2_state, (double*)D1_Y_plusIm_substract_D1_Y_minusIm, (double*)D1_Y_plusRe, (double*)D1_Y_minusRe, \
                    (double*)D1_state, (double*)D2_Y_plus, (double*)D2_Y_minus, (double*)D2_Phi_plus, (double*)D2_Phi_minus, (double*)D1_Y_plusIm_sum_D1_Y_minusIm);
   
    // implicitly solve
    LAPACKE_zgbtrs (LAPACK_ROW_MAJOR, 'N' , n_max+1 , 2 , 2 , 1 , &(ab_LU[0][0]) , n_max+1 , LU_ipiv , psi , 1 ); // * note that the number of sub/super diagonals is 2 
    normalize(psi);
}/*
static void __attribute__((hot)) sum_them_all(MKL_Complex16* state, double dt, double dW, double dZ, MKL_Complex16* D2_state, MKL_Complex16* D1_Y_plusIm_substract_D1_Y_minusIm, MKL_Complex16* D1_Y_plusRe, \
    MKL_Complex16* D1_Y_minusRe, MKL_Complex16*  D1_state, MKL_Complex16* D2_Y_plus, MKL_Complex16* D2_Y_minus, MKL_Complex16* D2_Phi_plus, MKL_Complex16* D2_Phi_minus){
    // pending: change all types to double[2*n_max+2]
    MKL_Complex16 D1_Y_plusRe_subtract_D1_Y_minusRe[n_max+1];
    cblas_zcopy (n_max+1, D1_Y_plusRe, 1, D1_Y_plusRe_subtract_D1_Y_minusRe, 1); 
    MKL_Complex16 a={-1.,0.}; cblas_zaxpy (n_max+1, &a, D1_Y_minusRe, 1, D1_Y_plusRe_subtract_D1_Y_minusRe, 1);

    MKL_Complex16* D1_Y_plusRe_add_D1_Y_minusRe; D1_Y_plusRe_add_D1_Y_minusRe=D1_Y_plusRe;
    a={1.,0.}; cblas_zaxpy (n_max+1, &a, D1_Y_minusRe, 1, D1_Y_plusRe_add_D1_Y_minusRe, 1);

    MKL_Complex16* term1; term1=D1_Y_plusRe_subtract_D1_Y_minusRe;
    a={1.,0.}; cblas_zaxpy (n_max+1, &a, D1_Y_plusIm_substract_D1_Y_minusIm, 1, term1, 1);
                
    MKL_Complex16* term2; term2=D1_Y_plusRe_add_D1_Y_minusRe;
    a={2.,0.}; cblas_zaxpy (n_max+1, &a, D1_state, 1, term2, 1);

    MKL_Complex16* D2_Phi_plus_subtract_D2_Phi_minus; D2_Phi_plus_subtract_D2_Phi_minus=D2_Phi_plus;
    a={-1.,0.}; cblas_zaxpy (n_max+1, &a, D2_Phi_minus, 1, D2_Phi_plus_subtract_D2_Phi_minus, 1);
    // use the memory of D2_Phi_minus:
    MKL_Complex16* D2_Y_plus_subtract_D2_Y_minus; D2_Y_plus_subtract_D2_Y_minus=D2_Phi_minus;
    cblas_zcopy (n_max+1, D2_Y_plus, 1, D2_Y_plus_subtract_D2_Y_minus, 1);
    a={-1.,0.}; cblas_zaxpy (n_max+1, &a, D2_Y_minus, 1, D2_Y_plus_subtract_D2_Y_minus, 1);
    MKL_Complex16* term3; term3=D2_Y_plus_subtract_D2_Y_minus;

    MKL_Complex16* term4; term4=D2_Y_plus;
    a={1.,0.}; cblas_zaxpy (n_max+1, &a, D2_Y_minus, 1, term4, 1);
    a={-2.,0.}; cblas_zaxpy (n_max+1, &a, D2_state, 1, term4, 1);

    MKL_Complex16* term5; term5=D2_Phi_plus_subtract_D2_Phi_minus;
    a={-1.,0.}; cblas_zaxpy (n_max+1, &a, D2_Y_plus_subtract_D2_Y_minus, 1, term5, 1);

    MKL_Complex16* term6; term6=D1_Y_plusIm_substract_D1_Y_minusIm;

    // the memory of D2_Y_minus is still not used:
    MKL_Complex16* term7; term7=D2_Y_minus;
    // note that D1_state has included a factor of -1.j ***
    mkl_sparse_z_mv (SPARSE_OPERATION_NON_TRANSPOSE, {1., 0.}, Hamiltonian_addup_factor, descr, D1_state, {0.,0.}, term7);

    // (assert?)
    // sum up
    a={dW                        , 0.}; cblas_zaxpy (n_max+1, &a, D2_state, 1, state, 1);
    a={(0.5/sqrt(dt))*dZ         , 0.}; cblas_zaxpy (n_max+1, &a, term1, 1, state, 1);
    a={0.25*dt                   , 0.}; cblas_zaxpy (n_max+1, &a, term2, 1, state, 1);
    a={(0.25/sqrt(dt))*(dW*dW-dt), 0.}; cblas_zaxpy (n_max+1, &a, term3, 1, state, 1);
    a={(0.5/dt)*(dW*dt-dZ)       , 0.}; cblas_zaxpy (n_max+1, &a, term4, 1, state, 1);
    a={(0.25/dt)*(dW*dW/3-dt)*dW , 0.}; cblas_zaxpy (n_max+1, &a, term5, 1, state, 1);
    a={ -0.25*sqrt(dt)*dW        , 0.}; cblas_zaxpy (n_max+1, &a, term6, 1, state, 1);
    a={ 1.                       , 0.}; cblas_zaxpy (n_max+1, &a, term7, 1, state, 1);
      state = state + D2_state_dW + 0.5/sqrt(dt)*dZ*(D1_Y_plusIm_substract_D1_Y_minusIm + D1_Y_plusRe-D1_Y_minusRe) + \
                0.25*dt*(D1_Y_plusRe+2*D1_state+D1_Y_minusRe) + \
                0.25/sqrt(dt)*(dW*dW-dt)*(D2_Y_plus - D2_Y_minus) + \
                0.5/dt*(dW*dt-dZ)*(D2_Y_plus + D2_Y_minus - 2*D2_state) + \
                0.25/dt*(dW*dW/3-dt)*dW*(D2_Phi_plus - D2_Phi_minus - D2_Y_plus + D2_Y_minus) \
              - 0.25*sqrt(dt)*dW*(D1_Y_plusIm_substract_D1_Y_minusIm) \
              + self.Hamiltonian_addup_factor.dot(D1_state)    
}*/
static void __attribute__((hot)) simple_sum_up(double* state, double dt, double dW, double dZ, double* D2_state, double* D1_Y_plusIm_substract_D1_Y_minusIm, double* D1_Y_plusRe, \
    double* D1_Y_minusRe, double*  D1_state, double* D2_Y_plus, double* D2_Y_minus, double* D2_Phi_plus, double* D2_Phi_minus, double* D1_Y_plusIm_sum_D1_Y_minusIm){

    MKL_Complex16 term7[n_max+1];
    // note that D1_state has included a factor of -1.j ***
    mkl_sparse_z_mv (SPARSE_OPERATION_NON_TRANSPOSE, {1., 0.}, Hamiltonian_addup_factor, descr, (MKL_Complex16*)D1_state, {0.,0.}, term7);
    double* dterm7;
    dterm7=(double*)term7;
    // summation for the implicit method
    for(int i=0;i<2*n_max+2;i++){
        state[i]+=D2_state[i]*dW+0.5/sqrt(dt)*dZ*(D1_Y_plusIm_substract_D1_Y_minusIm[i]+D1_Y_plusRe[i]-D1_Y_minusRe[i])+\
                0.25*dt*(D1_Y_plusRe[i]+2*D1_state[i]+D1_Y_minusRe[i]) + \
                0.25/sqrt(dt)*(dW*dW-dt)*(D2_Y_plus[i] - D2_Y_minus[i]) + \
                0.5/dt*(dW*dt-dZ)*(D2_Y_plus[i] + D2_Y_minus[i] - 2*D2_state[i]) + \
                0.25/dt*(dW*dW/3-dt)*dW*(D2_Phi_plus[i] - D2_Phi_minus[i] - D2_Y_plus[i] + D2_Y_minus[i]) \
              - 0.25*sqrt(dt)*dW*(D1_Y_plusIm_substract_D1_Y_minusIm[i]) \
              + dterm7[i];
    }
    // summation for the explicit method
    /*
    for(int i=0;i<2*n_max+2;i++){
        state[i]+=D2_state[i]*dW+0.5/sqrt(dt)*dZ*(D1_Y_plusIm_substract_D1_Y_minusIm[i]+D1_Y_plusRe[i]-D1_Y_minusRe[i])+\
                0.25*dt*(D1_Y_plusRe[i]+2*D1_state[i]+D1_Y_minusRe[i] + D1_Y_plusIm_sum_D1_Y_minusIm[i]) + \
                0.25/sqrt(dt)*(dW*dW-dt)*(D2_Y_plus[i] - D2_Y_minus[i]) + \
                0.5/dt*(dW*dt-dZ)*(D2_Y_plus[i] + D2_Y_minus[i] - 2*D2_state[i]) + \
                0.25/dt*(dW*dW/3-dt)*dW*(D2_Phi_plus[i] - D2_Phi_minus[i] - D2_Y_plus[i] + D2_Y_minus[i]); // \
              + dterm7[i];*/
}
static PyObject* set_seed(PyObject* self, PyObject *args){
    int seed;
    if (!PyArg_ParseTuple(args, "i", &seed)){printf("Parse fail.\n"); return NULL;}
    vslNewStream( &stream, VSL_BRNG_MT19937, seed );
    Py_RETURN_NONE;
}

static PyObject* check_settings(PyObject* self, PyObject *args){
    return Py_BuildValue("(id)", n_max, omega);
}

static PyObject* Hamiltonian_dot_psi(PyObject *self, PyObject *args){
    PyObject* temp;
    PyArrayObject* state;
    if (!PyArg_ParseTuple(args, "O", &temp))
        return NULL;
    PyArray_OutputConverter(temp, &state);
    if (!state) return NULL;
    // get the c array from numpy (complex128 is stored as successive float64 two by two)
    MKL_Complex16* psi;
    psi = (MKL_Complex16*) PyArray_GETPTR1(state, 0);
    MKL_Complex16 temp2[n_max+1];

    cblas_zcopy (n_max+1, psi, 1, temp2, 1);
    mkl_sparse_z_mv (SPARSE_OPERATION_NON_TRANSPOSE, {1., 0.}, world.harmonic_Hamil, {SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT}, temp2, {0.,0.}, psi);

    return Py_BuildValue("d", 0.); 
}

static PyObject* solve_ab(PyObject *self, PyObject *args){
    PyObject* temp;
    PyArrayObject* state;
    if (!PyArg_ParseTuple(args, "O", &temp))
        return NULL;
    PyArray_OutputConverter(temp, &state);
    if (!state) return NULL;
    // get the c array from numpy (complex128 is stored as successive float64 two by two)
    MKL_Complex16* psi;
    psi = (MKL_Complex16*) PyArray_GETPTR1(state, 0);
    LAPACKE_zgbtrs (LAPACK_ROW_MAJOR, 'N' , n_max+1 , 1 , 1 , 1 , &(ab_LU[0][0]) , n_max+1 , LU_ipiv , psi , 1 );

    return Py_BuildValue("d", 0.); 
}

static PyMethodDef methods[] = {
    {"step", (PyCFunction)step, METH_VARARGS,
     "Do one simulation step."},
    {"simulate_10_steps", (PyCFunction)simulate_10_steps, METH_VARARGS,
     "Do 10 simulation steps."},
    {"set_seed", (PyCFunction)set_seed, METH_VARARGS,
     "Initialize the random number generator with a seed."},
    {"check_settings", (PyCFunction)check_settings, METH_VARARGS,
     "test whether the imported C module responds and return (n_max,omega)."},
    {"x_expectation", (PyCFunction)x_expectation,METH_VARARGS,""},
    {"Hamiltonian_dot_psi", (PyCFunction)Hamiltonian_dot_psi,METH_VARARGS,""},
    {"solve_ab", (PyCFunction)solve_ab,METH_VARARGS,""},
    {NULL, NULL, 0, NULL}   /* sentinel */
};

static struct PyModuleDef simulationmodule = {
    PyModuleDef_HEAD_INIT,
    "simulation",
    NULL,
    -1,
    methods
};

extern "C"{

PyMODINIT_FUNC
__attribute__((externally_visible)) PyInit_simulation(void)
{
    //mkl_set_memory_limit (MKL_MEM_MCDRAM, 256); // mbytes
    if(world.init_status!=0){
        printf("initialization status: %d\n",world.init_status);
        PyErr_SetString(PyExc_RuntimeError, "Initialization Failure");
    }
    init_();
    return PyModule_Create(&simulationmodule);
}

}
