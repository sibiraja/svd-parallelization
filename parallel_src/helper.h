#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
#include <random>
// #include <immintrin.h>
#include <x86intrin.h>
#include <omp.h>

using namespace std;
using namespace std::chrono;

typedef std::vector<std::vector<double>> Matrix;

vector<double> wall_times(6, 0.0);

struct Eigenmodes
{
    std::vector<double> eigenvalues;
    Matrix eigenvectors;
};

struct SVD
{
    Matrix U;
    Matrix Sigma;
    Matrix V;
};

void print_vector(std::vector<double> v) {
    std::cout << "[";
    for (size_t i = 0; i < v.size(); i++) {
        std::cout << v[i];
        if (i < v.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

void print_matrix(Matrix A) {
    std::cout << "[";
    for (size_t i = 0; i < A.size(); i++) {
        std::cout << "[";
        for (size_t j = 0; j < A[0].size(); j++) {
            std::cout << A[i][j];
            if (j < A[0].size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]";
        if (i < A.size() - 1) {
            std::cout << ", " << std::endl;
        }
    }
    std::cout << "]" << std::endl;
}

// WORK: M^3 * 2 = 2M^3
// DATA TRANSFERS: M^3 * 8 * 3 = 24M^3
// OPERATIONAL INTENSITY FOR GEMM: 2M^3 / 24M^3 = O(1)

// Tiled implementation and cache-aware
Matrix gemm(Matrix a, Matrix b) {
    // int tile_size = 4;
    // Matrix c(n, std::vector<double>(n, 0.0));
    // for (int ii = 0; ii < n; ii += tile_size) {
    //     for (int kk = 0; kk < n; kk += tile_size) {
    //         for (int jj = 0; jj < n; jj += tile_size) {
    //             for (int i = ii; i < ii + tile_size; i++) {
    //                 for (int k = kk; k < kk + tile_size; k++) {
    //                     for (int j = jj; j < jj + tile_size; j++) {
    //                         c[i][j] = c[i][j] + a[i][k] * b[k][j];
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
    // return c;

    int rows_A = a.size();
    int cols_A = a[0].size();
    int rows_B = b.size();
    int cols_B = b[0].size();

    if (cols_A != rows_B) {
        throw std::invalid_argument("The number of columns in A must match the number of rows in B");
    }

    Matrix C(rows_A, std::vector<double>(cols_B, 0));

    int tile_size = 4;

    for (int ii = 0; ii < rows_A; ii += tile_size) {
        for (int kk = 0; kk < cols_A; kk += tile_size) {
            for (int jj = 0; jj < cols_B; jj += tile_size) {
                for (int i = ii; i < std::min(ii + tile_size, rows_A); ++i) {
                    for (int k = kk; k < std::min(kk + tile_size, cols_A); ++k) {
                        for (int j = jj; j < std::min(jj + tile_size, cols_B); ++j) {
                            C[i][j] += a[i][k] * b[k][j];
                        }
                    }
                }
            }
        }
    }

    return C;
}
// OLD GEMM
// Matrix gemm(Matrix A, Matrix B) {
//     int rows_A = A.size();
//     int cols_A = A[0].size();
//     int rows_B = B.size();
//     int cols_B = B[0].size();

//     if (cols_A != rows_B) {
//         throw std::invalid_argument("The number of columns in A must match the number of rows in B");
//     }

//     Matrix C(rows_A, std::vector<double>(cols_B, 0));

//     // M^2 ELEMENTS
//     for (int i = 0; i < rows_A; ++i) {
//         for (int j = 0; j < cols_B; ++j) {
//             // M TIMES
//             for (int k = 0; k < cols_A; ++k) {
//                 // 2 READS, 1 WRITE, 2 FLOPS/OPERATIONS
//                 C[i][j] += A[i][k] * B[k][j];
//             }
//         }
//     }

//     return C;
// }

// WORK: M * M * 2 = 2M^2
// DATA TRANSFERS: M * M * 3 = 3M^2
// OPERATIONAL INTENSITY FOR GEMV: WORK / DATA TRANSFERS = 2M^2 / 3M^2 = O(1)
// manual gemv function that returns a vector
// OLD GEMV
// std::vector<double> gemv(Matrix A, std::vector<double> v) {
//     int rows_A = A.size();
//     int cols_A = A[0].size();
//     int rows_v = v.size();

//     if (cols_A != rows_v) {
//         throw std::invalid_argument("The number of columns in A must match the number of rows in v");
//     }

//     std::vector<double> w(rows_A, 0.0);

//     // M TIMES
//     for (int i = 0; i < rows_A; ++i) {
//         // M TIMES
//         for (int k = 0; k < cols_A; ++k) {
//             // 2 READ, 1 WRITE, 2 FLOPS/OPERATIONS
//             w[i] += A[i][k] * v[k];
//         }
//     }

//     return w;
// }

// Helper function to perform horizontal addition of 256-bit wide registers
inline double horizontal_sum_pd(__m256d v) {
    __m128d vlow  = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1); // Extract the high-order 128 bits
    __m128d sum = _mm_add_pd(vlow, vhigh);       // Add the low and high 128-bit values

    // Complete the horizontal sum using scalar operations
    __m128d high64 = _mm_unpackhi_pd(sum, sum);
    return _mm_cvtsd_f64(_mm_add_sd(sum, high64));
}

// std::vector<double> gemv_intrinsics_parallel(Matrix A, std::vector<double> v) {
std::vector<double> gemv(Matrix A, std::vector<double> v) {
    int rows_A = A.size();
    int cols_A = A[0].size();
    int rows_v = v.size();

    if (cols_A != rows_v) {
        throw std::invalid_argument("The number of columns in A must match the number of rows in v");
    }

    std::vector<double> w(rows_A, 0.0);

    // M TIMES
    #pragma omp parallel for // Parallelize the outer loop
    for (int i = 0; i < rows_A; ++i) {
        __m256d w_i = _mm256_setzero_pd(); // Initialize a 256-bit wide register with zeros

        int k;
        // M TIMES
        for (k = 0; k + 3 < cols_A; k += 4) {
            __m256d a_ik = _mm256_loadu_pd(&A[i][k]); // Load 4 elements from A[i][k]
            __m256d v_k  = _mm256_loadu_pd(&v[k]);     // Load 4 elements from v[k]
            __m256d a_ik_times_v_k = _mm256_mul_pd(a_ik, v_k); // Compute the product of a_ik and v_k
            w_i = _mm256_add_pd(w_i, a_ik_times_v_k); // Add the result to w_i
        }

        // Handle the tail elements
        double tail_sum = 0.0;
        for (; k < cols_A; ++k) {
            tail_sum += A[i][k] * v[k];
        }

        // Horizontal sum of the 256-bit wide register
        w[i] = horizontal_sum_pd(w_i) + tail_sum;
    }

    return w;
}

// WORK: M * (1 + 1) = 2M
// DATA TRANSFERS: (M * 2 * 8) + 1 * 8 = 16M + 8
// OPERATIONAL INTENSITY: WORK / DATA TRANSFERS = 2M / (16M + 8) = O (1)
// dot product function
// OLD DOT
// double dot(std::vector<double> v, std::vector<double> w) {
//     int rows_v = v.size();
//     int rows_w = w.size();

//     if (rows_v != rows_w) {
//         throw std::invalid_argument("The number of rows in v must match the number of rows in w");
//     }

//     double sum = 0;
//     // M TIMES
//     for (int i = 0; i < rows_v; ++i) {
//         // 2 READS, 0 WRITES (SUM IS IN REGISTER)
//         // 1 MULTIPLICATION, 1 ADDITION
//         sum += v[i] * w[i];
//     }
//     // 1 WRITE

//     return sum;
// }

// double dot_parallel_intrinsics(std::vector<double> v, std::vector<double> w) {
double dot(std::vector<double> v, std::vector<double> w) {
    int rows_v = v.size();
    int rows_w = w.size();

    if (rows_v != rows_w) {
        throw std::invalid_argument("The number of rows in v must match the number of rows in w");
    }

    double sum = 0.0;

    #pragma omp parallel
    {
        // Initialize a 256-bit wide register with zeros
        __m256d sum_vec = _mm256_setzero_pd();
        
        // M TIMES
        #pragma omp for nowait
        for (int i = 0; i < rows_v - 3; i += 4) {
            __m256d v_i = _mm256_loadu_pd(&v[i]);
            __m256d w_i = _mm256_loadu_pd(&w[i]);
            __m256d mul_result = _mm256_mul_pd(v_i, w_i);
            sum_vec = _mm256_add_pd(sum_vec, mul_result);
        }

        // Horizontal sum of the 256-bit wide register
        double partial_sum = horizontal_sum_pd(sum_vec);

        #pragma omp atomic
        sum += partial_sum;
    }

    // Handle the tail elements
    for (int i = (rows_v / 4) * 4; i < rows_v; ++i) {
        sum += v[i] * w[i];
    }

    return sum;
}

// WORK: 2M + 1
// DATA TRANSFERS: (16M + 8) = 16M + 8
// OPERATIONAL INTENSITY FOR VECTOR NORM: WORK / DATA TRANSFERS = (2M + 1) / (16M + 8) = O (1)
// vector norm function
double vector_norm(std::vector<double> v) {
    // OPERATIONAL INTENSITY FOR DOT: WORK / DATA TRANSFERS = 2M / (16M + 8) = O (1)
    double sum = dot(v, v);

    // 0 READ AND 0 WRITE, AND 1 FLOP/OPERATION
    sum = sqrt(sum);
    return sum;
}


// WORK: (2M + 1) + (1 * M) = 3M + 1
// DATA TRANSFERS: (16M + 8)  + (2 * M * 8) = 32M + 8
// OPERATIONAL INTENSITY FOR NORMALIZATION: (3M + 1) / (32M + 8) = O(1)

// vector normalization function
// OLD NORMALIZE
// std::vector<double> normalize(std::vector<double> v) {
//     int rows_v = v.size();
//     std::vector<double> w(rows_v, 0.0);

//     // OPERATIONAL INTENSITY FOR VECTOR NORM: WORK / DATA TRANSFERS = (2M + 1) / (16M + 8) = O (1)
//     double norm = vector_norm(v);

//     // M TIMES
//     for (int i = 0; i < rows_v; ++i) {
//         // 1 READ, 1 WRITE, 1 FLOP
//         w[i] = v[i] / norm;
//     }

//     return w;
// }

// std::vector<double> normalize_parallel_intrinsics(std::vector<double> v) {
std::vector<double> normalize(std::vector<double> v) {
    int rows_v = v.size();
    std::vector<double> w(rows_v, 0.0);

    // OPERATIONAL INTENSITY FOR VECTOR NORM: WORK / DATA TRANSFERS = (2M + 1) / (16M + 8) = O (1)
    double norm = vector_norm(v);

    // M TIMES
    int i = 0;
    #pragma omp parallel for
    for (i = 0; i < rows_v - 3; i += 4) {
        __m256d v_i = _mm256_loadu_pd(&v[i]);
        __m256d norm_vec = _mm256_set1_pd(norm);         // Broadcast norm to all elements in the 256-bit register
        __m256d div_result = _mm256_div_pd(v_i, norm_vec); // Divide each element in v_i by norm
        _mm256_storeu_pd(&w[i], div_result);            // Store the result in w
    }

    // Handle the tail elements
    for (; i < rows_v; ++i) {
        w[i] = v[i] / norm;
    }

    return w;
}

// WORK: 1 * M = M
// DATA TRANSFER: M * 2 * 8 = 16M
// OPERATIONAL INTENSITY FOR SCALAR VECTOR MULT: M / 16M = O(1)

// scalar vector multiplication function
// OLD SCALAR_VECTOR_MULT
// std::vector<double> scalar_vector_mult(double a, std::vector<double> v) {
//     int rows_v = v.size();
//     std::vector<double> w(rows_v, 0.0);

//     // M TIMES
//     for (int i = 0; i < rows_v; ++i) {
//         // 1 READ, 1 WRITE, 1 FLOP
//         w[i] = a * v[i];
//     }

//     return w;
// }

// std::vector<double> scalar_vector_mult_parallel_intrinsics(double a, std::vector<double> v) {
std::vector<double> scalar_vector_mult(double a, std::vector<double> v) {
    int rows_v = v.size();
    std::vector<double> w(rows_v, 0.0);

    // M TIMES
    int i = 0;
    #pragma omp parallel for
    for (i = 0; i < rows_v - 3; i += 4) {
        __m256d v_i = _mm256_loadu_pd(&v[i]);
        __m256d a_vec = _mm256_set1_pd(a);              // Broadcast a to all elements in the 256-bit register
        __m256d mul_result = _mm256_mul_pd(a_vec, v_i); // Multiply each element in v_i by a
        _mm256_storeu_pd(&w[i], mul_result);            // Store the result in w
    }

    // Handle the tail elements
    for (; i < rows_v; ++i) {
        w[i] = a * v[i];
    }

    return w;
}

// WORK: 0 FLOPS
// DATA TRANSFERS: 2M^2 * 8 = 16M^2
// OPERATIONAL INTENSITY FOR TRANSPOSE: 0 / 16M^2 = O(1)

// matrix transpose function
// OLD TRANSPOSE
// Matrix transpose(Matrix A) {
//     int rows_A = A.size();
//     int cols_A = A[0].size();

//     Matrix B(cols_A, std::vector<double>(rows_A, 0.0));


//     // M TIMES
//     for (int i = 0; i < rows_A; ++i) {
//         // M TIMES
//         for (int j = 0; j < cols_A; ++j) {
//             // 1 READ, 1 WRITE
//             B[j][i] = A[i][j];
//         }
//     }

//     return B;
// }

// Matrix transpose_parallel_intrinsics(Matrix A) {
Matrix transpose(Matrix A) {
    int rows_A = A.size();
    int cols_A = A[0].size();

    Matrix B(cols_A, std::vector<double>(rows_A, 0.0));

    // M TIMES
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < (rows_A / 4) * 4; i += 4) {
        // M TIMES
        for (int j = 0; j < (cols_A / 4) * 4; j += 4) {
            // Load 4x4 block from A
            __m256d a0 = _mm256_loadu_pd(&A[i + 0][j]);
            __m256d a1 = _mm256_loadu_pd(&A[i + 1][j]);
            __m256d a2 = _mm256_loadu_pd(&A[i + 2][j]);
            __m256d a3 = _mm256_loadu_pd(&A[i + 3][j]);

            // Transpose the 4x4 block
            __m256d t0 = _mm256_shuffle_pd(a0, a1, 0x0);
            __m256d t1 = _mm256_shuffle_pd(a0, a1, 0xF);
            __m256d t2 = _mm256_shuffle_pd(a2, a3, 0x0);
            __m256d t3 = _mm256_shuffle_pd(a2, a3, 0xF);

            __m256d b0 = _mm256_permute2f128_pd(t0, t2, 0x20);
            __m256d b1 = _mm256_permute2f128_pd(t1, t3, 0x20);
            __m256d b2 = _mm256_permute2f128_pd(t0, t2, 0x31);
            __m256d b3 = _mm256_permute2f128_pd(t1, t3, 0x31);

            // Store the transposed 4x4 block in B
            _mm256_storeu_pd(&B[j + 0][i], b0);
            _mm256_storeu_pd(&B[j + 1][i], b1);
            _mm256_storeu_pd(&B[j + 2][i], b2);
            _mm256_storeu_pd(&B[j + 3][i], b3);
        }
    }

    // Handle the tail elements
    for (int i = 0; i < rows_A; ++i) {
        for (int j = (cols_A / 4) * 4; j < cols_A; ++j) {
            B[j][i] = A[i][j];
        }
    }

    for (int i = (rows_A / 4) * 4; i < rows_A; ++i) {
        for (int j = 0; j < (cols_A / 4) * 4; ++j) {
            B[j][i] = A[i][j];
        }
    }

    return B;
}

// matrix-matrix subtraction
// OLD MATRIX_SUBTRACTION
// Matrix matrix_subtraction(Matrix A, Matrix B) {
//     int rows_A = A.size();
//     int cols_A = A[0].size();
//     int rows_B = B.size();
//     int cols_B = B[0].size();

//     if (rows_A != rows_B) {
//         throw std::invalid_argument("The number of rows in A must match the number of rows in B");
//     }
//     if (cols_A != cols_B) {
//         throw std::invalid_argument("The number of columns in A must match the number of columns in B");
//     }
    
//     Matrix C(rows_A, std::vector<double>(cols_A, 0));
//     for (int i = 0; i < rows_A; ++i) {
//         for (int j = 0; j < cols_A; ++j) {
//             C[i][j] += A[i][j] - B[i][j]; 
//         }
//     }
//     return C;
// }

// Matrix matrix_subtraction_parallel_intrinsics(Matrix A, Matrix B) {
Matrix matrix_subtraction(Matrix A, Matrix B) {
    int rows_A = A.size();
    int cols_A = A[0].size();
    int rows_B = B.size();
    int cols_B = B[0].size();

    if (rows_A != rows_B) {
        throw std::invalid_argument("The number of rows in A must match the number of rows in B");
    }
    if (cols_A != cols_B) {
        throw std::invalid_argument("The number of columns in A must match the number of columns in B");
    }

    Matrix C(rows_A, std::vector<double>(cols_A, 0));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows_A; ++i) {
        for (int j = 0; j < (cols_A / 4) * 4; j += 4) {
            __m256d a_vec = _mm256_loadu_pd(&A[i][j]);
            __m256d b_vec = _mm256_loadu_pd(&B[i][j]);
            __m256d c_vec = _mm256_sub_pd(a_vec, b_vec);
            _mm256_storeu_pd(&C[i][j], c_vec);
        }
    }

    // Handle the tail elements
    for (int i = 0; i < rows_A; ++i) {
        for (int j = (cols_A / 4) * 4; j < cols_A; ++j) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }

    return C;
}

// Calculating A^T * A
Matrix AtA(Matrix A) {
    Matrix B = transpose(A);
    Matrix C = gemm(B, A);
    return C;
}


Matrix vector_to_matrix(std::vector<double> v) {
    Matrix A(1, v);
    return transpose(A);
}


void assert_vectors_equal(const std::vector<double>& vec1, const std::vector<double>& vec2, double tolerance) {
    if (vec1.size() != vec2.size()) {
        throw std::runtime_error("The vectors have different sizes.");
    }

    for (size_t i = 0; i < vec1.size(); ++i) {
        if (std::abs(vec1[i] - vec2[i]) > tolerance) {
            throw std::runtime_error("The vectors are not equal within the specified tolerance.");
        }
    }
}


void assert_matrices_equal(const Matrix& mat1, const Matrix& mat2, double tolerance) {
    if (mat1.size() != mat2.size()) {
        throw std::runtime_error("The matrices have different numbers of rows.");
    }

    for (size_t i = 0; i < mat1.size(); ++i) {
        if (mat1[i].size() != mat2[i].size()) {
            throw std::runtime_error("The matrices have different numbers of columns.");
        }

        for (size_t j = 0; j < mat1[i].size(); ++j) {
            if (std::abs(mat1[i][j] - mat2[i][j]) > tolerance) {
                throw std::runtime_error("The matrices are not equal within the specified tolerance.");
            }
        }
    }
}
