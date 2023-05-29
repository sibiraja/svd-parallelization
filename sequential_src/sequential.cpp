#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
#include <random>

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

// manual gemm function
Matrix gemm(Matrix A, Matrix B) {
    int rows_A = A.size();
    int cols_A = A[0].size();
    int rows_B = B.size();
    int cols_B = B[0].size();

    if (cols_A != rows_B) {
        throw std::invalid_argument("The number of columns in A must match the number of rows in B");
    }

    Matrix C(rows_A, std::vector<double>(cols_B, 0));

    for (int i = 0; i < rows_A; ++i) {
        for (int j = 0; j < cols_B; ++j) {
            for (int k = 0; k < cols_A; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

// manual gemv function that returns a vector
std::vector<double> gemv(Matrix A, std::vector<double> v) {
    int rows_A = A.size();
    int cols_A = A[0].size();
    int rows_v = v.size();

    if (cols_A != rows_v) {
        throw std::invalid_argument("The number of columns in A must match the number of rows in v");
    }

    std::vector<double> w(rows_A, 0.0);

    for (int i = 0; i < rows_A; ++i) {
        for (int k = 0; k < cols_A; ++k) {
            w[i] += A[i][k] * v[k];
        }
    }

    return w;
}

// dot product function
double dot(std::vector<double> v, std::vector<double> w) {
    int rows_v = v.size();
    int rows_w = w.size();

    if (rows_v != rows_w) {
        throw std::invalid_argument("The number of rows in v must match the number of rows in w");
    }

    double sum = 0;
    for (int i = 0; i < rows_v; ++i) {
        sum += v[i] * w[i];
    }

    return sum;
}

// vector norm function
double vector_norm(std::vector<double> v) {
    double sum = dot(v, v);
    sum = sqrt(sum);
    return sum;
}

// vector normalization function
std::vector<double> normalize(std::vector<double> v) {
    int rows_v = v.size();
    std::vector<double> w(rows_v, 0.0);

    double norm = vector_norm(v);

    for (int i = 0; i < rows_v; ++i) {
        w[i] = v[i] / norm;
    }

    return w;
}

// scalar vector multiplication function
std::vector<double> scalar_vector_mult(double a, std::vector<double> v) {
    int rows_v = v.size();
    std::vector<double> w(rows_v, 0.0);

    for (int i = 0; i < rows_v; ++i) {
        w[i] = a * v[i];
    }

    return w;
}

// matrix transpose function
Matrix transpose(Matrix A) {
    int rows_A = A.size();
    int cols_A = A[0].size();

    Matrix B(cols_A, std::vector<double>(rows_A, 0.0));

    for (int i = 0; i < rows_A; ++i) {
        for (int j = 0; j < cols_A; ++j) {
            B[j][i] = A[i][j];
        }
    }

    return B;
}

// matrix-matrix subtraction
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
    for (int i = 0; i < rows_A; ++i) {
        for (int j = 0; j < cols_A; ++j) {
            C[i][j] += A[i][j] - B[i][j]; 
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

// Power Method to find Eigenvalues
double power_method(Matrix A) {
    int N = A.size();
    std::vector<double> q0(N, 0.0); // temporary eigenvector 1
    std::vector<double> q1(N, 0.0); // temporary eigenvector 2

    // initial condition
    q0[0] = 1.0;

    // Power method
    constexpr double tol = 1.0e-12;
    double lambda0 = HUGE_VAL; // temporary eigenvalue 1
    double lambda1 = 0.0;      // temporary eigenvalue 2 (final result)
    // size_t iter = 0;           // iteration index k

    q1 = gemv(A, q0);
    while (true)
    {
        // ++iter;
        q1 = normalize(q1);
        q0 = gemv(A, q1);
        lambda1 = dot(q1, q0);
        if (std::abs(lambda1 - lambda0) < tol)
        {
            break;
        }
        lambda0 = lambda1;
        q1.swap(q0);
    }

    return lambda1;
}

Matrix deflate_matrix(Matrix A, double lambda1, std::vector<double> e) {
    std::vector<double> v = scalar_vector_mult(lambda1, e);
    Matrix temp = gemm(vector_to_matrix(v), transpose(vector_to_matrix(e)));
    
    Matrix B = matrix_subtraction(A, temp);
    return B;
}

// Define a function to perform Gauss-Jordan elimination
vector<vector<double>> gaussjordan(vector<vector<double>> A) {
    int n = A.size();

    int i, j, k;
    double d;

    //Swapping rows for row-echelon form
    for(i = 0; i  < n; i++) {                   
        for(j = i + 1; j < n; j++){
            if(abs(A[i][i]) < abs(A[j][i])){
                for(k = 0; k < n; k++){
                    A[i][k] = A[i][k] + A[j][k];
                    A[j][k] = A[i][k] - A[j][k];
                    A[i][k] = A[i][k] - A[j][k];
                }
            }
        }
    }

    // Gauss Jordan Elimination
    for(i = 0; i < n - 1; i++){
        if (A[i][i] != 0){
            for(j = i + 1; j < n; j++){
                d = A[j][i] / A[i][i];
                for(k = 0 ; k < n; k++){
                    A[j][k] = A[j][k] - d * A[i][k];
                }
            }
        }
    }

    // Make diagonal into ones (Optional, may be deleted)
    // for (i = 0; i < n; i++) {
    //     d = A[i][i];
    //     for (j = i; j < n; j++) {
    //         A[i][j] = A[i][j] / d;
    //     }
    // }

    return A;
}


// Calculating U = A*V*(Sigma^-1) : The gauss jordan elimination method and the back-substitution method are what consist of calculating the eigenvectors (V) 
vector<vector<double>> calculate_U(Matrix A, Matrix V, Matrix Sigma) {
    int M = A.size();
    int N = A[0].size();
    vector<vector<double>> U(M, vector<double>(M, 0));
    vector<vector<double>> Sigma_inv(N, vector<double>(M, 0));
    for (int i = 0; i < N; i++) {
        Sigma_inv[i][i] = 1.0 / Sigma[i][i];
    }

    vector<vector<double>> temp = gemm(A, V);
    U = gemm(temp, Sigma_inv);

    Matrix Utemp = transpose(U);
    if (M > N) {
        for (int i = N; i < M; ++i) {
            std::vector<double> qi(M, 1.0);
            for (int j = 0; j < i; ++j) {
                std::vector<double> temp = scalar_vector_mult(dot(Utemp[j], qi),Utemp[j]);
                for (int k = 0; k < M; ++k) {
                    qi[k] = qi[k] - temp[k];
                }
            }
            qi = normalize(qi);
            Utemp[i] = qi;
        }
    }
    U = transpose(Utemp);
    return U;
}

vector<double> back_substitution(vector<vector<double>> A) {
    int n = A.size();
    vector<double> x(n, 0);
    x[n-1] = 1;  // Assume the last entry is 1

    for (int i = n - 2; i >= 0; i--) {
        double sum = 0;
        for (int j = i + 1; j < n; j++) {
            sum += A[i][j] * x[j];
        }
        x[i] = -sum / A[i][i];
    }
    return x;
}

// Singular value calculation (Sigma)
vector<double> sqrt_vector(vector<double> v) {
    vector<double> w(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        w[i] = sqrt(v[i]);
    }
    return w;
}

std::vector<double> find_eigenvector(Matrix A, double lambda1) {
    int size = A.size();
    Matrix lambdaI(size, std::vector<double>(size, 0));
    for (int i = 0; i < size; ++i) {
        lambdaI[i][i] += lambda1;
    }
    Matrix newA = matrix_subtraction(A,lambdaI);
    auto t2 = steady_clock::now();
    Matrix reduced_A = gaussjordan(newA);
    auto t3 = steady_clock::now();
    wall_times[1] += duration_cast<microseconds>(t3 - t2).count();
    auto t4 = steady_clock::now();
    std::vector<double> v = back_substitution(reduced_A);
    auto t5 = steady_clock::now();
    wall_times[2] += duration_cast<microseconds>(t5 - t4).count();
    return normalize(v);
}

Eigenmodes calculate_eigenmodes(Matrix A) {
    int rows_A = A.size();
    Matrix B = A;

    Eigenmodes final_eigenmodes;
    std::vector<double> final_lambdas(rows_A, 0.0);
    Matrix final_eigenvectors(rows_A, std::vector<double>(rows_A, 0));
    for (int i = 0; i < rows_A; ++i) {
        auto t0 = steady_clock::now();
        double lambda1 = power_method(B);
        auto t1 = steady_clock::now();
        wall_times[0] += duration_cast<microseconds>(t1 - t0).count();
        std::vector<double> e = find_eigenvector(B, lambda1);
        auto t6 = steady_clock::now();
        B = deflate_matrix(B, lambda1, e);
        auto t7 = steady_clock::now();
        wall_times[3] += duration_cast<microseconds>(t7 - t6).count();
        final_lambdas[i] = lambda1;
        final_eigenvectors[i] = e;
    }
    final_eigenmodes = {final_lambdas, final_eigenvectors};
    return final_eigenmodes;
}


// function that sorts the eigenvalues and eigenvectors in descending order
Eigenmodes sort_eigenmodes(Eigenmodes eigenmodes) {
    int n = eigenmodes.eigenvalues.size();
    std::vector<double> eigenvalues = eigenmodes.eigenvalues;
    std::vector<std::vector<double>> eigenvectors = eigenmodes.eigenvectors;

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (eigenvalues[i] < eigenvalues[j]) {
                double temp = eigenvalues[i];
                eigenvalues[i] = eigenvalues[j];
                eigenvalues[j] = temp;

                std::vector<double> temp2 = eigenvectors[i];
                eigenvectors[i] = eigenvectors[j];
                eigenvectors[j] = temp2;
            }
        }
    }
    Eigenmodes sorted_eigenmodes = {eigenvalues, eigenvectors};
    return sorted_eigenmodes;
}

SVD calculate_SVD (Matrix A) {
    int M = A.size();
    int N = A[0].size();

    Matrix U(M, vector<double>(M, 0));
    Matrix Sigma(M, vector<double>(N, 0));
    Matrix V(N, vector<double>(N, 0));

    
    Matrix ATA = AtA(A);
    
    Eigenmodes all_eigens = calculate_eigenmodes(ATA);

    std::vector<double> singular_values = sqrt_vector(all_eigens.eigenvalues);
    for (int i = 0; i < N; ++i) {
        Sigma[i][i] = singular_values[i];
    }

    V = transpose(all_eigens.eigenvectors);

    auto t14 = steady_clock::now();
    U = calculate_U(A, V, Sigma);
    auto t15 = steady_clock::now();
    wall_times[4] += duration_cast<microseconds>(t15 - t14).count();

    SVD final_SVD = {U, Sigma, V};
    return final_SVD;
}

int main(int argc, char *argv[]) {
    if (argc != 7) {
        std::cout << "Usage: " << argv[0] << " <noprint> <smallest N> <largest N> <largest M> <step> <seed>\n";
        exit(1);
    }
    const int smallestN = atoi(argv[2]);
    const int largestN = atoi(argv[3]);
    const int largestM = atoi(argv[4]);
    const int step = atoi(argv[5]);
    const int seed = atoi(argv[6]);

    for (int n = smallestN; n <= largestN; n+=step) {
        for (int m = n; m <= largestM; m+=step) {
            const int M = m;
            const int N = n;

            Matrix A(M, vector<double>(N)); // declare the matrix
            
            std::default_random_engine rd(seed); // random seed every run
            // uniform distribution in [0, 1]
            std::uniform_real_distribution<double> u(0.0, 1.0);
            // fill the matrix with random values
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    double v = u(rd);
                    A[i][j] = v;
                }
            }

            wall_times = std::vector<double>(6,0.0);

            
            auto t8 = steady_clock::now();
            SVD result_SVD = calculate_SVD(A);
            auto t9 = steady_clock::now();
            wall_times[5] += duration_cast<microseconds>(t9 - t8).count();

            if (atoi(argv[1]) == 1) {
                std::cout << "Dimensions: " << M << " x " << N << "\n";
                std::cout << "U: \n";
                print_matrix(result_SVD.U);
                std::cout << "Sigma: \n";
                print_matrix(result_SVD.Sigma);
                std::cout << "V: \n";
                print_matrix(result_SVD.V);
                // Correctness test:
                std::cout << "Original A: \n";
                print_matrix(A);
                Matrix reconstructed = gemm(gemm(result_SVD.U, result_SVD.Sigma), transpose(result_SVD.V));
                std::cout << "Reconstructed: \n";
                print_matrix(reconstructed);
                std::cout << "power_method: " << wall_times[0] << " microseconds" << std::endl;
                std::cout << "gaussjordan: " << wall_times[1] << " microseconds" << std::endl;
                std::cout << "back_substitution: " << wall_times[2] << " microseconds" << std::endl;
                std::cout << "deflate_matrix: " << wall_times[3] << " microseconds" << std::endl;
                std::cout << "calculate_U: " << wall_times[4] << " microseconds" << std::endl;
                std::cout << "total calculate_svd: " << wall_times[5] << " microseconds" << "\n" 
                          << std::endl;
            }
            else {
                std::cout << M << " " << N << " " << wall_times[0] << " "
                        << wall_times[1] << " " << wall_times[2] << " "
                        << wall_times[3] << " " << wall_times[4] << " " << wall_times[5] << std::endl;
            }
        }
    }
    return 0;
}