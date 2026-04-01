#include "crossed_re.h"
#include <cmath>
#include <stdexcept>

namespace fastmlm {

// ============================================================================
// BlockDiagonalPreconditioner
// ============================================================================

void BlockDiagonalPreconditioner::compute(
    const SpMatd& A, const VectorXi& block_starts,
    const VectorXi& block_sizes)
{
    block_starts_ = block_starts;
    block_sizes_ = block_sizes;
    int n_blocks = block_sizes.size();

    block_factors_.resize(n_blocks);

    for (int b = 0; b < n_blocks; ++b) {
        int start = block_starts_[b];
        int size = block_sizes_[b];

        // Extract the diagonal block as a dense matrix
        MatrixXd block = MatrixXd::Zero(size, size);
        for (int j = 0; j < size; ++j) {
            int col = start + j;
            for (SpMatd::InnerIterator it(A, col); it; ++it) {
                int row = it.row() - start;
                if (row >= 0 && row < size) {
                    block(row, j) = it.value();
                }
            }
        }

        block_factors_[b].compute(block);
        if (block_factors_[b].info() != Eigen::Success) {
            // If a block fails, use identity for that block
            block = MatrixXd::Identity(size, size);
            block_factors_[b].compute(block);
        }
    }

    computed_ = true;
}

VectorXd BlockDiagonalPreconditioner::solve(const VectorXd& r) const {
    VectorXd z = VectorXd::Zero(r.size());
    int n_blocks = block_sizes_.size();

    for (int b = 0; b < n_blocks; ++b) {
        int start = block_starts_[b];
        int size = block_sizes_[b];

        z.segment(start, size) =
            block_factors_[b].solve(r.segment(start, size));
    }

    return z;
}

// ============================================================================
// PCGSolver
// ============================================================================

PCGSolver::Result PCGSolver::solve(
    const SpMatd& A, const VectorXd& b,
    const BlockDiagonalPreconditioner& precond,
    double tol, int maxiter)
{
    int n = b.size();
    Result result;
    result.converged = false;

    VectorXd x = VectorXd::Zero(n);
    VectorXd r = b;  // r = b - A*x, but x=0
    VectorXd z = precond.solve(r);
    VectorXd p = z;

    double rz = r.dot(z);
    double b_norm = b.norm();
    if (b_norm < 1e-15) {
        result.x = x;
        result.iterations = 0;
        result.residual_norm = 0.0;
        result.converged = true;
        return result;
    }

    for (int iter = 0; iter < maxiter; ++iter) {
        VectorXd Ap = A * p;
        double pAp = p.dot(Ap);

        if (std::abs(pAp) < 1e-30) {
            result.iterations = iter + 1;
            break;
        }

        double alpha = rz / pAp;
        x += alpha * p;
        r -= alpha * Ap;

        double r_norm = r.norm();
        result.residual_norm = r_norm / b_norm;

        if (result.residual_norm < tol) {
            result.x = x;
            result.iterations = iter + 1;
            result.converged = true;
            return result;
        }

        VectorXd z_new = precond.solve(r);
        double rz_new = r.dot(z_new);
        double beta = rz_new / rz;

        p = z_new + beta * p;
        z = z_new;
        rz = rz_new;
    }

    result.x = x;
    result.iterations = maxiter;
    return result;
}

MatrixXd PCGSolver::solve_multi(
    const SpMatd& A, const MatrixXd& B,
    const BlockDiagonalPreconditioner& precond,
    double tol, int maxiter)
{
    int n = B.rows();
    int nrhs = B.cols();
    MatrixXd X(n, nrhs);

    for (int j = 0; j < nrhs; ++j) {
        Result res = solve(A, B.col(j), precond, tol, maxiter);
        X.col(j) = res.x;
    }

    return X;
}

// ============================================================================
// StochasticLogDet
// ============================================================================

double StochasticLogDet::slq_logdet_probe(
    const SpMatd& A, const VectorXd& z, int lanczos_steps)
{
    // Lanczos iteration: build tridiagonal matrix T such that
    // eigenvalues of T approximate eigenvalues of A.
    int n = z.size();
    int m = std::min(lanczos_steps, n);

    VectorXd alpha_vec(m);  // diagonal of T
    VectorXd beta_vec(m);   // sub-diagonal of T

    VectorXd v = z / z.norm();
    VectorXd v_prev = VectorXd::Zero(n);
    double beta_prev = 0.0;

    for (int j = 0; j < m; ++j) {
        VectorXd w = A * v;

        double alpha = v.dot(w);
        alpha_vec[j] = alpha;

        w -= alpha * v;
        if (j > 0) {
            w -= beta_prev * v_prev;
        }

        // Re-orthogonalisation (partial, for numerical stability)
        double h = v.dot(w);
        w -= h * v;
        alpha_vec[j] += h;

        double beta = w.norm();
        beta_vec[j] = beta;

        if (beta < 1e-14) {
            // Lanczos breakdown — invariant subspace found
            m = j + 1;
            alpha_vec.conservativeResize(m);
            beta_vec.conservativeResize(m);
            break;
        }

        v_prev = v;
        v = w / beta;
        beta_prev = beta;
    }

    // Build tridiagonal matrix T and compute its eigenvalues
    Eigen::SelfAdjointEigenSolver<MatrixXd> eigsolver;
    MatrixXd T = MatrixXd::Zero(m, m);
    for (int j = 0; j < m; ++j) {
        T(j, j) = alpha_vec[j];
        if (j < m - 1) {
            T(j, j + 1) = beta_vec[j];
            T(j + 1, j) = beta_vec[j];
        }
    }

    eigsolver.compute(T);
    if (eigsolver.info() != Eigen::Success) {
        return 0.0;
    }

    // log|A| ≈ n * sum_i tau_i^2 * log(lambda_i)
    // where tau_i = e_1^T Q_i (first element of i-th eigenvector of T)
    // and lambda_i are eigenvalues of T
    const VectorXd& eigenvalues = eigsolver.eigenvalues();
    const MatrixXd& eigenvectors = eigsolver.eigenvectors();

    double logdet_contribution = 0.0;
    for (int i = 0; i < m; ++i) {
        double lambda = eigenvalues[i];
        if (lambda <= 0.0) continue;

        double tau = eigenvectors(0, i);  // first element of eigenvector
        logdet_contribution += tau * tau * std::log(lambda);
    }

    // Scale by n (dimension of A) since the Rademacher vector has norm ~sqrt(n)
    // and we normalized to unit norm
    return static_cast<double>(n) * logdet_contribution;
}

double StochasticLogDet::estimate(
    const SpMatd& A,
    const BlockDiagonalPreconditioner& /* precond */,
    int n_probes, int lanczos_steps, unsigned int seed)
{
    int n = A.rows();

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> coin(0, 1);

    double logdet_sum = 0.0;

    for (int probe = 0; probe < n_probes; ++probe) {
        // Generate Rademacher random vector
        VectorXd z(n);
        for (int i = 0; i < n; ++i) {
            z[i] = coin(rng) ? 1.0 : -1.0;
        }

        logdet_sum += slq_logdet_probe(A, z, lanczos_steps);
    }

    return logdet_sum / n_probes;
}

// ============================================================================
// CrossedRESolver
// ============================================================================

CrossedRESolver::CrossedRESolver(const MLMData& data, int pcg_threshold)
    : data_(data), pcg_threshold_(pcg_threshold)
{
    // Use PCG if the problem is crossed AND q exceeds the threshold
    use_pcg_ = data_.is_crossed && data_.q > pcg_threshold_;
}

CrossedRESolver::DevianceComponents CrossedRESolver::compute(
    const SpMatd& A,
    const VectorXd& LamtZty,
    const MatrixXd& LamtZtX,
    int n_probes)
{
    DevianceComponents dc;
    dc.pcg_iterations = 0;

    // Build preconditioner from diagonal blocks of A
    precond_.compute(A, data_.block_starts, data_.block_sizes);

    // Solve A * cu = LamtZty
    PCGSolver::Result cu_result = PCGSolver::solve(
        A, LamtZty, precond_, 1e-10, 500);
    dc.cu = cu_result.x;
    dc.pcg_iterations += cu_result.iterations;

    // Solve A * RZX_col = LamtZtX_col for each column
    int p = LamtZtX.cols();
    dc.RZX = MatrixXd(data_.q, p);
    for (int j = 0; j < p; ++j) {
        PCGSolver::Result col_result = PCGSolver::solve(
            A, LamtZtX.col(j), precond_, 1e-10, 500);
        dc.RZX.col(j) = col_result.x;
        dc.pcg_iterations += col_result.iterations;
    }

    // Estimate log|A| via stochastic Lanczos quadrature
    dc.logdet_A = StochasticLogDet::estimate(A, precond_, n_probes);

    return dc;
}

} // namespace fastmlm
