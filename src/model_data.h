#ifndef FASTMLM_MODEL_DATA_H
#define FASTMLM_MODEL_DATA_H

#include "fastmlm_types.h"

namespace fastmlm {

class MLMData {
public:
    // Dimensions
    int n;    // number of observations
    int p;    // number of fixed-effect columns
    int q;    // total random-effect columns
    int nth;  // length of theta

    // Core data (set once)
    VectorXd y;          // response (n)
    MatrixXd X;          // fixed-effects design matrix (n x p)
    SpMatd Zt;           // Z^T (q x n), stored as sparse
    SpMatd Lambdat;      // relative covariance factor template (q x q)

    // Mapping
    VectorXi Lind;       // maps theta → Lambdat nonzero positions
    VectorXd lower;      // lower bounds on theta

    // Precomputed cross-products
    SpMatd ZtZ;          // Z^T Z (q x q)
    MatrixXd XtX;        // X^T X (p x p)
    VectorXd Xty;        // X^T y (p)
    VectorXd Zty;        // Z^T y (q)

    // Structure classification
    int n_re_terms;       // number of random-effect grouping factors
    bool is_crossed;      // true if any RE terms share observations across groups
    VectorXi block_sizes; // size of each RE term's block (q_i for term i)
    VectorXi block_starts;// start index of each RE term's block in q

    // Construct from R objects (output of lme4::lFormula)
    MLMData(const VectorXd& y_,
            const MatrixXd& X_,
            const SpMatd& Zt_,
            const SpMatd& Lambdat_,
            const VectorXi& Lind_,
            const VectorXd& lower_);

    // Set block structure from R's Gp vector
    void set_block_structure(const VectorXi& Gp);

    // Update Lambdat nonzeros from theta
    void update_Lambdat(const VectorXd& theta);

    // Get current Lambdat (after update)
    const SpMatd& get_Lambdat() const { return Lambdat; }

private:
    void precompute();
    void detect_crossed();
};

} // namespace fastmlm

#endif // FASTMLM_MODEL_DATA_H
