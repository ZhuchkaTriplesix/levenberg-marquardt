use nalgebra::{
    ComplexField, Dim, Matrix, Vector,
    storage::{IsContiguous, RawStorageMut, Storage},
};

/// A least squares minimization problem.
///
/// This is what [`LevenbergMarquardt`](struct.LevenbergMarquardt.html) needs
/// to compute the residuals and the Jacobian. See the [module documentation](index.html)
/// for a usage example.
pub trait LeastSquaresProblem<F, M, N>
where
    F: ComplexField + Copy,
    N: Dim,
    M: Dim,
{
    /// Storage type used for the residuals. Use `nalgebra::storage::Owned<F, M>`
    /// if you want to use `VectorN` or `MatrixMN`.
    type ResidualStorage: RawStorageMut<F, M> + Storage<F, M> + IsContiguous;
    type JacobianStorage: RawStorageMut<F, M, N> + Storage<F, M, N> + IsContiguous;
    type ParameterStorage: RawStorageMut<F, N> + Storage<F, N> + IsContiguous + Clone;

    /// Set the stored parameters `$\vec{x}$`.
    fn set_params(&mut self, x: &Vector<F, N, Self::ParameterStorage>);

    /// Get the current parameter vector `$\vec{x}$`.
    fn params(&self) -> Vector<F, N, Self::ParameterStorage>;

    /// Compute the residual vector.
    fn residuals(&self) -> Option<Vector<F, M, Self::ResidualStorage>>;

    /// Compute the residual vector into an existing buffer.
    ///
    /// The default implementation calls [`residuals`](#tymethod.residuals) and copies the result.
    /// Override this method to avoid heap allocations in dynamic problems.
    fn residuals_into(&self, out: &mut Vector<F, M, Self::ResidualStorage>) -> bool {
        if let Some(res) = self.residuals() {
            *out = res;
            true
        } else {
            false
        }
    }

    /// Compute the Jacobian of the residual vector.
    fn jacobian(&self) -> Option<Matrix<F, M, N, Self::JacobianStorage>>;

    /// Compute the Jacobian of the residual vector into an existing buffer.
    ///
    /// The default implementation calls [`jacobian`](#tymethod.jacobian) and copies the result.
    /// Override this method to avoid heap allocations in dynamic problems.
    fn jacobian_into(&self, out: &mut Matrix<F, M, N, Self::JacobianStorage>) -> bool {
        if let Some(jac) = self.jacobian() {
            *out = jac;
            true
        } else {
            false
        }
    }
}
