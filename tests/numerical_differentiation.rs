use approx::assert_relative_eq;
use levenberg_marquardt::{
    LeastSquaresProblem, differentiate_holomorphic_numerically, differentiate_numerically,
};
use nalgebra::{
    ArrayStorage, Complex, ComplexField, Const, Matrix2, OMatrix, U2, Vector2, convert,
};

#[derive(Clone)]
struct QuadraticProblem<F: ComplexField> {
    params: Vector2<F>,
    fail_residual: bool,
}

impl<F: ComplexField + Copy> LeastSquaresProblem<F, Const<2>, U2> for QuadraticProblem<F> {
    type ResidualStorage = ArrayStorage<F, 2, 1>;
    type JacobianStorage = ArrayStorage<F, 2, 2>;
    type ParameterStorage = ArrayStorage<F, 2, 1>;

    fn set_params(&mut self, p: &Vector2<F>) {
        self.params = *p;
    }

    fn params(&self) -> Vector2<F> {
        self.params
    }

    fn residuals(&self) -> Option<Vector2<F>> {
        if self.fail_residual {
            return None;
        }
        let x = self.params[0];
        let y = self.params[1];
        let two: F = convert(2.0);
        let three: F = convert(3.0);
        // r0 = x^2 + 2*y
        // r1 = 3*x*y - y^2
        Some(Vector2::new(x * x + two * y, three * x * y - y * y))
    }

    fn jacobian(&self) -> Option<OMatrix<F, Const<2>, U2>> {
        let x = self.params[0];
        let y = self.params[1];
        let two: F = convert(2.0);
        let three: F = convert(3.0);
        let mut jac = Matrix2::zeros();
        // dr0/dx = 2*x, dr0/dy = 2
        jac[(0, 0)] = two * x;
        jac[(0, 1)] = two;
        // dr1/dx = 3*y, dr1/dy = 3*x - 2*y
        jac[(1, 0)] = three * y;
        jac[(1, 1)] = three * x - two * y;
        Some(jac)
    }
}

#[test]
fn test_numerical_differentiation_matches_analytical() {
    let mut problem = QuadraticProblem::<f64> {
        params: Vector2::new(1.5, -2.0),
        fail_residual: false,
    };

    let analytical_jac = problem.jacobian().unwrap();
    let numerical_jac = differentiate_numerically(&mut problem).unwrap();

    assert_relative_eq!(
        analytical_jac[(0, 0)],
        numerical_jac[(0, 0)],
        epsilon = 1e-5
    );
    assert_relative_eq!(
        analytical_jac[(0, 1)],
        numerical_jac[(0, 1)],
        epsilon = 1e-5
    );
    assert_relative_eq!(
        analytical_jac[(1, 0)],
        numerical_jac[(1, 0)],
        epsilon = 1e-5
    );
    assert_relative_eq!(
        analytical_jac[(1, 1)],
        numerical_jac[(1, 1)],
        epsilon = 1e-5
    );
}

#[test]
fn test_holomorphic_differentiation_matches_analytical() {
    let x = Vector2::new(2.0, 3.0);
    let analytical_jac = (QuadraticProblem::<f64> {
        params: x,
        fail_residual: false,
    })
    .jacobian()
    .unwrap();

    let mut complex_problem = QuadraticProblem::<Complex<f64>> {
        params: convert(x),
        fail_residual: false,
    };
    let holomorphic_jac = differentiate_holomorphic_numerically(&mut complex_problem).unwrap();

    assert_relative_eq!(
        analytical_jac[(0, 0)],
        holomorphic_jac[(0, 0)],
        epsilon = 1e-12
    );
    assert_relative_eq!(
        analytical_jac[(0, 1)],
        holomorphic_jac[(0, 1)],
        epsilon = 1e-12
    );
    assert_relative_eq!(
        analytical_jac[(1, 0)],
        holomorphic_jac[(1, 0)],
        epsilon = 1e-12
    );
    assert_relative_eq!(
        analytical_jac[(1, 1)],
        holomorphic_jac[(1, 1)],
        epsilon = 1e-12
    );
}

#[test]
fn test_numerical_differentiation_handles_none() {
    let mut real_problem = QuadraticProblem::<f64> {
        params: Vector2::new(1.0, 1.0),
        fail_residual: true,
    };
    assert!(differentiate_numerically(&mut real_problem).is_none());

    let mut complex_problem = QuadraticProblem::<Complex<f64>> {
        params: convert(Vector2::new(1.0, 1.0)),
        fail_residual: true,
    };
    assert!(differentiate_holomorphic_numerically(&mut complex_problem).is_none());
}
