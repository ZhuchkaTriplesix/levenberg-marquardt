use approx::assert_relative_eq;
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt, TerminationReason};
use nalgebra::{ArrayStorage, Const, OMatrix, U2, Vector2};

type F = f64;

/// Rosenbrock problem for testing convergence and limits
#[derive(Clone)]
struct RosenbrockProblem {
    params: Vector2<F>,
}

impl LeastSquaresProblem<F, Const<2>, U2> for RosenbrockProblem {
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
        let x = self.params[0];
        let y = self.params[1];
        Some(Vector2::new(10.0 * (y - x * x), 1.0 - x))
    }

    fn jacobian(&self) -> Option<OMatrix<F, Const<2>, U2>> {
        let x = self.params[0];
        let mut jac = OMatrix::<F, Const<2>, U2>::zeros();
        jac[(0, 0)] = -20.0 * x;
        jac[(0, 1)] = 10.0;
        jac[(1, 0)] = -1.0;
        jac[(1, 1)] = 0.0;
        Some(jac)
    }
}

#[test]
fn test_lost_patience() {
    let problem = RosenbrockProblem {
        params: Vector2::new(-1.2, 1.0),
    };

    // Setting patience to 1 limits the number of evaluations to patience * (n + 1) = 1 * 3 = 3
    let (_result, report) = LevenbergMarquardt::new().with_patience(1).minimize(problem);

    assert_eq!(report.termination, TerminationReason::LostPatience);
    assert!(!report.termination.was_successful());
}

#[test]
fn test_builder_options() {
    let problem = RosenbrockProblem {
        params: Vector2::new(-1.2, 1.0),
    };

    let (result, report) = LevenbergMarquardt::new()
        .with_ftol(1e-10)
        .with_xtol(1e-10)
        .with_gtol(1e-10)
        .with_stepbound(100.0)
        .with_patience(100)
        .with_scale_diag(true)
        .minimize(problem);

    assert!(report.termination.was_successful());
    assert_relative_eq!(result.params[0], 1.0, epsilon = 1e-4);
    assert_relative_eq!(result.params[1], 1.0, epsilon = 1e-4);
    assert_relative_eq!(report.objective_function, 0.0, epsilon = 1e-8);
}
