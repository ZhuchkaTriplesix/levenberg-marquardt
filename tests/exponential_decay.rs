use approx::assert_relative_eq;
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt, differentiate_numerically};
use nalgebra::{ArrayStorage, Const, DVector, Dyn, OMatrix, U3, VecStorage, Vector3};

type F = f64;

/// Fits exponential decay: y = a * exp(-b * x) + c
/// Parameters: [a, b, c]
#[derive(Clone)]
struct ExponentialDecayProblem {
    x_data: Vec<F>,
    y_data: Vec<F>,
    params: Vector3<F>,
}

impl LeastSquaresProblem<F, Dyn, U3> for ExponentialDecayProblem {
    type ResidualStorage = VecStorage<F, Dyn, Const<1>>;
    type JacobianStorage = VecStorage<F, Dyn, U3>;
    type ParameterStorage = ArrayStorage<F, 3, 1>;

    fn set_params(&mut self, p: &Vector3<F>) {
        self.params = *p;
    }

    fn params(&self) -> Vector3<F> {
        self.params
    }

    fn residuals(&self) -> Option<DVector<F>> {
        let a = self.params[0];
        let b = self.params[1];
        let c = self.params[2];

        let mut res = DVector::zeros(self.x_data.len());
        for (i, (&x, &y)) in self.x_data.iter().zip(self.y_data.iter()).enumerate() {
            let model_y = a * (-b * x).exp() + c;
            res[i] = y - model_y;
        }
        Some(res)
    }

    fn jacobian(&self) -> Option<OMatrix<F, Dyn, U3>> {
        let a = self.params[0];
        let b = self.params[1];

        let mut jac = OMatrix::<F, Dyn, U3>::zeros_generic(Dyn(self.x_data.len()), Const::<3>);
        for (i, &x) in self.x_data.iter().enumerate() {
            let exp_term = (-b * x).exp();
            // dr/da = -exp(-b*x)
            jac[(i, 0)] = -exp_term;
            // dr/db = a * x * exp(-b*x)
            jac[(i, 1)] = a * x * exp_term;
            // dr/dc = -1
            jac[(i, 2)] = -1.0;
        }
        Some(jac)
    }
}

#[test]
fn test_exponential_decay_analytical() {
    let true_a = 5.0;
    let true_b = 0.8;
    let true_c = 1.2;

    let n = 30;
    let mut x_data = Vec::with_capacity(n);
    let mut y_data = Vec::with_capacity(n);
    for i in 0..n {
        let x = (i as f64) * 0.2;
        let y = true_a * (-true_b * x).exp() + true_c;
        x_data.push(x);
        y_data.push(y);
    }

    let problem = ExponentialDecayProblem {
        x_data,
        y_data,
        params: Vector3::new(2.0, 0.2, 0.0), // initial guess
    };

    let (result, report) = LevenbergMarquardt::new().minimize(problem);

    assert!(report.termination.was_successful());
    assert_relative_eq!(result.params[0], true_a, epsilon = 1e-6);
    assert_relative_eq!(result.params[1], true_b, epsilon = 1e-6);
    assert_relative_eq!(result.params[2], true_c, epsilon = 1e-6);
    assert_relative_eq!(report.objective_function, 0.0, epsilon = 1e-10);
}

#[test]
fn test_exponential_decay_numerical_jacobian_check() {
    let mut problem = ExponentialDecayProblem {
        x_data: vec![0.0, 0.5, 1.0, 1.5, 2.0],
        y_data: vec![5.0, 3.5, 2.5, 1.8, 1.4],
        params: Vector3::new(4.0, 0.7, 1.0),
    };

    let analytical_jac = problem.jacobian().unwrap();
    let numerical_jac = differentiate_numerically(&mut problem).unwrap();

    for r in 0..problem.x_data.len() {
        for c in 0..3 {
            assert_relative_eq!(
                analytical_jac[(r, c)],
                numerical_jac[(r, c)],
                epsilon = 1e-5
            );
        }
    }
}
