use approx::assert_relative_eq;
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{Const, DVector, Dyn, OMatrix, VecStorage};

type F = f64;

/// Fits a dynamic-degree polynomial: P(x) = sum_{k=0}^{degree-1} c_k * x^k
#[derive(Clone)]
struct DynamicPolynomialProblem {
    x_data: Vec<F>,
    y_data: Vec<F>,
    degree: usize,
    params: DVector<F>,
}

impl LeastSquaresProblem<F, Dyn, Dyn> for DynamicPolynomialProblem {
    type ResidualStorage = VecStorage<F, Dyn, Const<1>>;
    type JacobianStorage = VecStorage<F, Dyn, Dyn>;
    type ParameterStorage = VecStorage<F, Dyn, Const<1>>;

    fn set_params(&mut self, p: &DVector<F>) {
        self.params = p.clone();
    }

    fn params(&self) -> DVector<F> {
        self.params.clone()
    }

    fn residuals(&self) -> Option<DVector<F>> {
        let n = self.x_data.len();
        let mut res = DVector::zeros(n);
        for i in 0..n {
            let x = self.x_data[i];
            let mut val = 0.0;
            let mut x_pow = 1.0;
            for k in 0..self.degree {
                val += self.params[k] * x_pow;
                x_pow *= x;
            }
            res[i] = self.y_data[i] - val;
        }
        Some(res)
    }

    fn jacobian(&self) -> Option<OMatrix<F, Dyn, Dyn>> {
        let n = self.x_data.len();
        let mut jac = OMatrix::<F, Dyn, Dyn>::zeros_generic(Dyn(n), Dyn(self.degree));
        for i in 0..n {
            let x = self.x_data[i];
            let mut x_pow = 1.0;
            for k in 0..self.degree {
                // dr_i / dc_k = -x^k
                jac[(i, k)] = -x_pow;
                x_pow *= x;
            }
        }
        Some(jac)
    }
}

#[test]
fn test_dynamic_polynomial_cubic() {
    let degree = 4; // cubic polynomial: c0 + c1*x + c2*x^2 + c3*x^3
    let true_coeffs = [1.5, -2.0, 0.5, 0.1];

    let n = 20;
    let mut x_data = Vec::with_capacity(n);
    let mut y_data = Vec::with_capacity(n);
    for i in 0..n {
        let x = (i as f64) * 0.5 - 5.0;
        let mut y = 0.0;
        let mut x_pow = 1.0;
        for &coeff in &true_coeffs {
            y += coeff * x_pow;
            x_pow *= x;
        }
        x_data.push(x);
        y_data.push(y);
    }

    let problem = DynamicPolynomialProblem {
        x_data,
        y_data,
        degree,
        params: DVector::zeros(degree),
    };

    let (result, report) = LevenbergMarquardt::new().minimize(problem);

    assert!(report.termination.was_successful());
    for (k, &true_coeff) in true_coeffs.iter().enumerate() {
        assert_relative_eq!(result.params[k], true_coeff, epsilon = 1e-6);
    }
    assert_relative_eq!(report.objective_function, 0.0, epsilon = 1e-10);
}
