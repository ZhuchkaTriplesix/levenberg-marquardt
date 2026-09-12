use approx::assert_relative_eq;
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{ArrayStorage, Const, DVector, Dyn, OMatrix, U3, VecStorage, Vector2, Vector3};

type F = f64;

/// Fits a 2D circle: parameters are (center_x, center_y, radius).
#[derive(Clone)]
struct CircleFittingProblem {
    points: Vec<Vector2<F>>,
    params: Vector3<F>, // [center_x, center_y, radius]
}

impl LeastSquaresProblem<F, Dyn, U3> for CircleFittingProblem {
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
        let cx = self.params[0];
        let cy = self.params[1];
        let r = self.params[2];

        let mut res = DVector::zeros(self.points.len());
        for (i, p) in self.points.iter().enumerate() {
            let dx = p[0] - cx;
            let dy = p[1] - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            res[i] = dist - r;
        }
        Some(res)
    }

    fn jacobian(&self) -> Option<OMatrix<F, Dyn, U3>> {
        let cx = self.params[0];
        let cy = self.params[1];

        let mut jac = OMatrix::<F, Dyn, U3>::zeros_generic(Dyn(self.points.len()), Const::<3>);
        for (i, p) in self.points.iter().enumerate() {
            let dx = p[0] - cx;
            let dy = p[1] - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist > 1e-12 {
                jac[(i, 0)] = -dx / dist;
                jac[(i, 1)] = -dy / dist;
            } else {
                jac[(i, 0)] = 0.0;
                jac[(i, 1)] = 0.0;
            }
            jac[(i, 2)] = -1.0;
        }
        Some(jac)
    }
}

#[test]
fn test_circle_fitting_analytical() {
    let true_cx = 2.5;
    let true_cy = -1.5;
    let true_r = 4.0;

    // Generate circle points
    let n_points = 50;
    let mut points = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n_points as f64);
        let x = true_cx + true_r * angle.cos();
        let y = true_cy + true_r * angle.sin();
        points.push(Vector2::new(x, y));
    }

    // Initial guess perturbed from true values
    let problem = CircleFittingProblem {
        points,
        params: Vector3::new(0.0, 0.0, 1.0),
    };

    let (result, report) = LevenbergMarquardt::new().minimize(problem);

    assert!(report.termination.was_successful());
    assert_relative_eq!(result.params[0], true_cx, epsilon = 1e-7);
    assert_relative_eq!(result.params[1], true_cy, epsilon = 1e-7);
    assert_relative_eq!(result.params[2], true_r, epsilon = 1e-7);
    assert_relative_eq!(report.objective_function, 0.0, epsilon = 1e-10);
}

#[test]
fn test_circle_fitting_with_noise() {
    let true_cx = 10.0;
    let true_cy = 20.0;
    let true_r = 5.0;
    let frac = std::f64::consts::FRAC_1_SQRT_2;

    let points = vec![
        Vector2::new(15.01, 20.0),
        Vector2::new(5.0, 19.98),
        Vector2::new(10.02, 25.01),
        Vector2::new(9.99, 14.99),
        Vector2::new(10.0 + 5.0 * frac, 20.0 + 5.0 * frac),
        Vector2::new(10.0 - 5.0 * frac, 20.0 + 5.0 * frac),
        Vector2::new(10.0 + 5.0 * frac, 20.0 - 5.0 * frac),
        Vector2::new(10.0 - 5.0 * frac, 20.0 - 5.0 * frac),
    ];

    let problem = CircleFittingProblem {
        points,
        params: Vector3::new(8.0, 18.0, 3.0),
    };

    let (result, report) = LevenbergMarquardt::new().minimize(problem);

    assert!(report.termination.was_successful());
    assert_relative_eq!(result.params[0], true_cx, epsilon = 1e-2);
    assert_relative_eq!(result.params[1], true_cy, epsilon = 1e-2);
    assert_relative_eq!(result.params[2], true_r, epsilon = 1e-2);
}
