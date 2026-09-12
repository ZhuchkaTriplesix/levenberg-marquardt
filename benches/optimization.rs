use criterion::{Criterion, black_box, criterion_group, criterion_main};
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{Const, Dyn, OMatrix, OVector, Owned, U6, Vector3, Vector6};

// 1. Small fixed-size problem: 3D rigid transform / pose estimation (n=6 parameters, m=24 residuals).
// Parameters: 3 rotation angles (rx, ry, rz) + 3 translation components (tx, ty, tz).
// Points: 8 3D points (8 * 3 = 24 residuals).
#[derive(Clone)]
struct PoseEstimationProblem {
    params: Vector6<f64>,
    source_points: [Vector3<f64>; 8],
    target_points: [Vector3<f64>; 8],
}

impl PoseEstimationProblem {
    fn new() -> Self {
        let source_points = [
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(-1.0, 1.0, 1.0),
            Vector3::new(1.0, -1.0, 1.0),
            Vector3::new(-1.0, -1.0, 1.0),
            Vector3::new(1.0, 1.0, -1.0),
            Vector3::new(-1.0, 1.0, -1.0),
            Vector3::new(1.0, -1.0, -1.0),
            Vector3::new(-1.0, -1.0, -1.0),
        ];

        // True transform: small rotation + translation
        let true_rx = 0.1;
        let true_ry = 0.2;
        let true_rz = -0.15;
        let true_t = Vector3::new(0.5, -0.3, 0.8);

        let rot_x = nalgebra::Rotation3::from_axis_angle(&Vector3::x_axis(), true_rx);
        let rot_y = nalgebra::Rotation3::from_axis_angle(&Vector3::y_axis(), true_ry);
        let rot_z = nalgebra::Rotation3::from_axis_angle(&Vector3::z_axis(), true_rz);
        let true_rot = rot_z * rot_y * rot_x;

        let mut target_points = [Vector3::zeros(); 8];
        for (i, p) in source_points.iter().enumerate() {
            target_points[i] = true_rot * p + true_t;
        }

        Self {
            // Initial guess: zeros
            params: Vector6::zeros(),
            source_points,
            target_points,
        }
    }
}

impl LeastSquaresProblem<f64, Const<24>, U6> for PoseEstimationProblem {
    type ParameterStorage = Owned<f64, U6>;
    type ResidualStorage = Owned<f64, Const<24>>;
    type JacobianStorage = Owned<f64, Const<24>, U6>;

    fn set_params(&mut self, params: &Vector6<f64>) {
        self.params.copy_from(params);
    }

    fn params(&self) -> Vector6<f64> {
        self.params
    }

    fn residuals(&self) -> Option<OVector<f64, Const<24>>> {
        let rx = self.params[0];
        let ry = self.params[1];
        let rz = self.params[2];
        let t = Vector3::new(self.params[3], self.params[4], self.params[5]);

        let rot_x = nalgebra::Rotation3::from_axis_angle(&Vector3::x_axis(), rx);
        let rot_y = nalgebra::Rotation3::from_axis_angle(&Vector3::y_axis(), ry);
        let rot_z = nalgebra::Rotation3::from_axis_angle(&Vector3::z_axis(), rz);
        let rot = rot_z * rot_y * rot_x;

        let mut res = OVector::<f64, Const<24>>::zeros();
        for (i, (src, tgt)) in self
            .source_points
            .iter()
            .zip(self.target_points.iter())
            .enumerate()
        {
            let transformed = rot * src + t;
            let diff = transformed - tgt;
            res[3 * i] = diff.x;
            res[3 * i + 1] = diff.y;
            res[3 * i + 2] = diff.z;
        }
        Some(res)
    }

    fn jacobian(&self) -> Option<OMatrix<f64, Const<24>, U6>> {
        let mut jac = OMatrix::<f64, Const<24>, U6>::zeros();
        let eps = 1e-7;
        let base_res = self.residuals()?;

        let mut problem = self.clone();
        for j in 0..6 {
            let mut perturbed_params = self.params;
            perturbed_params[j] += eps;
            problem.set_params(&perturbed_params);
            let perturbed_res = problem.residuals()?;
            for i in 0..24 {
                jac[(i, j)] = (perturbed_res[i] - base_res[i]) / eps;
            }
        }
        Some(jac)
    }
}

// 2. Large dynamic-size problem: polynomial curve fitting with n=20 parameters and m=1000 data points.
#[derive(Clone)]
struct LargeDynamicProblem {
    params: OVector<f64, Dyn>,
    x_coords: Vec<f64>,
    y_values: Vec<f64>,
}

impl LargeDynamicProblem {
    fn new(num_params: usize, num_samples: usize) -> Self {
        let x_coords: Vec<f64> = (0..num_samples)
            .map(|i| -1.0 + 2.0 * (i as f64) / (num_samples as f64 - 1.0))
            .collect();

        // True coefficients: decaying powers
        let true_coeffs: Vec<f64> = (0..num_params)
            .map(|j| 0.5 * (-1.0_f64).powi(j as i32) / (j as f64 + 1.0))
            .collect();

        let y_values: Vec<f64> = x_coords
            .iter()
            .map(|&x| {
                let mut val = 0.0;
                let mut x_pow = 1.0;
                for &c in &true_coeffs {
                    val += c * x_pow;
                    x_pow *= x;
                }
                val
            })
            .collect();

        Self {
            params: OVector::<f64, Dyn>::zeros_generic(Dyn(num_params), Const::<1>),
            x_coords,
            y_values,
        }
    }
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for LargeDynamicProblem {
    type ParameterStorage = Owned<f64, Dyn>;
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Dyn>;

    fn set_params(&mut self, params: &OVector<f64, Dyn>) {
        self.params.copy_from(params);
    }

    fn params(&self) -> OVector<f64, Dyn> {
        self.params.clone()
    }

    fn residuals(&self) -> Option<OVector<f64, Dyn>> {
        let m = self.x_coords.len();
        let n = self.params.nrows();
        let mut res = OVector::<f64, Dyn>::zeros_generic(Dyn(m), Const::<1>);

        for (i, (&x, &y_target)) in self.x_coords.iter().zip(self.y_values.iter()).enumerate() {
            let mut model_val = 0.0;
            let mut x_pow = 1.0;
            for j in 0..n {
                model_val += self.params[j] * x_pow;
                x_pow *= x;
            }
            res[i] = model_val - y_target;
        }
        Some(res)
    }

    fn jacobian(&self) -> Option<OMatrix<f64, Dyn, Dyn>> {
        let m = self.x_coords.len();
        let n = self.params.nrows();
        let mut jac = OMatrix::<f64, Dyn, Dyn>::zeros_generic(Dyn(m), Dyn(n));

        for (i, &x) in self.x_coords.iter().enumerate() {
            let mut x_pow = 1.0;
            for j in 0..n {
                jac[(i, j)] = x_pow;
                x_pow *= x;
            }
        }
        Some(jac)
    }
}

fn bench_fixed_pose_estimation(c: &mut Criterion) {
    let problem = PoseEstimationProblem::new();
    c.bench_function("fixed_pose_n6_m24", |b| {
        b.iter(|| {
            let lm = LevenbergMarquardt::new();
            let (_res, report) = lm.minimize(black_box(problem.clone()));
            assert!(report.termination.was_successful());
        })
    });
}

fn bench_dynamic_polynomial_fit(c: &mut Criterion) {
    let problem = LargeDynamicProblem::new(20, 1000);
    c.bench_function("dynamic_polynomial_n20_m1000", |b| {
        b.iter(|| {
            let lm = LevenbergMarquardt::new();
            let (_res, report) = lm.minimize(black_box(problem.clone()));
            assert!(report.termination.was_successful());
        })
    });
}

criterion_group!(
    benches,
    bench_fixed_pose_estimation,
    bench_dynamic_polynomial_fit
);
criterion_main!(benches);
