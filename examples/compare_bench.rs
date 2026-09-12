use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{
    Const, Dyn, Matrix2, OMatrix, OVector, Owned, Rotation3, Vector2, Vector3, Vector6,
};
use std::time::Instant;

// =========================================================================
// Problem 1: Rosenbrock Function (n=2, m=2)
// f(x, y) = [10*(y - x^2), 1 - x]
// Minimum at (1, 1) with objective = 0
// =========================================================================
#[derive(Clone)]
struct RosenbrockProblem {
    params: Vector2<f64>,
}

impl LeastSquaresProblem<f64, Const<2>, Const<2>> for RosenbrockProblem {
    type ParameterStorage = Owned<f64, Const<2>>;
    type ResidualStorage = Owned<f64, Const<2>>;
    type JacobianStorage = Owned<f64, Const<2>, Const<2>>;

    fn set_params(&mut self, p: &Vector2<f64>) {
        self.params.copy_from(p);
    }
    fn params(&self) -> Vector2<f64> {
        self.params
    }
    fn residuals(&self) -> Option<Vector2<f64>> {
        let x = self.params[0];
        let y = self.params[1];
        Some(Vector2::new(10.0 * (y - x * x), 1.0 - x))
    }
    fn jacobian(&self) -> Option<Matrix2<f64>> {
        let x = self.params[0];
        Some(Matrix2::new(-20.0 * x, 10.0, -1.0, 0.0))
    }
}

// =========================================================================
// Problem 2: Circle Fitting (n=3, m=100)
// Fits center (cx, cy) and radius r to points lying on a circle
// =========================================================================
#[derive(Clone)]
struct CircleFittingProblem {
    params: Vector3<f64>, // [cx, cy, r]
    points: Vec<(f64, f64)>,
}

impl LeastSquaresProblem<f64, Dyn, Const<3>> for CircleFittingProblem {
    type ParameterStorage = Owned<f64, Const<3>>;
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Const<3>>;

    fn set_params(&mut self, p: &Vector3<f64>) {
        self.params.copy_from(p);
    }
    fn params(&self) -> Vector3<f64> {
        self.params
    }
    fn residuals(&self) -> Option<OVector<f64, Dyn>> {
        let cx = self.params[0];
        let cy = self.params[1];
        let r = self.params[2];
        let mut res = OVector::<f64, Dyn>::zeros_generic(Dyn(self.points.len()), Const::<1>);
        for (i, &(px, py)) in self.points.iter().enumerate() {
            let dist = ((px - cx).powi(2) + (py - cy).powi(2)).sqrt();
            res[i] = dist - r;
        }
        Some(res)
    }
    fn jacobian(&self) -> Option<OMatrix<f64, Dyn, Const<3>>> {
        let cx = self.params[0];
        let cy = self.params[1];
        let mut jac =
            OMatrix::<f64, Dyn, Const<3>>::zeros_generic(Dyn(self.points.len()), Const::<3>);
        for (i, &(px, py)) in self.points.iter().enumerate() {
            let dx = px - cx;
            let dy = py - cy;
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

// =========================================================================
// Problem 3: Exponential Decay (n=3, m=50)
// y = a * exp(-b * x) + c
// =========================================================================
#[derive(Clone)]
struct ExponentialDecayProblem {
    params: Vector3<f64>, // [a, b, c]
    x_data: Vec<f64>,
    y_data: Vec<f64>,
}

impl LeastSquaresProblem<f64, Dyn, Const<3>> for ExponentialDecayProblem {
    type ParameterStorage = Owned<f64, Const<3>>;
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Const<3>>;

    fn set_params(&mut self, p: &Vector3<f64>) {
        self.params.copy_from(p);
    }
    fn params(&self) -> Vector3<f64> {
        self.params
    }
    fn residuals(&self) -> Option<OVector<f64, Dyn>> {
        let a = self.params[0];
        let b = self.params[1];
        let c = self.params[2];
        let mut res = OVector::<f64, Dyn>::zeros_generic(Dyn(self.x_data.len()), Const::<1>);
        for (i, (&x, &y)) in self.x_data.iter().zip(self.y_data.iter()).enumerate() {
            res[i] = a * (-b * x).exp() + c - y;
        }
        Some(res)
    }
    fn jacobian(&self) -> Option<OMatrix<f64, Dyn, Const<3>>> {
        let a = self.params[0];
        let b = self.params[1];
        let mut jac =
            OMatrix::<f64, Dyn, Const<3>>::zeros_generic(Dyn(self.x_data.len()), Const::<3>);
        for (i, &x) in self.x_data.iter().enumerate() {
            let exp_term = (-b * x).exp();
            jac[(i, 0)] = exp_term;
            jac[(i, 1)] = -a * x * exp_term;
            jac[(i, 2)] = 1.0;
        }
        Some(jac)
    }
}

// =========================================================================
// Problem 4: 3D Pose Estimation (n=6, m=24)
// Rotates and translates 8 points in 3D
// =========================================================================
#[derive(Clone)]
struct PoseEstimationProblem {
    params: Vector6<f64>,
    source_points: [Vector3<f64>; 8],
    target_points: [Vector3<f64>; 8],
}

impl LeastSquaresProblem<f64, Const<24>, Const<6>> for PoseEstimationProblem {
    type ParameterStorage = Owned<f64, Const<6>>;
    type ResidualStorage = Owned<f64, Const<24>>;
    type JacobianStorage = Owned<f64, Const<24>, Const<6>>;

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

        let rot = Rotation3::from_axis_angle(&Vector3::z_axis(), rz)
            * Rotation3::from_axis_angle(&Vector3::y_axis(), ry)
            * Rotation3::from_axis_angle(&Vector3::x_axis(), rx);

        let mut res = OVector::<f64, Const<24>>::zeros();
        for (i, (src, tgt)) in self
            .source_points
            .iter()
            .zip(self.target_points.iter())
            .enumerate()
        {
            let diff = rot * src + t - tgt;
            res[3 * i] = diff.x;
            res[3 * i + 1] = diff.y;
            res[3 * i + 2] = diff.z;
        }
        Some(res)
    }
    fn jacobian(&self) -> Option<OMatrix<f64, Const<24>, Const<6>>> {
        let mut jac = OMatrix::<f64, Const<24>, Const<6>>::zeros();
        let eps = 1e-7;
        let base_res = self.residuals()?;

        let mut problem = self.clone();
        for j in 0..6 {
            let mut perturbed = self.params;
            perturbed[j] += eps;
            problem.set_params(&perturbed);
            let perturbed_res = problem.residuals()?;
            for i in 0..24 {
                jac[(i, j)] = (perturbed_res[i] - base_res[i]) / eps;
            }
        }
        Some(jac)
    }
}

// =========================================================================
// Problem 5: High-Dimensional Polynomial (n=20, m=1000)
// =========================================================================
#[derive(Clone)]
struct LargeDynamicProblem {
    params: OVector<f64, Dyn>,
    x_coords: Vec<f64>,
    y_values: Vec<f64>,
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

macro_rules! run_bench {
    ($name:expr, $create:expr, $get_error:expr, $warmup:expr, $iters:expr $(,)?) => {{
        let create_fn = $create;
        for _ in 0..$warmup {
            let p = create_fn();
            let _ = LevenbergMarquardt::new().minimize(p);
        }
        let mut last_res = None;
        let start = Instant::now();
        for _ in 0..$iters {
            let p = create_fn();
            last_res = Some(LevenbergMarquardt::new().minimize(p));
        }
        let elapsed = start.elapsed();
        let avg_us = elapsed.as_secs_f64() * 1_000_000.0 / ($iters as f64);

        let (final_problem, report) = last_res.unwrap();
        let converged = report.termination.was_successful();
        let param_err = $get_error(&final_problem);

        println!(
            "{{\"problem\": \"{}\", \"converged\": {}, \"objective\": {:.10e}, \"param_error\": {:.10e}, \"evals\": {}, \"avg_us\": {:.3}}}",
            $name, converged, report.objective_function, param_err, report.number_of_evaluations, avg_us
        );
    }};
}

fn main() {
    // 1. Rosenbrock
    run_bench!(
        "Rosenbrock (n=2, m=2)",
        || RosenbrockProblem {
            params: Vector2::new(-1.2, 1.0)
        },
        |p: &RosenbrockProblem| (p.params - Vector2::new(1.0, 1.0)).norm(),
        500,
        5000,
    );

    // 2. Circle Fitting
    let true_center = (2.5, -1.0);
    let true_r = 5.0;
    let circle_points: Vec<(f64, f64)> = (0..100)
        .map(|i| {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / 100.0;
            (
                true_center.0 + true_r * angle.cos(),
                true_center.1 + true_r * angle.sin(),
            )
        })
        .collect();
    let circle_points_clone = circle_points.clone();
    run_bench!(
        "Circle Fitting (n=3, m=100)",
        move || CircleFittingProblem {
            params: Vector3::new(0.0, 0.0, 1.0),
            points: circle_points_clone.clone(),
        },
        |p: &CircleFittingProblem| {
            let d_c = ((p.params[0] - 2.5).powi(2) + (p.params[1] - (-1.0)).powi(2)).sqrt();
            let d_r = (p.params[2] - 5.0).abs();
            d_c + d_r
        },
        200,
        2000,
    );

    // 3. Exponential Decay
    let true_a = 2.5;
    let true_b = 1.3;
    let true_c = 0.5;
    let exp_x: Vec<f64> = (0..50).map(|i| (i as f64) * 0.1).collect();
    let exp_y: Vec<f64> = exp_x
        .iter()
        .map(|&x| true_a * (-true_b * x).exp() + true_c)
        .collect();
    let exp_x_clone = exp_x.clone();
    let exp_y_clone = exp_y.clone();
    run_bench!(
        "Exponential Decay (n=3, m=50)",
        move || ExponentialDecayProblem {
            params: Vector3::new(1.0, 0.5, 0.0),
            x_data: exp_x_clone.clone(),
            y_data: exp_y_clone.clone(),
        },
        |p: &ExponentialDecayProblem| {
            ((p.params[0] - true_a).powi(2)
                + (p.params[1] - true_b).powi(2)
                + (p.params[2] - true_c).powi(2))
            .sqrt()
        },
        200,
        2000,
    );

    // 4. 3D Pose Estimation
    let pose_src = [
        Vector3::new(1.0, 1.0, 1.0),
        Vector3::new(-1.0, 1.0, 1.0),
        Vector3::new(1.0, -1.0, 1.0),
        Vector3::new(-1.0, -1.0, 1.0),
        Vector3::new(1.0, 1.0, -1.0),
        Vector3::new(-1.0, 1.0, -1.0),
        Vector3::new(1.0, -1.0, -1.0),
        Vector3::new(-1.0, -1.0, -1.0),
    ];
    let true_pose = Vector6::new(0.1, 0.2, -0.15, 0.5, -0.3, 0.8);
    let true_rot = Rotation3::from_axis_angle(&Vector3::z_axis(), true_pose[2])
        * Rotation3::from_axis_angle(&Vector3::y_axis(), true_pose[1])
        * Rotation3::from_axis_angle(&Vector3::x_axis(), true_pose[0]);
    let true_t = Vector3::new(true_pose[3], true_pose[4], true_pose[5]);
    let mut pose_tgt = [Vector3::zeros(); 8];
    for (i, p) in pose_src.iter().enumerate() {
        pose_tgt[i] = true_rot * p + true_t;
    }
    run_bench!(
        "3D Pose Estimation (n=6, m=24)",
        move || PoseEstimationProblem {
            params: Vector6::zeros(),
            source_points: pose_src,
            target_points: pose_tgt,
        },
        |p: &PoseEstimationProblem| (p.params - true_pose).norm(),
        200,
        2000,
    );

    // 5. Large Dynamic Polynomial
    let poly_n = 20;
    let poly_m = 1000;
    let poly_x: Vec<f64> = (0..poly_m)
        .map(|i| -1.0 + 2.0 * (i as f64) / (poly_m as f64 - 1.0))
        .collect();
    let true_poly_coeffs: Vec<f64> = (0..poly_n)
        .map(|j| 0.5 * (-1.0_f64).powi(j as i32) / (j as f64 + 1.0))
        .collect();
    let poly_y: Vec<f64> = poly_x
        .iter()
        .map(|&x| {
            let mut val = 0.0;
            let mut x_pow = 1.0;
            for &c in &true_poly_coeffs {
                val += c * x_pow;
                x_pow *= x;
            }
            val
        })
        .collect();
    let poly_x_clone = poly_x.clone();
    let poly_y_clone = poly_y.clone();
    let true_poly_vec =
        OVector::<f64, Dyn>::from_column_slice_generic(Dyn(poly_n), Const::<1>, &true_poly_coeffs);
    run_bench!(
        "Dynamic Polynomial (n=20, m=1000)",
        move || LargeDynamicProblem {
            params: OVector::<f64, Dyn>::zeros_generic(Dyn(poly_n), Const::<1>),
            x_coords: poly_x_clone.clone(),
            y_values: poly_y_clone.clone(),
        },
        |p: &LargeDynamicProblem| (&p.params - &true_poly_vec).norm(),
        50,
        200,
    );
}
