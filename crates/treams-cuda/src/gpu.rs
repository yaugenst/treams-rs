//! Device-resident dense complex128 operations, with no reduced-precision path.
#![allow(unsafe_code)] // CUDA FFI is confined to validated dimensions and owned buffers.

use std::sync::Arc;

use cudarc::{
    cublas::{CudaBlas, sys as blas},
    cusolver::{DnHandle, sys as solver},
    driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut},
};
use nalgebra::DMatrix;
use treams_core::{Complex, linalg};

/// Errors from the optional GPU boundary.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Invalid matrix shape, device ownership, or non-finite host input.
    #[error("{0}")]
    InvalidInput(String),
    /// Missing driver/libraries or an unsupported host.
    #[error("CUDA unavailable: {0}")]
    Unavailable(String),
    /// Driver or NVIDIA numerical library failure.
    #[error("CUDA operation failed: {0}")]
    Cuda(String),
    /// Pivoted LU found an exactly singular matrix.
    #[error("singular operator (zero pivot {0})")]
    Singular(i32),
}

/// Result type for GPU operations.
pub type Result<T> = std::result::Result<T, Error>;

fn cuda(error: impl std::fmt::Display) -> Error {
    Error::Cuda(error.to_string())
}

fn dimension(value: usize) -> Result<i32> {
    i32::try_from(value)
        .ok()
        .filter(|&x| x > 0)
        .ok_or_else(|| Error::InvalidInput("matrix dimensions must lie in 1..=i32::MAX".into()))
}

#[derive(Debug)]
struct Handles {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    solver: DnHandle,
}

/// One device and stream, shared by owned device matrices and factorizations.
#[derive(Clone, Debug)]
pub struct Gpu(Arc<Handles>);

/// A column-major complex128 matrix retained on its originating GPU stream.
#[derive(Debug)]
pub struct DeviceMatrix {
    gpu: Gpu,
    values: CudaSlice<[f64; 2]>,
    rows: usize,
    cols: usize,
}

impl DeviceMatrix {
    /// Matrix dimensions.
    #[must_use]
    pub const fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Resident matrix storage, excluding allocator and library workspace.
    #[must_use]
    pub const fn bytes(&self) -> usize {
        self.rows * self.cols * size_of::<Complex>()
    }

    /// Copy on the device without a host transfer.
    pub fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            gpu: self.gpu.clone(),
            values: self.values.try_clone().map_err(cuda)?,
            rows: self.rows,
            cols: self.cols,
        })
    }

    /// Copy to a CPU matrix and synchronize the owning stream.
    pub fn download(&self) -> Result<DMatrix<Complex>> {
        self.gpu.download(self)
    }
}

impl Gpu {
    /// Open a CUDA device. CPU-only builds never enter or link this path.
    pub fn new(device: usize) -> Result<Self> {
        if !cfg!(target_os = "linux") {
            return Err(Error::Unavailable("the CUDA backend requires Linux".into()));
        }
        // Check expected deployment errors before cudarc's lazy symbol loader,
        // which otherwise panics when a shared library is missing.
        for name in ["libcuda.so.1", "libcublas.so.13", "libcusolver.so.12"] {
            // SAFETY: only NVIDIA library initialization runs; no symbols or
            // borrowed pointers escape the probe. Cudarc owns its later handles.
            unsafe { libloading::Library::new(name) }
                .map_err(|error| Error::Unavailable(format!("{name}: {error}")))?;
        }
        let context = CudaContext::new(device).map_err(cuda)?;
        let stream = context.new_stream().map_err(cuda)?;
        Ok(Self(Arc::new(Handles {
            blas: CudaBlas::new(stream.clone()).map_err(cuda)?,
            solver: DnHandle::new(stream.clone()).map_err(cuda)?,
            stream,
        })))
    }

    /// NVIDIA device name.
    pub fn name(&self) -> Result<String> {
        self.0.stream.context().name().map_err(cuda)
    }

    /// Wait for all previously submitted work on this stream.
    pub fn synchronize(&self) -> Result<()> {
        self.0.stream.synchronize().map_err(cuda)
    }

    fn check_owner(&self, matrix: &DeviceMatrix) -> Result<()> {
        if !Arc::ptr_eq(&self.0, &matrix.gpu.0) {
            return Err(Error::InvalidInput(
                "matrix belongs to a different GPU session".into(),
            ));
        }
        Ok(())
    }

    fn zeros(&self, rows: usize, cols: usize) -> Result<DeviceMatrix> {
        dimension(rows)?;
        dimension(cols)?;
        let len = rows
            .checked_mul(cols)
            .filter(|&len| len <= isize::MAX as usize / size_of::<Complex>())
            .ok_or_else(|| {
                Error::InvalidInput("matrix allocation overflows address space".into())
            })?;
        Ok(DeviceMatrix {
            gpu: self.clone(),
            values: self.0.stream.alloc_zeros(len).map_err(cuda)?,
            rows,
            cols,
        })
    }

    /// Upload a finite, nonempty CPU matrix without changing its precision/layout.
    pub fn upload(&self, value: &DMatrix<Complex>) -> Result<DeviceMatrix> {
        dimension(value.nrows())?;
        dimension(value.ncols())?;
        if value.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
            return Err(Error::InvalidInput("matrix entries must be finite".into()));
        }
        self.upload_validated(value)
    }

    fn upload_validated(&self, value: &DMatrix<Complex>) -> Result<DeviceMatrix> {
        // SAFETY: num_complex::Complex is repr(C), containing re then im;
        // [f64;2] has the same size/alignment. The shared slice cannot outlive
        // value. CUDA storage uses its own aligned allocation and owns its copy.
        let values =
            unsafe { std::slice::from_raw_parts(value.as_ptr().cast::<[f64; 2]>(), value.len()) };
        Ok(DeviceMatrix {
            gpu: self.clone(),
            values: self.0.stream.clone_htod(values).map_err(cuda)?,
            rows: value.nrows(),
            cols: value.ncols(),
        })
    }

    /// Download and synchronize, rejecting non-finite numerical results.
    pub fn download(&self, value: &DeviceMatrix) -> Result<DMatrix<Complex>> {
        self.check_owner(value)?;
        let host = self.0.stream.clone_dtoh(&value.values).map_err(cuda)?;
        if host.iter().flatten().any(|x| !x.is_finite()) {
            return Err(Error::Cuda("non-finite matrix result".into()));
        }
        let mut host = std::mem::ManuallyDrop::new(host);
        // SAFETY: Complex<f64> and [f64;2] have identical repr(C) fields, size,
        // alignment and no destructor. Preserve the allocation's length/capacity
        // and transfer its unique ownership rather than copying an entire result.
        let values = unsafe {
            Vec::from_raw_parts(
                host.as_mut_ptr().cast::<Complex>(),
                host.len(),
                host.capacity(),
            )
        };
        Ok(DMatrix::from_vec(value.rows, value.cols, values))
    }

    /// Multiply two resident matrices with NVIDIA's double-complex GEMM.
    pub fn matmul(&self, left: &DeviceMatrix, right: &DeviceMatrix) -> Result<DeviceMatrix> {
        self.matmul_with_adjoint(left, right, false, false)
    }

    /// Multiply resident matrices, optionally conjugate-transposing either input.
    ///
    /// Adjoint operands share their existing storage. In particular, `Fᴴ g`
    /// computes the coefficient pullback of a fixed sampling operator without
    /// allocating or uploading a transposed copy of `F`.
    pub fn matmul_with_adjoint(
        &self,
        left: &DeviceMatrix,
        right: &DeviceMatrix,
        adjoint_left: bool,
        adjoint_right: bool,
    ) -> Result<DeviceMatrix> {
        self.product(left, right, adjoint_left, adjoint_right, 1.0)
    }

    fn scale_rows(&self, matrix: &mut DeviceMatrix, scales: &DeviceMatrix) -> Result<()> {
        let stream = &self.0.stream;
        let (a, a_guard) = matrix.values.device_ptr_mut(stream);
        let (x, _x_guard) = scales.values.device_ptr(stream);
        // SAFETY: internal scales have one entry per row and share this session.
        // cuBLAS documents A=C as supported when lda=ldc, as used here.
        unsafe {
            blas::cublasZdgmm(
                *self.0.blas.handle(),
                blas::cublasSideMode_t::CUBLAS_SIDE_LEFT,
                dimension(matrix.rows)?,
                dimension(matrix.cols)?,
                a as *const blas::cuDoubleComplex,
                dimension(matrix.rows)?,
                x as *const blas::cuDoubleComplex,
                1,
                a as *mut blas::cuDoubleComplex,
                dimension(matrix.rows)?,
            )
            .result()
            .map_err(cuda)?;
        }
        drop(a_guard);
        Ok(())
    }

    /// Equilibrate with the shared CPU rule, then upload and factor.
    ///
    /// Prefer this for host-built multipole operators. The scaling is retained
    /// on device and applied to normal solves and adjoints without round trips.
    pub fn factor_host(&self, mut operator: DMatrix<Complex>) -> Result<CudaLu> {
        let scales = linalg::equilibrate(&mut operator).map_err(|error| match error {
            treams_core::Error::Singular => Error::Singular(0),
            other => Error::InvalidInput(other.to_string()),
        })?;
        let mut factor = self.factor(self.upload_validated(&operator)?)?;
        if let Some(scales) = scales {
            let upload = |values: Vec<f64>| {
                self.upload(&DMatrix::from_iterator(
                    values.len(),
                    1,
                    values.into_iter().map(|x| Complex::new(x, 0.0)),
                ))
            };
            factor.scales = Some((upload(scales.row)?, upload(scales.column)?));
        }
        Ok(factor)
    }

    fn product(
        &self,
        left: &DeviceMatrix,
        right: &DeviceMatrix,
        adjoint_left: bool,
        adjoint_right: bool,
        scale: f64,
    ) -> Result<DeviceMatrix> {
        self.check_owner(left)?;
        self.check_owner(right)?;
        let (m, k) = if adjoint_left {
            (left.cols, left.rows)
        } else {
            left.shape()
        };
        let (other_k, n) = if adjoint_right {
            (right.cols, right.rows)
        } else {
            right.shape()
        };
        if k != other_k {
            return Err(Error::InvalidInput(
                "matrix product dimensions do not match".into(),
            ));
        }
        let mut output = self.zeros(m, n)?;
        let (a, _a_guard) = left.values.device_ptr(&self.0.stream);
        let (b, _b_guard) = right.values.device_ptr(&self.0.stream);
        let (c, c_guard) = output.values.device_ptr_mut(&self.0.stream);
        let alpha = blas::cuDoubleComplex { x: scale, y: 0.0 };
        let beta = blas::cuDoubleComplex { x: 0.0, y: 0.0 };
        let operation = |adjoint| {
            if adjoint {
                blas::cublasOperation_t::CUBLAS_OP_C
            } else {
                blas::cublasOperation_t::CUBLAS_OP_N
            }
        };
        // SAFETY: all matrices share this stream, sizes/leading dimensions are
        // checked, inputs are immutably borrowed and output is a fresh allocation.
        // Device pointer guards retain event-tracked borrows through submission.
        unsafe {
            blas::cublasZgemm_v2(
                *self.0.blas.handle(),
                operation(adjoint_left),
                operation(adjoint_right),
                dimension(m)?,
                dimension(n)?,
                dimension(k)?,
                &raw const alpha,
                a as *const blas::cuDoubleComplex,
                dimension(left.rows)?,
                b as *const blas::cuDoubleComplex,
                dimension(right.rows)?,
                &raw const beta,
                c as *mut blas::cuDoubleComplex,
                dimension(m)?,
            )
            .result()
            .map_err(cuda)?;
        }
        drop(c_guard);
        Ok(output)
    }

    /// Factor an owned square matrix in place; retain LU and pivots on the GPU.
    ///
    /// This resident path does not equilibrate. Use [`Self::factor_host`] for a
    /// host-built operator with highly unequal multipole scales.
    pub fn factor(&self, mut operator: DeviceMatrix) -> Result<CudaLu> {
        self.check_owner(&operator)?;
        if operator.rows != operator.cols {
            return Err(Error::InvalidInput("LU requires a square operator".into()));
        }
        let n = dimension(operator.rows)?;
        let stream = &self.0.stream;
        let mut pivots = stream.alloc_zeros::<i32>(operator.rows).map_err(cuda)?;
        let mut info = stream.alloc_zeros::<i32>(1).map_err(cuda)?;
        let mut length = 0;
        let (a, a_guard) = operator.values.device_ptr_mut(stream);
        // SAFETY: query receives a valid n×n column-major allocation and a
        // writable host integer; it does not modify the matrix.
        unsafe {
            solver::cusolverDnZgetrf_bufferSize(
                self.0.solver.cu(),
                n,
                n,
                a as *mut solver::cuDoubleComplex,
                n,
                &raw mut length,
            )
            .result()
            .map_err(cuda)?;
        }
        let mut workspace = stream
            .alloc_zeros::<[f64; 2]>(usize::try_from(length).map_err(cuda)?)
            .map_err(cuda)?;
        let (work, work_guard) = workspace.device_ptr_mut(stream);
        let (ipiv, pivot_guard) = pivots.device_ptr_mut(stream);
        let (status, info_guard) = info.device_ptr_mut(stream);
        // SAFETY: dimensions and workspace follow the library's exact query;
        // owned LU, workspace, pivots and status are disjoint device allocations.
        unsafe {
            solver::cusolverDnZgetrf(
                self.0.solver.cu(),
                n,
                n,
                a as *mut solver::cuDoubleComplex,
                n,
                work as *mut solver::cuDoubleComplex,
                ipiv as *mut i32,
                status as *mut i32,
            )
            .result()
            .map_err(cuda)?;
        }
        drop((a_guard, work_guard, pivot_guard, info_guard));
        self.check_info(&info)?;
        Ok(CudaLu {
            operator,
            pivots,
            scales: None,
        })
    }

    fn check_info(&self, info: &CudaSlice<i32>) -> Result<()> {
        let code = self
            .0
            .stream
            .clone_dtoh(info)
            .map_err(cuda)?
            .into_iter()
            .next()
            .unwrap_or(-1);
        if code > 0 {
            return Err(Error::Singular(code));
        }
        if code < 0 {
            return Err(Error::Cuda(format!("invalid LAPACK argument {}", -code)));
        }
        Ok(())
    }
}

/// Reusable pivoted LU; subsequent illuminations need only triangular solves.
#[derive(Debug)]
pub struct CudaLu {
    operator: DeviceMatrix,
    pivots: CudaSlice<i32>,
    scales: Option<(DeviceMatrix, DeviceMatrix)>,
}

impl CudaLu {
    /// Square operator dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.operator.rows
    }

    /// Resident factors and pivot storage, excluding temporary library workspace.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.operator.bytes()
            + self.dimension() * size_of::<i32>()
            + self
                .scales
                .as_ref()
                .map_or(0, |(row, column)| row.bytes() + column.bytes())
    }

    /// Solve A X=B, or Aᴴ X=B for an adjoint, overwriting the owned RHS on device.
    pub fn solve(&self, mut rhs: DeviceMatrix, adjoint: bool) -> Result<DeviceMatrix> {
        let gpu = &self.operator.gpu;
        gpu.check_owner(&rhs)?;
        if rhs.rows != self.dimension() {
            return Err(Error::InvalidInput(
                "LU and RHS dimensions do not match".into(),
            ));
        }
        if let Some((row, column)) = &self.scales {
            gpu.scale_rows(&mut rhs, if adjoint { column } else { row })?;
        }
        let stream = &gpu.0.stream;
        let mut info = stream.alloc_zeros::<i32>(1).map_err(cuda)?;
        let (a, a_guard) = self.operator.values.device_ptr(stream);
        let (b, b_guard) = rhs.values.device_ptr_mut(stream);
        let (ipiv, pivot_guard) = self.pivots.device_ptr(stream);
        let (status, info_guard) = info.device_ptr_mut(stream);
        let op = if adjoint {
            solver::cublasOperation_t::CUBLAS_OP_C
        } else {
            solver::cublasOperation_t::CUBLAS_OP_N
        };
        // SAFETY: same-stream n×n LU and n pivots came from successful getrf;
        // RHS is uniquely owned with n rows and the validated number of columns.
        unsafe {
            solver::cusolverDnZgetrs(
                gpu.0.solver.cu(),
                op,
                dimension(self.dimension())?,
                dimension(rhs.cols)?,
                a as *const solver::cuDoubleComplex,
                dimension(self.dimension())?,
                ipiv as *const i32,
                b as *mut solver::cuDoubleComplex,
                dimension(rhs.rows)?,
                status as *mut i32,
            )
            .result()
            .map_err(cuda)?;
        }
        drop((a_guard, b_guard, pivot_guard, info_guard));
        gpu.check_info(&info)?;
        if let Some((row, column)) = &self.scales {
            gpu.scale_rows(&mut rhs, if adjoint { row } else { column })?;
        }
        Ok(rhs)
    }

    /// Implicit pullback `(dA, dB)=(-A⁻ᴴ g Xᴴ, A⁻ᴴ g)`, all on device.
    ///
    /// The native pairing is `dL = Re(sum(conj(g) * dX))`, shared with the CPU.
    pub fn pullback(
        &self,
        value: &DeviceMatrix,
        cotangent: DeviceMatrix,
    ) -> Result<(DeviceMatrix, DeviceMatrix)> {
        if value.shape() != cotangent.shape() {
            return Err(Error::InvalidInput(
                "solution and cotangent shapes differ".into(),
            ));
        }
        let rhs = self.solve(cotangent, true)?;
        let operator = self.operator.gpu.product(&rhs, value, false, true, -1.0)?;
        Ok((operator, rhs))
    }
}
