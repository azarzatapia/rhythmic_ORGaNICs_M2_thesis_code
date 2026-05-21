import numpy as np

class MakeMatrices:

    def __init__(
        self,
        Ntheta=12,
        overlap_theta=2,
        overlap_xi=2,
        FOV_deg=(-60.0, 60.0),
        theta_res_factor=10,   
        xi_res_factor=10,      
        Nxi_base=100,
        magnification_lambda=0.9,
        magnification_power=3.5,
        apply_cortical_magnification=False,
    ):
        self.Ntheta = int(Ntheta)
        self.overlap_theta = float(overlap_theta)
        self.overlap_xi = float(overlap_xi)

        self.FOV_rad = (np.deg2rad(float(FOV_deg[0])), np.deg2rad(float(FOV_deg[1])))

        self.theta_res_factor = float(theta_res_factor)
        self.xi_res_factor = float(xi_res_factor)
        self.Nxi_base = int(Nxi_base)

        self.apply_cortical_magnification = bool(apply_cortical_magnification)
        self.magnification_lambda = float(magnification_lambda)
        self.magnification_power = float(magnification_power)

        self.delta_theta = np.pi / (self.theta_res_factor * self.Ntheta)
        self.theta = np.arange(-np.pi / 2, np.pi / 2, self.delta_theta)
        self.theta_deg = np.rad2deg(self.theta)
        self.Mtheta = int(self.theta.size)

        spacing_theta = np.pi / self.Ntheta
        self.theta_preferred = np.linspace(-np.pi / 2, np.pi / 2 - spacing_theta, self.Ntheta)
        self.theta_preferred_deg = np.rad2deg(self.theta_preferred)

        self.Nxi_full = int(self.Nxi_base * self.overlap_xi)
        self.delta_xi = np.pi / (self.xi_res_factor * self.Nxi_full)
        self.xi_full = np.arange(-np.pi, np.pi, self.delta_xi)  # [-pi, pi)
        self.Mxi_full = int(self.xi_full.size)

        spacing_xi = 2 * np.pi / self.Nxi_full
        self.xi_preferred_full = np.linspace(-np.pi, np.pi - spacing_xi, self.Nxi_full)

        self.xi = None
        self.xi_deg = None
        self.xi_preferred = None
        self.xi_preferred_deg = None
        self.Nxi = None
        self.fov_mask = None

        self.thetaRFs = None
        self.xiRFs = None
        self.Wtheta = None
        self.Wxi = None

    # -----------------------------
    # HELPERS
    # -----------------------------
    def _warp(self, x):
        lam = self.magnification_lambda
        p = self.magnification_power
        u = np.abs(x) / np.pi
        return np.pi * np.sign(x) * (lam * (u**p) + (1 - lam) * u)

    # -----------------------------
    # ORIENTATION RFs
    # -----------------------------
    def make_orientation_rfs(self):
        theta_scaled = self.theta * (self.Ntheta / 4) / self.overlap_theta
        base_rf = np.cos(theta_scaled) ** 2
        base_rf[np.abs(theta_scaled) >= (np.pi / 2)] = 0.0

        RFs = np.zeros((self.Ntheta, self.Mtheta))
        for i in range(self.Ntheta):
            sh = -(self.Mtheta // 2) + i * (self.Mtheta // self.Ntheta)
            RFs[i, :] = np.roll(base_rf, sh)

        scale = np.sqrt(np.mean(np.sum(RFs**2, axis=0)) + 1e-12)
        RFs /= scale

        self.thetaRFs = RFs
        return RFs

    # -----------------------------
    # SPATIAL RFs
    # -----------------------------
    def make_spatial_rfs(self):
        xi = self.xi_full
        Nxi = self.Nxi_full
        Mxi = self.Mxi_full

        xi_scaled = xi * (Nxi / 8) / self.overlap_xi
        base_rf = np.cos(xi_scaled) ** 2
        base_rf[np.abs(xi_scaled) >= (np.pi / 2)] = 0.0

        RFs_full = np.zeros((Nxi, Mxi))
        for i in range(Nxi):
            sh = -(Mxi // 2) + i * (Mxi // Nxi)
            RFs_full[i, :] = np.roll(base_rf, sh)

        scale = np.mean(np.sum(RFs_full, axis=1)) + 1e-12
        RFs_full /= scale

        if self.apply_cortical_magnification:
            xi_w = self._warp(self.xi_full)
            xiPref_w = self._warp(self.xi_preferred_full)
        else:
            xi_w = self.xi_full.copy()
            xiPref_w = self.xi_preferred_full.copy()

        lo, hi = self.FOV_rad
        ID = (xiPref_w >= lo) & (xiPref_w <= hi)
        self.fov_mask = ID

        self.xi = xi_w
        self.xi_deg = np.rad2deg(xi_w)
        self.xi_preferred = xiPref_w[ID]
        self.xi_preferred_deg = np.rad2deg(self.xi_preferred)
        self.Nxi = int(self.xi_preferred.size)

        self.xiRFs = RFs_full[ID, :]
        return self.xiRFs
    

    # -----------------------------
    # RECURRENCE MATRICES
    # -----------------------------
    def make_recurrence_matrix(self, recurrence_type="identity"):
        if recurrence_type == "identity":
            return np.eye(self.Ntheta)
        elif recurrence_type == "2D_identity":
            if self.Nxi is None:
                raise ValueError("Call make_spatial_rfs() first so Nxi is known for 2D_identity.")
            return np.eye(self.Ntheta * self.Nxi)
        else:
            raise ValueError(f"Unknown recurrence_type: {recurrence_type}")

    # -----------------------------
    # NORMALIZATION WEIGHTS
    # -----------------------------
    def make_orientation_norm_weights(self):
        self.Wtheta = np.ones((self.Ntheta, self.Ntheta))
        return self.Wtheta

    def make_spatial_norm_weights(self):
        Nxi = self.Nxi_full

        overlap = 4 * self.overlap_xi
        xi_pref_scaled = self.xi_preferred_full * (Nxi / 8) / overlap

        rf = np.cos(xi_pref_scaled) ** 2
        rf[np.abs(xi_pref_scaled) >= (np.pi / 2)] = 0.0

        Wxi_full = np.zeros((Nxi, Nxi))
        for i in range(Nxi):
            sh = int(-(Nxi // 2) + i)
            Wxi_full[i, :] = np.roll(rf, sh)

        Wxi_full /= max(self.overlap_xi, 1e-12)

        if self.fov_mask is None:
            self.make_spatial_rfs()

        ID = self.fov_mask
        self.Wxi = Wxi_full[np.ix_(ID, ID)]
        return self.Wxi

    def combine_normalization_weights(self, scale_spatial=0.5):
        if self.Wtheta is None:
            self.make_orientation_norm_weights()
        if self.Wxi is None:
            self.make_spatial_norm_weights()

        Wxi = scale_spatial * self.Wxi
        Wtheta = self.Wtheta
        W = np.kron(Wtheta, Wxi)

        Ntheta = Wtheta.shape[0]
        Nxi = Wxi.shape[0]

        ztmp = np.zeros((Nxi, Ntheta))
        ztmp[:, 0] = 1.0
        zvec = ztmp.reshape(-1, order="F")

        Wscale = np.max(W @ zvec)
        self.Wnorm = W / (Wscale + 1e-9)
        # Store the scaled spatial kernel for the Kronecker fast path in the ODE.
        # Wnorm = kron(ones(Ntheta,Ntheta), Wxi_norm), so the dense N×N matvec
        # can be replaced by a cheap sum + (Nxi×Nxi) matvec (see organics_dynamics.py).
        self.Wxi_norm = Wxi / (Wscale + 1e-9)
        return self.Wnorm