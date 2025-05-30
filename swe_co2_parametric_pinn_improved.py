# -*- coding: utf-8 -*-
# swe_co2_parametric_pinn_improved.py
import torch
import numpy as np
import sympy
from sympy import Symbol, Function, Abs, log, exp, tanh
import os
import omegaconf.errors
import shutil
import hydra.utils
import sys

# Import Modulus modules
import modulus.sym
from modulus.sym.hydra import to_yaml, to_absolute_path, instantiate_arch
from modulus.sym.hydra.config import ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry import Parameter, Parameterization
from modulus.sym.geometry.primitives_1d import Line1D
from modulus.sym.geometry.primitives_2d import Rectangle
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    PointwiseConstraint,
)
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.eq.pde import PDE

# --- 1. Define Improved PDE Class with Scaling ---
class ConservativeSWE_CO2_Scaled(PDE):
    """改良版: h_initに応じた動的スケーリングを含むPDE"""
    name = "ConservativeSWE_CO2_Scaled"
    
    def __init__(self, Cf, D, C0, h_epsilon=1e-7):
        x, y = Symbol("x"), Symbol("y")
        self.x, self.y = x, y
        
        h_init_param = Symbol("h_init")
        
        # Network outputs
        h_net = Function("h_net")(x, y, h_init_param)
        u_net = Function("u_net")(x, y, h_init_param)
        c_net = Function("c_net")(x, y, h_init_param)
        
        # Transform network outputs
        h = log(1 + exp(h_net))
        u = u_net
        c = 1 / (1 + exp(-c_net))
        
        self.Cf = Cf
        self.D = D
        self.C0 = C0
        self.h_epsilon = h_epsilon
        
        # Compute derivatives
        h_t = h.diff(y)
        h_x = h.diff(x)
        u_t = u.diff(y)
        u_x = u.diff(x)
        
        # Conservation form
        q = h * u
        E_cont = q
        E_mom = q**2 / h + 0.5 * h**2
        
        # Source terms with h_init-dependent scaling
        # 摩擦項をh_initでスケーリング（大きな初期高さでの安定性向上）
        scaling_factor = 1.0 / (1.0 + 0.5 * h_init_param)  # h_initが大きいほど摩擦を減らす
        friction_source = self.Cf * u * Abs(u) * scaling_factor
        
        # PDE equations
        self.equations = {}
        self.equations["continuity"] = h_t + E_cont.diff(x)
        self.equations["momentum"] = q.diff(y) + E_mom.diff(x) + friction_source
        
        # Transport equation
        c_t = c.diff(y)
        c_xx = c.diff(x, 2)
        uc_x = (u * c).diff(x)
        diffusion = self.D * c_xx
        self.equations["transport"] = c_t + uc_x - diffusion

# --- 2. Adaptive Loss Weighting Class ---
class AdaptiveLossWeighting:
    """h_initに応じて損失重みを動的に調整"""
    def __init__(self, base_weights, h_init_ref=0.5):
        self.base_weights = base_weights
        self.h_init_ref = h_init_ref
        
    def get_weights(self, h_init_range):
        """h_initの範囲に基づいて重みを調整"""
        h_init_mid = (h_init_range[0] + h_init_range[1]) / 2.0
        scale_factor = h_init_mid / self.h_init_ref
        
        # 大きなh_initではPDE残差の重みを増やし、IC/BCの重みを相対的に減らす
        adjusted_weights = {
            'lambda_ic': self.base_weights['lambda_ic'] / np.sqrt(scale_factor),
            'lambda_bc': self.base_weights['lambda_bc'] / np.sqrt(scale_factor),
            'lambda_f_h': self.base_weights['lambda_f_h'] * scale_factor,
            'lambda_f_u': self.base_weights['lambda_f_u'] * scale_factor,
            'lambda_f_c': self.base_weights['lambda_f_c'] * np.sqrt(scale_factor)
        }
        
        return adjusted_weights

# --- Main Hydra Script ---
@modulus.sym.main(config_path="conf", config_name="config_parametric_improved")
def run(cfg: ModulusConfig) -> None:
    print(f"Starting Improved Parametric Conservative SWE+CO2 Simulation with config:\n{to_yaml(cfg)}")
    
    # --- 1. Load Parameters ---
    try:
        L_dim = cfg.custom.physical.L_dim
        T_dim = cfg.custom.physical.T_dim
        H_ref_dim = cfg.custom.physical.H_ref_dim
        h_init_min_dim = cfg.custom.physical.h_init_min_dim
        h_init_max_dim = cfg.custom.physical.h_init_max_dim
        g_accel = cfg.custom.physical.g_accel
        dam_pos_dim = cfg.custom.physical.dam_pos_dim
        C0 = cfg.custom.physical.C0_init
        Cf = cfg.custom.physical.Cf_param
        D_physical = cfg.custom.physical.D_physical
        h_epsilon = cfg.custom.h_epsilon_stab
        x_min_nd = cfg.custom.domain.x_min_nd
        x_max_nd = cfg.custom.domain.x_max_nd
        t_min_nd = cfg.custom.domain.t_min_nd
        
        # 新規: 段階的学習パラメータ
        use_progressive_training = cfg.custom.get('use_progressive_training', True)
        progressive_stages = cfg.custom.get('progressive_stages', 3)
        
    except omegaconf.errors.ConfigAttributeError as e:
        print(f"\n*** Error loading parameters: {e}")
        return
    
    # --- Calculate Derived Parameters ---
    print("\n--- Calculating Derived Parameters ---")
    L_char = L_dim
    H_char_for_scaling = H_ref_dim
    V_char_for_scaling = np.sqrt(g_accel * H_char_for_scaling)
    T_char_for_scaling = L_char / V_char_for_scaling if V_char_for_scaling > 1e-9 else 0
    
    h_init_nd_min = h_init_min_dim / H_char_for_scaling
    h_init_nd_max = h_init_max_dim / H_char_for_scaling
    
    D_param_calc = D_physical / (L_char * V_char_for_scaling) if (L_char * V_char_for_scaling) > 1e-9 else 0
    dam_pos_nd_calc = dam_pos_dim / L_char if L_char > 1e-9 else 0
    t_max_nd_calc = T_dim / T_char_for_scaling if T_char_for_scaling > 1e-9 else 0
    
    print(f"Characteristic Scales: L={L_char:.1f}mm, H_ref={H_char_for_scaling:.1f}mm")
    print(f"Non-dim h_init range: [{h_init_nd_min:.4f}, {h_init_nd_max:.4f}]")
    
    # --- Define Adaptive Weights ---
    base_weights = {
        'lambda_ic': 20.0,  # 増加: より強い初期条件制約
        'lambda_bc': 15.0,  # 増加: より強い境界条件制約
        'lambda_f_h': 200.0,  # 減少: 過度な制約を避ける
        'lambda_f_u': 200.0,
        'lambda_f_c': 300.0
    }
    
    adaptive_weighting = AdaptiveLossWeighting(base_weights, h_init_ref=0.5)
    
    # --- 2. Setup PDE and Network ---
    pde = ConservativeSWE_CO2_Scaled(Cf=Cf, D=D_param_calc, C0=C0, h_epsilon=h_epsilon)
    x_sympy, y_sympy = pde.x, pde.y
    h_init_sympy = Symbol("h_init")
    
    # 改良: より大きなネットワーク容量
    # Modulusのデフォルト設定を使用し、カスタマイズが必要な場合はコード側で対応
    net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("h_init")],
        output_keys=[Key("h_net"), Key("u_net"), Key("c_net")],
        cfg=cfg.arch.fully_connected
    )
    
    # もしカスタム活性化関数が必要な場合は、ここでネットワークを修正
    # 例: ネットワークの各層にSiLU活性化を適用（Modulusがサポートしている場合）
    # ただし、デフォルトのtanhでも十分な性能が得られる可能性がある
    
    # Define symbolic expressions
    h_net_func = Function("h_net")(x_sympy, y_sympy, h_init_sympy)
    u_net_func = Function("u_net")(x_sympy, y_sympy, h_init_sympy)
    c_net_func = Function("c_net")(x_sympy, y_sympy, h_init_sympy)
    
    h_expr = log(1 + exp(h_net_func))
    u_expr = u_net_func
    c_expr = 1 / (1 + exp(-c_net_func))
    
    # Gradients for BC
    h_x_expr = sympy.diff(h_expr, x_sympy)
    u_x_expr = sympy.diff(u_expr, x_sympy)
    c_x_expr = sympy.diff(c_expr, x_sympy)
    
    # Create nodes
    h_node = Node.from_sympy(h_expr, "h")
    u_node = Node.from_sympy(u_expr, "u")
    c_node = Node.from_sympy(c_expr, "c")
    h_x_node = Node.from_sympy(h_x_expr, "h_x")
    u_x_node = Node.from_sympy(u_x_expr, "u_x")
    c_x_node = Node.from_sympy(c_x_expr, "c_x")
    
    nodes = [net.make_node(name="pinn_network")] + pde.make_nodes() + [
        h_node, u_node, c_node, h_x_node, u_x_node, c_x_node
    ]
    
    # --- 3. Progressive Training Setup ---
    if use_progressive_training:
        print("\n--- Using Progressive Training Strategy ---")
        h_init_stages = []
        for i in range(progressive_stages):
            stage_min = h_init_nd_min + (h_init_nd_max - h_init_nd_min) * i / progressive_stages
            stage_max = h_init_nd_min + (h_init_nd_max - h_init_nd_min) * (i + 1) / progressive_stages
            h_init_stages.append((stage_min, stage_max))
            print(f"Stage {i+1}: h_init range [{stage_min:.4f}, {stage_max:.4f}]")
    else:
        h_init_stages = [(h_init_nd_min, h_init_nd_max)]
    
    # --- 4. Define Domain and Constraints ---
    geo = Rectangle((x_min_nd, t_min_nd), (x_max_nd, t_max_nd_calc))
    
    # Process each training stage
    for stage_idx, (h_init_stage_min, h_init_stage_max) in enumerate(h_init_stages):
        print(f"\n=== Training Stage {stage_idx + 1}/{len(h_init_stages)} ===")
        print(f"h_init range: [{h_init_stage_min:.4f}, {h_init_stage_max:.4f}]")
        
        # Get adaptive weights for this stage
        weights = adaptive_weighting.get_weights((h_init_stage_min, h_init_stage_max))
        print(f"Adaptive weights: {weights}")
        
        # Create domain for this stage
        domain = Domain()
        
        # Parameterization for this stage
        param_h_init_stage = Parameterization({h_init_sympy: (h_init_stage_min, h_init_stage_max)})
        param_y_and_h_init_stage = Parameterization({
            y_sympy: (t_min_nd, t_max_nd_calc),
            h_init_sympy: (h_init_stage_min, h_init_stage_max)
        })
        
        # Interior constraint
        interior = PointwiseInteriorConstraint(
            nodes=nodes,
            geometry=geo,
            outvar={"continuity": 0.0, "momentum": 0.0, "transport": 0.0},
            batch_size=cfg.batch_size.interior,
            lambda_weighting={
                "continuity": weights['lambda_f_h'],
                "momentum": weights['lambda_f_u'],
                "transport": weights['lambda_f_c']
            },
            quasirandom=True,
            parameterization=param_h_init_stage
        )
        domain.add_constraint(interior, "interior")
        
        # Initial condition with smoothed step function
        geo_ic_space = Line1D(x_min_nd, x_max_nd)
        ic_param_stage = Parameterization({
            y_sympy: t_min_nd,
            h_init_sympy: (h_init_stage_min, h_init_stage_max)
        })
        
        x_sympy_ic = Symbol("x")
        downstream_h_ic_nd = 1e-6 / H_char_for_scaling
        
        # 改良: わずかに平滑化された初期条件（tanh関数使用）
        transition_width = 0.05  # 遷移幅
        h_physical_ic_expr = downstream_h_ic_nd + (h_init_sympy - downstream_h_ic_nd) * \
                            0.5 * (1 + tanh((dam_pos_nd_calc - x_sympy_ic) / transition_width))
        
        h_physical_ic_safe = sympy.Max(h_physical_ic_expr, 1e-7)
        h_net_ic_sympy = log(exp(h_physical_ic_safe) - 1)
        u_net_ic_sympy = 0.0
        
        # Concentration initial condition (also smoothed)
        c_physical_ic_expr = 1e-6 + (C0 - 1e-6) * \
                            0.5 * (1 + tanh((dam_pos_nd_calc - x_sympy_ic) / transition_width))
        c_physical_ic_safe = sympy.Min(sympy.Max(c_physical_ic_expr, 1e-6), 1.0 - 1e-6)
        c_net_ic_sympy = log(c_physical_ic_safe / (1.0 - c_physical_ic_safe))
        
        ic_constraint = PointwiseInteriorConstraint(
            nodes=nodes,
            geometry=geo_ic_space,
            outvar={"h_net": h_net_ic_sympy, "u_net": u_net_ic_sympy, "c_net": c_net_ic_sympy},
            batch_size=cfg.batch_size.initial_conditions,
            lambda_weighting={
                "h_net": weights['lambda_ic'],
                "u_net": weights['lambda_ic'],
                "c_net": weights['lambda_ic']
            },
            quasirandom=True,
            parameterization=ic_param_stage
        )
        domain.add_constraint(ic_constraint, "IC")
        
        # Boundary conditions (Neumann)
        criteria_left = sympy.Eq(x_sympy, x_min_nd)
        bc_left = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo,
            criteria=criteria_left,
            outvar={"h_x": 0.0, "u_x": 0.0, "c_x": 0.0},
            batch_size=cfg.batch_size.boundary,
            lambda_weighting={
                "h_x": weights['lambda_bc'],
                "u_x": weights['lambda_bc'],
                "c_x": weights['lambda_bc']
            },
            parameterization=param_y_and_h_init_stage,
            quasirandom=True
        )
        domain.add_constraint(bc_left, "BC_Left")
        
        criteria_right = sympy.Eq(x_sympy, x_max_nd)
        bc_right = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo,
            criteria=criteria_right,
            outvar={"h_x": 0.0, "u_x": 0.0, "c_x": 0.0},
            batch_size=cfg.batch_size.boundary,
            lambda_weighting={
                "h_x": weights['lambda_bc'],
                "u_x": weights['lambda_bc'],
                "c_x": weights['lambda_bc']
            },
            parameterization=param_y_and_h_init_stage,
            quasirandom=True
        )
        domain.add_constraint(bc_right, "BC_Right")
        
        # --- 5. Inferencer for Visualization ---
        N_x_val = 201
        N_t_val = 101
        x_val_np = np.linspace(x_min_nd, x_max_nd, N_x_val)
        y_val_np = np.linspace(t_min_nd, t_max_nd_calc, N_t_val)
        x_mesh_nd, y_mesh_nd = np.meshgrid(x_val_np, y_val_np)
        
        h_init_vtk_nd = (h_init_stage_min + h_init_stage_max) / 2.0
        h_init_vtk_mesh = np.full_like(x_mesh_nd, h_init_vtk_nd)
        
        eval_points_vtk = {
            "x": x_mesh_nd.flatten()[:, None],
            "y": y_mesh_nd.flatten()[:, None],
            "h_init": h_init_vtk_mesh.flatten()[:, None]
        }
        
        inferencer = PointwiseInferencer(
            nodes=nodes,
            invar=eval_points_vtk,
            output_names=["h", "u", "c"],
            batch_size=1024,
            requires_grad=False,
        )
        domain.add_inferencer(inferencer, f"vis_stage_{stage_idx+1}_h_init_{h_init_vtk_nd:.4f}")
        
        # --- 6. Solver for this stage ---
        # Adjust training iterations based on stage
        if stage_idx > 0:
            # Reduce iterations for later stages (transfer learning effect)
            cfg.training.max_steps = int(cfg.training.max_steps * 0.7)
        
        slv = Solver(cfg, domain)
        print(f"Starting solver for stage {stage_idx + 1}...")
        slv.solve()
        print(f"Stage {stage_idx + 1} completed.")
    
    # --- 7. Final Inference and Save Results ---
    print("\n--- Performing final inference for all h_init values ---")
    
    try:
        device = modulus.sym.manager.Manager().device
    except AttributeError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test on full range
    h_init_test_values_nd = np.linspace(h_init_nd_min, h_init_nd_max, 5)
    results_to_save = {
        'x_mesh_nd': x_mesh_nd,
        'y_mesh_nd': y_mesh_nd,
        'h_init_test_values_nd': h_init_test_values_nd,
        'H_ref_dim': H_ref_dim,
        'L_char': L_char,
        'V_ref_char': V_char_for_scaling,
        'T_ref_char': T_char_for_scaling,
        'T_dim_simulation': T_dim,
        'adaptive_weights_used': base_weights,
        'progressive_training': use_progressive_training,
        'num_stages': len(h_init_stages)
    }
    
    net.eval()
    for i, h_init_val_nd in enumerate(h_init_test_values_nd):
        print(f"Inferring for h_init_nd = {h_init_val_nd:.4f} (h_init_dim = {h_init_val_nd * H_char_for_scaling:.1f}mm)")
        
        h_init_current_mesh = np.full_like(x_mesh_nd, h_init_val_nd)
        current_eval_points_numpy = {
            "x": x_mesh_nd.flatten()[:, None],
            "y": y_mesh_nd.flatten()[:, None],
            "h_init": h_init_current_mesh.flatten()[:, None]
        }
        
        current_eval_points_tensor = {
            key: torch.tensor(arr, dtype=torch.float32).to(device)
            for key, arr in current_eval_points_numpy.items()
        }
        
        with torch.no_grad():
            pred_tensor_dict = net(current_eval_points_tensor)
        
        h_net_pred = pred_tensor_dict['h_net']
        u_net_pred = pred_tensor_dict['u_net']
        c_net_pred = pred_tensor_dict['c_net']
        
        h_pred = torch.nn.functional.softplus(h_net_pred).detach().cpu().numpy()
        u_pred = u_net_pred.detach().cpu().numpy()
        c_pred = torch.sigmoid(c_net_pred).detach().cpu().numpy()
        
        # Reshape back to grid
        h_pred_grid = h_pred.reshape(y_mesh_nd.shape)
        u_pred_grid = u_pred.reshape(y_mesh_nd.shape)
        c_pred_grid = c_pred.reshape(y_mesh_nd.shape)
        
        results_to_save[f'h_pred_nd_{i}'] = h_pred_grid
        results_to_save[f'u_pred_nd_{i}'] = u_pred_grid
        results_to_save[f'c_pred_nd_{i}'] = c_pred_grid
        
        # Quick validation check
        h_max = np.max(h_pred_grid)
        h_min = np.min(h_pred_grid)
        print(f"  h range: [{h_min:.6f}, {h_max:.6f}]")
        
        # Check for unphysical behavior
        h_final = h_pred_grid[-1, :]  # Final time
        h_initial_upstream = h_pred_grid[0, 0]  # Initial upstream
        if np.max(h_final) > h_initial_upstream * 1.1:
            print(f"  WARNING: Potential unphysical increase in h detected!")
    
    # Save results
    output_dir = os.getcwd()
    results_filename = os.path.join(output_dir, "inference_results_parametric_improved.npz")
    
    try:
        np.savez(results_filename, **results_to_save)
        print(f"\nSaved all inference results to: {results_filename}")
    except Exception as e:
        print(f"Error saving results: {e}")
        import traceback
        traceback.print_exc()
    
    # --- 8. Copy files ---
    print("\n--- Copying script and config to output directory ---")
    try:
        script_path = os.path.abspath(__file__)
        if os.path.exists(script_path):
            shutil.copy2(script_path, output_dir)
            print(f"Copied script to {output_dir}")
        
        config_name = cfg.hydra.job.config_name + ".yaml"
        original_cwd = hydra.utils.get_original_cwd()
        config_path = os.path.join(original_cwd, "conf", config_name)
        if os.path.exists(config_path):
            shutil.copy2(config_path, output_dir)
            print(f"Copied config to {output_dir}")
    except Exception as e:
        print(f"Error copying files: {e}")

if __name__ == "__main__":
    run()
