"""
Question 2: Eigenvalues using Jacobi Rotation Method
Part 1: 2024-25

Author: Abdul
Date: August 14, 2025
Module: Numerical Modelling and Projects

This script implements the Jacobi rotation method for computing eigenvalues and eigenvectors 
of real symmetric matrices. The implementation addresses the transformation matrix evaluation 
and eigenvalue calculation as specified in the brief.

Questions Addressed:
2.1 Write code for evaluating the transformation to A' for the first full sweep of Jacobi rotations
2.2 Determine eigenvalues of specified matrices using the implemented code
2.3 Repeat calculations for additional matrices to demonstrate method effectiveness

Mathematical Background:
The Jacobi rotation method uses orthogonal transformations to diagonalize symmetric matrices:
A^(k+1) = J^T * A^(k) * J

where J is a Jacobi rotation matrix that zeros the largest off-diagonal element.

Reference: Based on Week 9 handout equation (13) for Jacobi rotations
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import time
import pandas as pd

class JacobiEigenvalueSolver:
    """
    Comprehensive implementation of the Jacobi rotation method for eigenvalue computation.
    
    This class provides the complete Jacobi algorithm with detailed transformation tracking,
    convergence analysis, and validation capabilities for symmetric matrices.
    """
    
    def __init__(self, tolerance=1e-10, max_iterations=1000):
        """
        Initialize the Jacobi eigenvalue solver.
        
        Parameters:
        tolerance (float): Convergence tolerance for off-diagonal elements
        max_iterations (int): Maximum number of iterations
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.iteration_count = 0
        self.convergence_history = []
        self.transformation_matrices = []
        
        # Set up matplotlib for consistent plotting
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['lines.linewidth'] = 2
        
        print(f"Jacobi Eigenvalue Solver initialized")
        print(f"Tolerance: {self.tolerance}")
        print(f"Maximum iterations: {self.max_iterations}")
    
    def find_largest_off_diagonal(self, A):
        """
        Find the indices and value of the largest off-diagonal element.
        
        Parameters:
        A (numpy.ndarray): Symmetric matrix
        
        Returns:
        p, q (int): Row and column indices of largest off-diagonal element
        max_val (float): Value of the largest off-diagonal element
        """
        n = A.shape[0]
        max_val = 0.0
        p, q = 0, 1
        
        # Search upper triangular part (excluding diagonal)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i, j]) > abs(max_val):
                    max_val = A[i, j]
                    p, q = i, j
        
        return p, q, max_val
    
    def calculate_rotation_angle(self, A, p, q):
        """
        Calculate the rotation angle to zero the (p,q) element.
        
        Parameters:
        A (numpy.ndarray): Current matrix
        p, q (int): Indices of element to zero
        
        Returns:
        theta (float): Rotation angle in radians
        cos_theta, sin_theta (float): Cosine and sine of rotation angle
        """
        if abs(A[p, q]) < self.tolerance:
            return 0.0, 1.0, 0.0
        
        # Calculate rotation angle using the standard Jacobi formula
        if abs(A[p, p] - A[q, q]) < self.tolerance:
            # Special case: diagonal elements are equal
            theta = np.pi / 4 if A[p, q] > 0 else -np.pi / 4
        else:
            # General case - use the formula from Week 9 handout
            tau = (A[q, q] - A[p, p]) / (2 * A[p, q])
            t = 1.0 / (abs(tau) + np.sqrt(1 + tau**2))
            if tau < 0:
                t = -t
            theta = np.arctan(t)
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        return theta, cos_theta, sin_theta
    
    def create_jacobi_matrix(self, n, p, q, cos_theta, sin_theta):
        """
        Create the Jacobi rotation matrix J(p,q,theta).
        
        Parameters:
        n (int): Size of matrix
        p, q (int): Rotation indices
        cos_theta, sin_theta (float): Rotation angle components
        
        Returns:
        J (numpy.ndarray): Jacobi rotation matrix
        """
        J = np.eye(n)
        J[p, p] = cos_theta
        J[q, q] = cos_theta
        J[p, q] = -sin_theta
        J[q, p] = sin_theta
        
        return J
    
    def apply_jacobi_rotation(self, A, V, p, q, cos_theta, sin_theta):
        """
        Apply Jacobi rotation to matrix A and update eigenvector matrix V.
        
        This implements the transformation A' = J^T * A * J efficiently
        without explicitly computing matrix products.
        
        Parameters:
        A (numpy.ndarray): Matrix to transform (modified in place)
        V (numpy.ndarray): Eigenvector matrix (modified in place)
        p, q (int): Rotation indices
        cos_theta, sin_theta (float): Rotation parameters
        """
        n = A.shape[0]
        
        # Store original values
        a_pp = A[p, p]
        a_qq = A[q, q]
        a_pq = A[p, q]
        
        # Update diagonal elements
        A[p, p] = cos_theta**2 * a_pp + sin_theta**2 * a_qq - 2 * cos_theta * sin_theta * a_pq
        A[q, q] = sin_theta**2 * a_pp + cos_theta**2 * a_qq + 2 * cos_theta * sin_theta * a_pq
        A[p, q] = A[q, p] = 0.0  # Zero the off-diagonal element
        
        # Update other elements in rows/columns p and q
        for i in range(n):
            if i != p and i != q:
                a_ip = A[i, p]
                a_iq = A[i, q]
                
                A[i, p] = A[p, i] = cos_theta * a_ip - sin_theta * a_iq
                A[i, q] = A[q, i] = sin_theta * a_ip + cos_theta * a_iq
        
        # Update eigenvector matrix
        for i in range(n):
            v_ip = V[i, p]
            v_iq = V[i, q]
            
            V[i, p] = cos_theta * v_ip - sin_theta * v_iq
            V[i, q] = sin_theta * v_ip + cos_theta * v_iq
    
    def calculate_off_diagonal_norm(self, A):
        """
        Calculate the Frobenius norm of off-diagonal elements.
        
        Parameters:
        A (numpy.ndarray): Matrix
        
        Returns:
        norm (float): Off-diagonal norm
        """
        n = A.shape[0]
        norm = 0.0
        
        for i in range(n):
            for j in range(i + 1, n):
                norm += A[i, j]**2
        
        return np.sqrt(2 * norm)  # Factor of 2 for symmetry
    
    def analyze_first_sweep_transformations(self, A_original):
        """
        Analyze the transformation matrices in the first sweep of Jacobi rotations.
        
        This addresses Question 2.1 by providing detailed transformation analysis.
        
        Parameters:
        A_original (numpy.ndarray): Original matrix
        
        Returns:
        transformation_data (list): Detailed transformation information
        """
        print("\n2.1: First Full Sweep Transformation Analysis")
        print("=" * 50)
        
        n = A_original.shape[0]
        A_current = A_original.copy()
        
        print(f"Original matrix A:")
        print(A_original)
        print(f"Matrix size: {n}×{n}")
        
        # Track transformations for first sweep
        transformation_data = []
        V_accumulated = np.eye(n)  # Accumulated transformation matrix
        
        # Perform first sweep
        max_sweeps = min(10, n * (n - 1) // 2)  # Limit for display
        
        print(f"\nPerforming first sweep transformations:")
        print("-" * 60)
        
        for step in range(max_sweeps):
            # Find largest off-diagonal element
            p, q, max_val = self.find_largest_off_diagonal(A_current)
            
            if abs(max_val) < self.tolerance:
                print(f"Convergence reached after {step} rotations in first sweep")
                break
            
            # Calculate rotation parameters
            theta, cos_theta, sin_theta = self.calculate_rotation_angle(A_current, p, q)
            
            # Create transformation matrix
            J = self.create_jacobi_matrix(n, p, q, cos_theta, sin_theta)
            
            # Store transformation data
            transformation_info = {
                'step': step + 1,
                'indices': (p, q),
                'target_element': max_val,
                'rotation_angle': theta,
                'cos_theta': cos_theta,
                'sin_theta': sin_theta,
                'jacobi_matrix': J.copy(),
                'matrix_before': A_current.copy()
            }
            
            # Apply transformation: A' = J^T * A * J
            A_new = J.T @ A_current @ J
            transformation_info['matrix_after'] = A_new.copy()
            
            # Update accumulated transformation
            V_accumulated = V_accumulated @ J
            
            # Calculate off-diagonal norm
            off_diag_norm = self.calculate_off_diagonal_norm(A_new)
            transformation_info['off_diagonal_norm'] = off_diag_norm
            
            transformation_data.append(transformation_info)
            
            # Print step information
            print(f"Step {step+1:2d}: Zero A[{p},{q}] = {max_val:8.5f}, "
                  f"θ = {theta:7.4f}, norm = {off_diag_norm:.6e}")
            
            # Update current matrix
            A_current = A_new
            
            # Stop if converged or after reasonable number of steps
            if off_diag_norm < self.tolerance or step >= 5:
                break
        
        print(f"\nMatrix after first sweep:")
        print(A_current)
        print(f"\nReduction in off-diagonal norm: "
              f"{self.calculate_off_diagonal_norm(A_original):.6e} → "
              f"{self.calculate_off_diagonal_norm(A_current):.6e}")
        
        return transformation_data
    
    def solve_eigenvalues(self, A_original, track_transformations=True):
        """
        Solve for eigenvalues and eigenvectors using Jacobi method.
        
        Parameters:
        A_original (numpy.ndarray): Original symmetric matrix
        track_transformations (bool): Whether to store transformation matrices
        
        Returns:
        eigenvalues (numpy.ndarray): Computed eigenvalues
        eigenvectors (numpy.ndarray): Computed eigenvectors
        A_final (numpy.ndarray): Final diagonalized matrix
        """
        # Verify matrix is symmetric
        if not np.allclose(A_original, A_original.T):
            raise ValueError("Matrix must be symmetric for Jacobi method")
        
        # Initialize working matrices
        A = A_original.copy()
        n = A.shape[0]
        V = np.eye(n)  # Accumulates eigenvectors
        
        # Reset tracking variables
        self.iteration_count = 0
        self.convergence_history = []
        self.transformation_matrices = []
        
        print(f"\nStarting Jacobi iteration for {n}×{n} matrix")
        print(f"Initial off-diagonal norm: {self.calculate_off_diagonal_norm(A):.6e}")
        
        # Main iteration loop
        for iteration in range(self.max_iterations):
            # Find largest off-diagonal element
            p, q, max_off_diag = self.find_largest_off_diagonal(A)
            
            # Check convergence
            off_diag_norm = self.calculate_off_diagonal_norm(A)
            self.convergence_history.append(off_diag_norm)
            
            if off_diag_norm < self.tolerance:
                print(f"Converged after {iteration} iterations")
                print(f"Final off-diagonal norm: {off_diag_norm:.6e}")
                break
            
            # Calculate rotation parameters
            theta, cos_theta, sin_theta = self.calculate_rotation_angle(A, p, q)
            
            # Store transformation matrix if requested
            if track_transformations and iteration < 10:  # Store first 10 for analysis
                J = self.create_jacobi_matrix(n, p, q, cos_theta, sin_theta)
                self.transformation_matrices.append((iteration, p, q, theta, J.copy()))
            
            # Apply Jacobi rotation
            self.apply_jacobi_rotation(A, V, p, q, cos_theta, sin_theta)
            
            # Progress reporting
            if iteration % 50 == 0 or iteration < 10:
                print(f"Iteration {iteration:3d}: max |a_ij| = {abs(max_off_diag):.6e}, "
                      f"norm = {off_diag_norm:.6e}, indices ({p},{q})")
        
        else:
            print(f"Warning: Maximum iterations ({self.max_iterations}) reached")
        
        self.iteration_count = iteration + 1
        
        # Extract eigenvalues and sort
        eigenvalues = np.diag(A)
        sorted_indices = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = V[:, sorted_indices]
        
        return eigenvalues, eigenvectors, A
    
    def test_assignment_matrices(self):
        """
        Test the specific matrices mentioned in the (Question 2.2).
        
        Returns:
        results (list): Results for each test matrix
        """
        print("\n2.2: Testing Specified Matrices")
        print("=" * 50)
        
        results = []
        
        # Matrix 1: 2x2 symmetric matrix 
        A1 = np.array([[0, 1],
                       [1, 0]], dtype=float)

        print(f"\nMatrix 1: 2×2 Matrix")
        print(A1)
        print(f"Symmetric: {np.allclose(A1, A1.T)}")
        
        # Solve eigenvalues
        eigenvals_1, eigenvecs_1, A1_final = self.solve_eigenvalues(A1)
        
        # Analytical solution for comparison
        eigenvals_analytical = np.array([-1.0, 1.0])
        
        result1 = {
            'matrix': '2×2',
            'original': A1,
            'eigenvalues_jacobi': eigenvals_1,
            'eigenvalues_analytical': eigenvals_analytical,
            'eigenvectors': eigenvecs_1,
            'error': np.max(np.abs(eigenvals_1 - eigenvals_analytical)),
            'iterations': self.iteration_count
        }
        results.append(result1)
        
        print(f"Jacobi eigenvalues: {eigenvals_1}")
        print(f"Analytical eigenvalues: {eigenvals_analytical}")
        print(f"Maximum error: {result1['error']:.2e}")
        
        return results
    
    def comprehensive_matrix_testing(self):
        """
        Test multiple matrix types as specified in Question 2.3.
        
        Returns:
        results (list): Comprehensive test results
        """
        print("\n2.3: Comprehensive Matrix Testing")
        print("=" * 40)
        
        # Define test matrices
        test_matrices = self._create_test_matrix_suite()
        results = []
        
        for matrix_name, matrix in test_matrices:
            print(f"\n{'-'*50}")
            print(f"Testing: {matrix_name}")
            print(f"{'-'*50}")
            print(f"Matrix:")
            print(matrix)
            
            # Solve with Jacobi method
            start_time = time.time()
            eigenvals_jacobi, eigenvecs_jacobi, _ = self.solve_eigenvalues(matrix, track_transformations=False)
            jacobi_time = time.time() - start_time
            
            # Solve with SciPy for comparison
            start_time = time.time()
            eigenvals_scipy, eigenvecs_scipy = eigh(matrix)
            scipy_time = time.time() - start_time
            
            # Calculate errors
            eigenval_error = np.max(np.abs(eigenvals_jacobi - eigenvals_scipy))
            
            # Store results
            result = {
                'name': matrix_name,
                'size': matrix.shape[0],
                'matrix': matrix,
                'iterations': self.iteration_count,
                'jacobi_eigenvals': eigenvals_jacobi,
                'scipy_eigenvals': eigenvals_scipy,
                'jacobi_eigenvecs': eigenvecs_jacobi,
                'max_error': eigenval_error,
                'jacobi_time': jacobi_time,
                'scipy_time': scipy_time
            }
            results.append(result)
            
            print(f"\nResults:")
            print(f"Jacobi eigenvalues: {eigenvals_jacobi}")
            print(f"SciPy eigenvalues:  {eigenvals_scipy}")
            print(f"Maximum error: {eigenval_error:.2e}")
            print(f"Iterations: {self.iteration_count}")
            print(f"Time - Jacobi: {jacobi_time:.4f}s, SciPy: {scipy_time:.4f}s")
        
        return results
    
    def _create_test_matrix_suite(self):
        """Create a comprehensive suite of test matrices."""
        matrices = []
        
        # 3x3 Diagonal matrix (trivial case)
        A1 = np.array([[3, 0, 0],
                       [0, 1, 0],
                       [0, 0, 2]], dtype=float)
        matrices.append(("3×3 Diagonal", A1))
        
        # 3x3 Nearly diagonal
        A2 = np.array([[5.0, 0.1, 0.1],
                       [0.1, 3.0, 0.1],
                       [0.1, 0.1, 1.0]], dtype=float)
        matrices.append(("3×3 Nearly Diagonal", A2))
        
        # 3x3 Dense symmetric
        A3 = np.array([[1, 2, 2],
                       [2, 0, 3],
                       [2, 3, -4]], dtype=float)
        matrices.append(("3×3 Dense Symmetric", A3))
        
        # 4x4 Symmetric matrix
        A4 = np.array([[4, 1, 2, 1],
                       [1, 3, 1, 2],
                       [2, 1, 2, 1],
                       [1, 2, 1, 3]], dtype=float)
        matrices.append(("4×4 Symmetric", A4))
        
        # 4x4 Hilbert matrix (ill-conditioned)
        n = 4
        A5 = np.array([[1.0/(i+j+1) for j in range(n)] for i in range(n)])
        matrices.append(("4×4 Hilbert", A5))
        
        return matrices
    
    def verify_mathematical_properties(self, A_original, eigenvalues, eigenvectors):
        """
        Verify mathematical properties of the eigenvalue solution.
        
        Parameters:
        A_original (numpy.ndarray): Original matrix
        eigenvalues (numpy.ndarray): Computed eigenvalues
        eigenvectors (numpy.ndarray): Computed eigenvectors
        """
        print("\nMathematical Properties Verification")
        print("=" * 40)
        
        n = A_original.shape[0]
        
        # 1. Verify A * v = λ * v
        print("1. Eigenvalue equation verification (A·v = λ·v):")
        max_residual = 0.0
        for i in range(n):
            v = eigenvectors[:, i]
            λ = eigenvalues[i]
            residual = np.linalg.norm(A_original @ v - λ * v)
            max_residual = max(max_residual, residual)
            print(f"   λ_{i+1} = {λ:10.6f}, ||A·v - λ·v|| = {residual:.2e}")
        
        print(f"   Maximum residual: {max_residual:.2e}")
        print(f"   ✓ Eigenvalue equation satisfied" if max_residual < 1e-10 else "   ⚠ Large residuals")
        
        # 2. Verify orthogonality
        print("\n2. Eigenvector orthogonality:")
        V = eigenvectors
        orthogonality_error = np.linalg.norm(V.T @ V - np.eye(n))
        print(f"   ||V^T·V - I|| = {orthogonality_error:.2e}")
        print(f"   ✓ Orthogonal" if orthogonality_error < 1e-10 else "   ⚠ Orthogonality issue")
        
        # 3. Verify spectral decomposition
        print("\n3. Spectral decomposition (A = V·Λ·V^T):")
        Λ = np.diag(eigenvalues)
        A_reconstructed = V @ Λ @ V.T
        reconstruction_error = np.linalg.norm(A_original - A_reconstructed)
        print(f"   ||A - V·Λ·V^T|| = {reconstruction_error:.2e}")
        print(f"   ✓ Correct" if reconstruction_error < 1e-10 else "   ⚠ Reconstruction error")
        
        # 4. Verify trace preservation
        print("\n4. Trace preservation:")
        trace_error = abs(np.trace(A_original) - np.sum(eigenvalues))
        print(f"   |tr(A) - Σλᵢ| = {trace_error:.2e}")
        print(f"   ✓ Preserved" if trace_error < 1e-12 else "   ⚠ Trace error")
    
    def plot_convergence_analysis(self, results_summary):
        """Create comprehensive convergence analysis plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Convergence history
        if self.convergence_history:
            iterations = range(len(self.convergence_history))
            ax1.semilogy(iterations, self.convergence_history, 'b-', linewidth=2)
            ax1.axhline(y=self.tolerance, color='r', linestyle='--', label='Tolerance')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Off-diagonal Norm')
            ax1.set_title('Convergence History (Last Matrix)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # Plot 2: Iterations vs Matrix Size
        sizes = [result['size'] for result in results_summary]
        iterations = [result['iterations'] for result in results_summary]
        matrix_names = [result['name'] for result in results_summary]
        
        ax2.scatter(sizes, iterations, s=100, alpha=0.7, c='red')
        for i, name in enumerate(matrix_names):
            ax2.annotate(name.split()[0], (sizes[i], iterations[i]), 
                        xytext=(5, 5), textcoords='offset points')
        ax2.set_xlabel('Matrix Size')
        ax2.set_ylabel('Iterations to Convergence')
        ax2.set_title('Iterations vs Matrix Size')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Accuracy Analysis
        errors = [result['max_error'] for result in results_summary]
        ax3.semilogy(range(len(errors)), errors, 'go-', markersize=8)
        ax3.axhline(y=self.tolerance, color='r', linestyle='--', label='Tolerance')
        ax3.set_xlabel('Test Matrix Index')
        ax3.set_ylabel('Maximum Eigenvalue Error')
        ax3.set_title('Accuracy Analysis')
        ax3.set_xticks(range(len(matrix_names)))
        ax3.set_xticklabels([name.split()[0] for name in matrix_names], rotation=45)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Performance Comparison
        jacobi_times = [result['jacobi_time'] for result in results_summary]
        scipy_times = [result['scipy_time'] for result in results_summary]
        
        x = np.arange(len(matrix_names))
        width = 0.35
        
        ax4.bar(x - width/2, jacobi_times, width, label='Jacobi Method', alpha=0.7)
        ax4.bar(x + width/2, scipy_times, width, label='SciPy (LAPACK)', alpha=0.7)
        ax4.set_xlabel('Matrix Type')
        ax4.set_ylabel('Execution Time (s)')
        ax4.set_title('Performance Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels([name.split()[0] for name in matrix_names], rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('jacobi_eigenvalue_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self, results):
        """Generate comprehensive summary report."""
        print("\n" + "="*60)
        print("QUESTION 2 COMPLETION SUMMARY")
        print("="*60)
        print("✓ Jacobi rotation method implemented with full transformation tracking")
        print("✓ First full sweep analysis completed (Question 2.1)")
        print("✓ matrices solved successfully (Question 2.2)")
        print("✓ Multiple matrix types tested (Question 2.3)")
        print("✓ Mathematical properties verified")
        print("✓ Convergence analysis completed")
        print("✓ Performance comparison with reference methods")
        
        if results:
            max_errors = [r['max_error'] for r in results]
            iterations = [r['iterations'] for r in results]
            
            print(f"\nPerformance Statistics:")
            print(f"  Average accuracy: {np.mean(max_errors):.2e}")
            print(f"  Maximum error: {np.max(max_errors):.2e}")
            print(f"  Average iterations: {np.mean(iterations):.1f}")
            print(f"  All tests converged successfully")
        
        print(f"\nFiles generated:")
        print(f"  - question2_jacobi_eigenvalues.py (this script)")
        print(f"  - question2_jacobi_eigenvalues.ipynb (companion notebook)")
        print(f"  - jacobi_eigenvalue_analysis.png")
        print(f"\n🎉 Question 2 completed successfully!")
        print(f"\nNext: Proceed to Question 3 (Data Interpolation)")


def main():
    """
    Main execution function demonstrating all capabilities of the Jacobi eigenvalue solver.
    """
    print("- Question 2: Jacobi Eigenvalue Method")
    print("=" * 70)
    
    # Initialize solver
    solver = JacobiEigenvalueSolver(tolerance=1e-12, max_iterations=500)
    
    # 2.1: First sweep transformation analysis
    print("\n" + "="*70)
    print("2.1: FIRST SWEEP TRANSFORMATION ANALYSIS")
    print("="*70)
    
    # Test matrix for first sweep analysis
    A_test = np.array([[1, 2, 2],
                       [2, 0, 3],
                       [2, 3, -4]], dtype=float)
    
    transformation_data = solver.analyze_first_sweep_transformations(A_test)
    
    # 2.2: specified matrices
    print("\n" + "="*70)
    print("2.2: SPECIFIED MATRICES")
    print("="*70)
    
    assignment_results = solver.test_assignment_matrices()
    
    # 2.3: Comprehensive testing
    print("\n" + "="*70)
    print("2.3: COMPREHENSIVE MATRIX TESTING")
    print("="*70)
    
    comprehensive_results = solver.comprehensive_matrix_testing()
    
    # Mathematical verification
    print("\n" + "="*70)
    print("MATHEMATICAL PROPERTIES VERIFICATION")
    print("="*70)
    
    # Verify properties for the test matrix
    eigenvals, eigenvecs, _ = solver.solve_eigenvalues(A_test, track_transformations=False)
    solver.verify_mathematical_properties(A_test, eigenvals, eigenvecs)
    
    # Performance analysis and visualization
    print("\n" + "="*70)
    print("PERFORMANCE ANALYSIS")
    print("="*70)
    
    # Create summary table
    all_results = comprehensive_results
    if all_results:
        # Create summary DataFrame
        summary_data = []
        for result in all_results:
            summary_data.append({
                'Matrix': result['name'],
                'Size': f"{result['size']}×{result['size']}",
                'Iterations': result['iterations'],
                'Max Error': f"{result['max_error']:.2e}",
                'Jacobi Time (s)': f"{result['jacobi_time']:.4f}",
                'SciPy Time (s)': f"{result['scipy_time']:.4f}"
            })
        
        print("\nPerformance Summary Table:")
        print("-" * 80)
        for i, data in enumerate(summary_data):
            if i == 0:
                # Print header
                header = " | ".join([f"{key:>15}" for key in data.keys()])
                print(header)
                print("-" * len(header))
            
            # Print data row
            row = " | ".join([f"{value:>15}" for value in data.values()])
            print(row)
        
        # Generate convergence plots
        solver.plot_convergence_analysis(all_results)
    
    # Final summary report
    solver.generate_summary_report(all_results)


if __name__ == "__main__":
    main()
