import numpy as np
import matplotlib.pyplot as plt
import os
import csv

pi = np.pi


def cheb(N):
    s = np.cos(pi * np.arange(N + 1) / N)[::-1]
    c = np.ones(N + 1)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * ((-1) ** np.arange(N + 1))

    S = np.tile(s, (N + 1, 1))
    dS = S - S.T

    D_s = np.outer(c, 1.0 / c) / (dS + np.eye(N + 1))
    D_s = D_s - np.diag(np.sum(D_s, axis=1))
    return D_s, s


def g5(x):
    return (pi**4 + 4 * pi**2 + 3.0) * np.cos(pi * x) + 3.0


def solve_problem5(N):
    D, x = cheb(N)
    I = np.eye(N + 1)
    D2 = D @ D
    D4 = D2 @ D2
    L = D4 - 4.0 * D2 + 3.0 * I
    A = L.copy()
    b = g5(x)

    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = 0.0
    A[1, :] = D[0, :]
    b[1] = 0.0
    A[N - 1, :] = D[N, :]
    b[N - 1] = 0.0
    A[N, :] = 0.0
    A[N, N] = 1.0
    b[N] = 0.0

    u_num = np.linalg.solve(A, b)
    u_exact = np.cos(pi * x) + 1.0
    err = np.abs(u_num - u_exact)
    return x, u_num, u_exact, err


def g6_x(x):
    x2 = x**2
    term_u4 = (120 * x2**2 - 240 * x2 + 24) / (1 + x2) ** 5
    term_u1 = -2 * x / (1 + x2) ** 2
    term_u0 = 1 / (1 + x2)
    return term_u4 + term_u1 + term_u0


def solve_problem6(N):
    D_s, s = cheb(N)
    I = np.eye(N + 1)
    s = s[::-1]

    x = 2.5 * (s + 1.0)
    ds_dx = 2.0 / 5.0

    D_x = ds_dx * D_s
    D_x2 = np.linalg.matrix_power(D_x, 2)
    D_x4 = D_x2 @ D_x2

    L_x = D_x4 + D_x + I
    A = L_x.copy()
    b = g6_x(x)

    u_at_0 = 1.0
    u_prime_at_0 = 0.0
    u_at_5 = 1.0 / 26.0
    u_prime_at_5 = -10.0 / 676.0

    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = u_at_5
    A[1, :] = D_x[0, :]
    b[1] = u_prime_at_5

    A[N - 1, :] = D_x[N, :]
    b[N - 1] = u_prime_at_0
    A[N, :] = 0.0
    A[N, N] = 1.0
    b[N] = u_at_0

    u_num = np.linalg.solve(A, b)

    u_exact = 1.0 / (1.0 + x**2)
    err = np.abs(u_num - u_exact)

    return x, u_num, u_exact, err


def run_experiments():
    out_dir = "./cheb_outputs"
    os.makedirs(out_dir, exist_ok=True)

    N_list = [8, 12, 16, 24, 32, 48]
    conv5, conv6 = [], []

    for N in N_list:
        x5, u5num, u5ex, err5 = solve_problem5(N)
        maxerr5 = np.max(err5)
        conv5.append((N, maxerr5))
        plt.figure(figsize=(7, 5))
        plt.plot(x5, u5ex, "k-", lw=2, label="Exact")
        plt.plot(x5, u5num, "ro--", ms=4, label="Numerical")
        plt.title(f"Problem 5: Solution N={N}")
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(out_dir, f"prob5_solution_N{N}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        plt.figure(figsize=(7, 5))
        plt.semilogy(x5, err5)
        plt.title(f"Problem 5: Error N={N} (max={maxerr5:.2e})")
        plt.xlabel("x")
        plt.ylabel("Absolute Error")
        plt.grid(True)
        plt.ylim(1e-16, 1)
        plt.savefig(
            os.path.join(out_dir, f"prob5_error_N{N}.png"), dpi=150, bbox_inches="tight"
        )
        plt.close()

        x6, u6num, u6ex, err6 = solve_problem6(N)
        maxerr6 = np.max(err6)
        conv6.append((N, maxerr6))
        plt.figure(figsize=(8, 6))
        plt.plot(x6, u6ex, "k-", lw=2, label="Exact")
        plt.plot(x6, u6num, "bo--", ms=5, label="Numerical")
        plt.title(f"Problem 6: Solution N={N}")
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(out_dir, f"prob6_solution_N{N}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        plt.figure(figsize=(7, 5))
        plt.semilogy(x6, err6)
        plt.title(f"Problem 6: Error N={N} (max={maxerr6:.2e})")
        plt.xlabel("x")
        plt.ylabel("Absolute Error")
        plt.grid(True)
        plt.ylim(1e-16, 1)
        plt.savefig(
            os.path.join(out_dir, f"prob6_error_N{N}.png"), dpi=150, bbox_inches="tight"
        )
        plt.close()

    with open(os.path.join(out_dir, "conv_problem5.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "max_error"])
        writer.writerows(conv5)
    with open(os.path.join(out_dir, "conv_problem6.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "max_error"])
        writer.writerows(conv6)


def run_error_vs_N():
    out_dir = "./cheb_outputs"
    os.makedirs(out_dir, exist_ok=True)

    N_values = np.arange(2, 101)
    error5_list = []
    error6_list = []

    for N in N_values:
        x5, u5num, u5ex, err5 = solve_problem5(N)
        error5_list.append(np.max(err5))
        x6, u6num, u6ex, err6 = solve_problem6(N)
        error6_list.append(np.max(err6))

    plt.figure(figsize=(8, 5))
    plt.semilogy(N_values, error5_list, "r-o", lw=1.5, ms=4, label="Problem 5")
    plt.semilogy(N_values, error6_list, "b-s", lw=1.5, ms=4, label="Problem 6")
    plt.xlabel("N")
    plt.ylabel("Max Absolute Error")
    plt.title("Convergence: N vs Max Error")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "N_vs_error.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    run_experiments()
    run_error_vs_N()
