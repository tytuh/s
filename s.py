import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def grad_step_search(f, x0, alpha=0.01, epsilon=1e-5, max_iter=1000):
    x = x0
    grad = grad_f(x)
    iter = 0
    while abs(np.linalg.norm(grad)) > epsilon and iter < max_iter:
        x = x - alpha * grad
        grad = grad_f(x)
        iter += 1
    return x

def grad_f(x):
    return -np.cos(x)

def f(x):
    return np.sin(x)

st.title("Gradient Step Search")
st.write("Find the minimum of the sin function")

x0 = st.number_input("Initial x:", min_value=-10, max_value=10, value=0)
alpha = st.number_input("Step size (alpha):", min_value=0.01, max_value=1.0, step=0.01, value=0.01)
epsilon = st.number_input("Precision (epsilon):", min_value=1e-5, max_value=1e-3, step=1e-5, value=1e-5)

x_min = grad_step_search(f, x0, alpha, epsilon)

st.write("Minimum found at x = %.5f" % x_min)

x = np.linspace(-np.pi, np.pi, 100)
y = f(x)

plt.plot(x, y)
plt.scatter(x_min, f(x_min), c='r')
st.pyplot()