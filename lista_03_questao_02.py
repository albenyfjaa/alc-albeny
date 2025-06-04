import numpy as np

def crossprod(u, v):
    """
    Computes the cross product of two 3D vectors u and v.

    Args:
        u: A 1D NumPy array or list of length 3 representing the first vector.
        v: A 1D NumPy array or list of length 3 representing the second vector.

    Returns:
        A 1D NumPy array of length 3 representing the cross product u x v.
        
    Raises:
        ValueError: If the input vectors are not 3-dimensional.
    """
    u_arr = np.asarray(u)
    v_arr = np.asarray(v)
    
    if u_arr.shape != (3,) or v_arr.shape != (3,):
        raise ValueError("Both vectors must be 3-dimensional.")
        
    # u = [u1, u2, u3] -> u_arr[0], u_arr[1], u_arr[2]
    # v = [v1, v2, v3] -> v_arr[0], v_arr[1], v_arr[2]
    
    # c1 = u2*v3 - u3*v2
    c1 = u_arr[1] * v_arr[2] - u_arr[2] * v_arr[1]
    # c2 = u3*v1 - u1*v3  
    c2 = u_arr[2] * v_arr[0] - u_arr[0] * v_arr[2]
    # c3 = u1*v2 - u2*v1
    c3 = u_arr[0] * v_arr[1] - u_arr[1] * v_arr[0]
    
    return np.array([c1, c2, c3])

# For the dot product, we can use NumPy's np.dot function
# If you need to implement it:
# def dotprod(a, b):
#   return np.sum(np.asarray(a) * np.asarray(b))

# Define vectors u and v
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

print(f"Vector u = {u}")
print(f"Vector v = {v}")
print("-" * 30)

# 1. Compute u x v
u_cross_v = crossprod(u, v)
print(f"u x v = {u_cross_v}")

# 2. Compute v x u
v_cross_u = crossprod(v, u)
print(f"v x u = {v_cross_u}")
print("(Note: v x u should be -(u x v))")
print("-" * 30)

# 3. Compute (u x v) . u
# The dot product of (u x v) with u should be 0, as u_cross_v is orthogonal to u.
dot_ucrossv_u = np.dot(u_cross_v, u)
print(f"(u x v) . u = {dot_ucrossv_u}")
print("-" * 30)

# 4. Compute (v x u) . v
# The dot product of (v x u) with v should be 0, as v_cross_u is orthogonal to v.
dot_vcrossu_v = np.dot(v_cross_u, v)
print(f"(v x u) . v = {dot_vcrossu_v}")
