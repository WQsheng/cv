Impliment the function HoughLines() using Cython(python code with c-type objects).

Because the opencv HoughLines function only returns the lines greater than threshold, this function here returns all the lines 
detected and their accumulation, which may represent the length of the line.

usage:
from houghline import HoughLines
np_array_lines = HoughLines(binary_image, rho=1, theta=np.pi/180)

the function returns a 2-D numpy array(shape=n*3).   
[[line_rho1, line_theta1, line_length1],   
 [line_rho2, line_theta2, line_length2],    
 ...     
 [line_rho_n, line_theta_n, line_length_n]]     
 
np_array_lines[np_array_lines[:, 2] > threshold] just returns the lines greater than threshold, like the opencv HoughLines
function does.

Because it's written main in python, the total running time is about 2 times longer than the opencv function.
