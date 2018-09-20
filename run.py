# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 21:15:45 2018

@author: ahmer
"""

from numpy import *

#y = mx + b
# m is slope and b is y_intercept
def compute_error_for_line_given_points(b, m, points):
    totalerror = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        
        totalerror += (y - (m*x + b))**2
        return totalerror / float(len(points))

def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]
    

def gradient_descent_runner(points, starting_m, starting_b, learning_rate, num_iterations):
    m = starting_m
    b = starting_b
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]    


def run():
    points = genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_m = 0
    initial_b = 0
    num_iterations = 1000
    print ("Starting Gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print ("...")
    [b, m] = gradient_descent_runner(points, initial_m, initial_b, learning_rate, num_iterations)
    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))
       
    

if __name__ == "__main__":
    run()