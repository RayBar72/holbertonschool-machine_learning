#!/usr/bin/env python3
'''
Function that plot a histogram of student scores for a project:
    The x-axis should be labeled Grades
    The y-axis should be labeled Number of Students
    The x-axis should have bins every 10 units
    The title should be Project A
    The bars should be outlined in black
'''
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.hist(student_grades, bins=range(10, 101, 10), edgecolor='black')
plt.xlabel('Grades')
plt.xlim(0, 100)
plt.xticks(range(0, 101, 10))
plt.ylabel('Number of Students')
plt.ylim(0, 30)
plt.title('Project A')
plt.show()
