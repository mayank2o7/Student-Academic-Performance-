#                                                                   MATH PROJECT

'''
INFORMATIONS

Project : Student Academic Performance Analysis Using Matrices in Python

students = [f"S(i)" for i in range(1, 31)]

subjects = [
    "Math",
    "Digital Logic",
    "DSA",
    "Computer Networks",
    "Operating System",
    "CIE-1"
]

Dataset :
| Student | Math | Digital Logic | DSA | CN | OS | CIE-1 |
| ------- | ---: | ------------: | --: | -: | --: | ----: |
| S1      |   82 |            76 |  88 | 79 | 84 |    81 |
| S2      |   74 |            81 |  79 | 85 | 77 |    80 |
| S3      |   91 |            89 |  94 | 88 | 92 |    90 |
| S4      |   68 |            72 |  75 | 70 | 73 |    71 |
| S5      |   85 |            83 |  81 | 86 | 88 |    84 |
| S6      |   77 |            69 |  73 | 78 | 75 |    76 |
| S7      |   93 |            91 |  89 | 95 | 90 |    94 |
| S8      |   71 |            75 |  68 | 73 | 70 |    74 |
| S9      |   88 |            84 |  91 | 87 | 85 |    89 |
| S10     |   79 |            82 |  76 | 80 | 83 |    78 |
| S11     |   66 |            70 |  72 | 68 | 71 |    69 |
| S12     |   90 |            87 |  93 | 91 | 89 |    92 |
| S13     |   81 |            78 |  85 | 82 | 80 |    83 |
| S14     |   73 |            77 |  71 | 75 | 78 |    72 |
| S15     |   86 |            90 |  88 | 84 | 87 |    89 |
| S16     |   69 |            65 |  74 | 71 | 68 |    70 |
| S17     |   95 |            93 |  97 | 92 | 94 |    96 |
| S18     |   78 |            80 |  82 | 77 | 79 |    81 |
| S19     |   84 |            86 |  80 | 89 | 85 |    87 |
| S20     |   72 |            74 |  77 | 69 | 73 |    75 |
| S21     |   89 |            92 |  90 | 88 | 91 |    93 |
| S22     |   76 |            71 |  79 | 74 | 77 |    73 |
| S23     |   83 |            85 |  87 | 81 | 86 |    84 |
| S24     |   67 |            73 |  69 | 72 | 65 |    70 |
| S25     |   92 |            88 |  95 | 90 | 93 |    91 |
| S26     |   80 |            79 |  83 | 85 | 81 |    82 |
| S27     |   75 |            68 |  78 | 76 | 74 |    77 |
| S28     |   87 |            91 |  86 | 89 | 88 |    90 |
| S29     |   70 |            76 |  73 | 71 | 69 |    74 |
| S30     |   94 |            90 |  92 | 96 | 95 |    93 |

Operations:

Part 1: Matrix representation
30 students × 6 subjects.

Part 2: Transpose
Change student-wise data into subject-wise representation.

Part 3: Subject-wise average
Find the average mark for each subject.

Part 4: Matrix multiplication
Use matrix multiplication with subject credits to calculate the SGPA of each student.

Part 5: Matrix addition
Apply an academic bonus to every mark.

Part 6: Matrix subtraction
Compare marks against a target value.

Part 7: Scalar multiplication
Demonstrate scaling of the complete dataset.
'''

import numpy as np

students = [f"S{i}" for i in range(1, 31)]

subjects = [
    "Math",
    "Digital Logic",
    "DSA",
    "Computer Networks",
    "Operating System",
    "CIE-1"
]


# --------------------------------------------------
# 1. MATRIX REPRESENTATION
# --------------------------------------------------

# Create the original marks matrix.
# Rows represent students and columns represent subjects.

marks = np.array([
    [82, 76, 88, 79, 84, 81],
    [74, 81, 79, 85, 77, 80],
    [91, 89, 94, 88, 92, 90],
    [68, 72, 75, 70, 73, 71],
    [85, 83, 81, 86, 88, 84],
    [77, 69, 73, 78, 75, 76],
    [93, 91, 89, 95, 90, 94],
    [71, 75, 68, 73, 70, 74],
    [88, 84, 91, 87, 85, 89],
    [79, 82, 76, 80, 83, 78],
    [66, 70, 72, 68, 71, 69],
    [90, 87, 93, 91, 89, 92],
    [81, 78, 85, 82, 80, 83],
    [73, 77, 71, 75, 78, 72],
    [86, 90, 88, 84, 87, 89],
    [69, 65, 74, 71, 68, 70],
    [95, 93, 97, 92, 94, 96],
    [78, 80, 82, 77, 79, 81],
    [84, 86, 80, 89, 85, 87],
    [72, 74, 77, 69, 73, 75],
    [89, 92, 90, 88, 91, 93],
    [76, 71, 79, 74, 77, 73],
    [83, 85, 87, 81, 86, 84],
    [67, 73, 69, 72, 65, 70],
    [92, 88, 95, 90, 93, 91],
    [80, 79, 83, 85, 81, 82],
    [75, 68, 78, 76, 74, 77],
    [87, 91, 86, 89, 88, 90],
    [70, 76, 73, 71, 69, 74],
    [94, 90, 92, 96, 95, 93]
])

print("Matrix Shape:", marks.shape)

print("\nOriginal Marks Matrix:")
print(marks)


# --------------------------------------------------
# 2. TRANSPOSE
# --------------------------------------------------

# Transpose changes rows into columns.
# The 30 × 6 student-wise matrix becomes a 6 × 30 subject-wise matrix.

transposed_matrix = marks.T

print("\nTransposed Matrix:")
print(transposed_matrix)

print("\nOriginal Shape:", marks.shape)
print("Transposed Shape:", transposed_matrix.shape)


# --------------------------------------------------
# 3. SUBJECT-WISE AVERAGE
# --------------------------------------------------

# axis=0 calculates the average down each column.
# Therefore, one average is obtained for each subject.

subject_average = marks.mean(axis=0)

print("\nSubject-wise Average:")
for i in range(len(subjects)):
    print(subjects[i], ":", round(subject_average[i], 2))


# --------------------------------------------------
# 4. SGPA CALCULATION USING MATRIX MULTIPLICATION
# --------------------------------------------------

# Credits of each subject.
# The order matches the subjects list:
# Math, Digital Logic, DSA, CN, OS, CIE-1

credits = np.array([
    [4],
    [5],
    [5],
    [4],
    [4],
    [2]
])

# Convert marks into grade points using the grading scale.
grade_points = np.where(
    marks >= 90, 10,
    np.where(
        marks >= 80, 9,
        np.where(
            marks >= 70, 8,
            np.where(
                marks >= 60, 7,
                np.where(
                    marks >= 50, 6,
                    np.where(marks >= 40, 5, 0)
                )
            )
        )
    )
)

print("\nGrade Point Matrix:")
print(grade_points)

# Matrix multiplication:
# Each student's grade-point row is multiplied by the credit column.
credit_points = grade_points @ credits

# Total credits used for SGPA calculation.
total_credits = np.sum(credits)

# SGPA = total weighted grade points / total credits.
sgpa = credit_points / total_credits

print("\nSGPA of Students:")

for i in range(30):
    print(students[i], ":", round(sgpa[i][0], 2))


# --------------------------------------------------
# 5. MATRIX ADDITION
# --------------------------------------------------

# Assume every student receives a 2-mark academic bonus.
bonus = np.full((30, 6), 2)

adjusted_marks = marks + bonus

print("\nMarks After Adding 2-Mark Bonus:")
print(adjusted_marks)


# --------------------------------------------------
# 6. MATRIX SUBTRACTION
# --------------------------------------------------

# Compare every mark with a target of 75.
target = np.full((30, 6), 75)

difference = marks - target

print("\nDifference From Target (75):")
print(difference)


# --------------------------------------------------
# 7. SCALAR MULTIPLICATION
# --------------------------------------------------

# Scale every mark to 80% of its original value.
scaled_marks = marks * 0.8

print("\nMarks After Scalar Multiplication by 0.8:")
print(scaled_marks)
