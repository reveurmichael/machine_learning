# Pickling in Python: Saving and Loading Data and Models

Pickling is a powerful technique in Python that allows you to serialize and deserialize Python objects. This means you can save your data structures or trained machine learning models to disk and load them back later. This tutorial will guide you through using pickle for both regular data and ML models.

## Videos on youtube:
- https://www.youtube.com/watch?v=6Q56r_fVqgw

## What is Pickling?

Pickling refers to the process of converting a Python object into a byte stream (serialization) that can be saved to a file. "Unpickling" is the reverse process, where the byte stream is converted back into a Python object (deserialization).

## Part 1: Pickling Regular Python Data Structures

Let's start with the basics - pickling common Python data structures like lists, dictionaries, and custom objects.

```python
import pickle

# Sample data to pickle
favorite_fruits = ['apple', 'banana', 'mango', 'strawberry']
student_scores = {
    'Alice': 92,
    'Bob': 85,
    'Charlie': 78,
    'Diana': 95
}
complex_data = {
    'fruits': favorite_fruits,
    'scores': student_scores,
    'metrics': [1.2, 3.4, 5.6, 7.8],
    'active': True
}

# Saving data using pickle
with open('fruits.pkl', 'wb') as file:  # Note the 'wb' mode - binary writing
    pickle.dump(favorite_fruits, file)

with open('scores.pkl', 'wb') as file:
    pickle.dump(student_scores, file)

with open('complex_data.pkl', 'wb') as file:
    pickle.dump(complex_data, file)

print("Data saved successfully!")
```

Now let's see how we can load this data back:

```python
# Loading data using pickle
with open('fruits.pkl', 'rb') as file:  # Note the 'rb' mode - binary reading
    loaded_fruits = pickle.load(file)

with open('scores.pkl', 'rb') as file:
    loaded_scores = pickle.load(file)

with open('complex_data.pkl', 'rb') as file:
    loaded_complex_data = pickle.load(file)

print("Loaded fruits:", loaded_fruits)
print("Loaded scores:", loaded_scores)
print("Loaded complex data:", loaded_complex_data)
```

## Part 2: Understanding Pickle with a Student Grades Example

Let's create a more practical example that demonstrates the power of pickling with a student grade tracker:

```python
import pickle
from datetime import datetime

class GradeTracker:
    def __init__(self, course_name):
        self.course_name = course_name
        self.student_grades = {}
        self.last_updated = datetime.now()
    
    def add_grade(self, student_name, grade):
        self.student_grades[student_name] = grade
        self.last_updated = datetime.now()
    
    def get_average(self):
        if not self.student_grades:
            return 0
        return sum(self.student_grades.values()) / len(self.student_grades)
    
    def get_top_student(self):
        if not self.student_grades:
            return None
        return max(self.student_grades.items(), key=lambda x: x[1])
    
    def __str__(self):
        return f"GradeTracker for '{self.course_name}' with {len(self.student_grades)} students"

# Create and use our grade tracker
ml_course = GradeTracker("Introduction to Machine Learning")

# Add student grades
ml_course.add_grade("Alice", 95)
ml_course.add_grade("Bob", 87)
ml_course.add_grade("Charlie", 91)
ml_course.add_grade("Diana", 84)
ml_course.add_grade("Eve", 78)

print(ml_course)
print(f"Class average: {ml_course.get_average():.1f}")
top_student, top_grade = ml_course.get_top_student()
print(f"Top student: {top_student} with grade {top_grade}")

# Save our grade tracker object
with open('ml_course_grades.pkl', 'wb') as file:
    pickle.dump(ml_course, file)

print(f"\nGrade data for '{ml_course.course_name}' saved successfully!")
```

Now, imagine we need to continue tracking grades in the future:

```python
# Load our previous grade data
with open('ml_course_grades.pkl', 'rb') as file:
    loaded_grades = pickle.load(file)

print(loaded_grades)
print(f"Last updated: {loaded_grades.last_updated}")
print(f"Current class average: {loaded_grades.get_average():.1f}")

# Add new student grades
loaded_grades.add_grade("Frank", 89)
loaded_grades.add_grade("Grace", 92)
loaded_grades.add_grade("Hannah", 79)

print(f"\nUpdated class average with {len(loaded_grades.student_grades)} students: {loaded_grades.get_average():.1f}")
top_student, top_grade = loaded_grades.get_top_student()
print(f"New top student: {top_student} with grade {top_grade}")

# Save the updated data
with open('ml_course_grades_updated.pkl', 'wb') as file:
    pickle.dump(loaded_grades, file)
```

## Part 3: Pickling Machine Learning Models (scikit-learn Linear Regression)

Now, let's use pickle to save and load a scikit-learn model. We'll train a simple linear regression model to predict student performance based on study hours.

```python
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic student data
np.random.seed(42)

# Study hours per week (between a 1 and 25 hours)
study_hours = np.random.uniform(1, 25, 30).reshape(-1, 1)

# Test scores (base score of 50 plus 2 points per study hour with some random variation)
test_scores = 50 + 2 * study_hours + np.random.normal(0, 5, 30).reshape(-1, 1)

# Create and train a linear regression model
model = LinearRegression()
model.fit(study_hours, test_scores)

# Print model coefficients
print(f"Model trained!")
print(f"Coefficient: {model.coef_[0][0]:.2f} (score points per study hour)")
print(f"Intercept: {model.intercept_[0]:.2f} (base score)")

# Make predictions
predictions = model.predict(study_hours)

# Calculate metrics
mse = mean_squared_error(test_scores, predictions)
r2 = r2_score(test_scores, predictions)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(study_hours, test_scores, color='blue', label='Actual scores')
plt.plot(study_hours, predictions, color='red', linewidth=2, label='Predicted scores')
plt.xlabel('Study Hours per Week')
plt.ylabel('Test Score')
plt.title('Study Hours vs. Test Scores')
plt.legend()
plt.grid(True)
plt.savefig('study_hours_model.png')
plt.close()

# Save the trained model using pickle
with open('study_hours_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved successfully!")
```

Later, when a teacher wants to predict student scores:

```python
# Load the trained model
with open('study_hours_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# New student study hours to evaluate
new_study_hours = np.array([5, 10, 15, 20]).reshape(-1, 1)

# Predict scores with loaded model
predicted_scores = loaded_model.predict(new_study_hours)

print("Study Hours | Predicted Score")
print("-" * 30)
for hours, score in zip(new_study_hours.flatten(), predicted_scores.flatten()):
    print(f"{hours:11.1f} | {score:.1f}")

# Create a prediction function that teachers can use easily
def predict_score(hours_studied):
    """Predict a student's score based on hours studied."""
    if hours_studied < 0:
        return "Invalid input: study hours cannot be negative"
    
    # Reshape for the model (which expects a 2D array)
    hours = np.array([[hours_studied]])
    
    # Predict and return the score
    predicted = loaded_model.predict(hours)[0][0]
    return predicted

# Try out the prediction function
print("\nPrediction function test:")
print(f"If a student studies for 12.5 hours, predicted score: {predict_score(12.5):.1f}")
print(f"If a student studies for 7.8 hours, predicted score: {predict_score(7.8):.1f}")
print(f"If a student studies for 22 hours, predicted score: {predict_score(22):.1f}")
```

## Part 4: Using joblib Instead of pickle

The `joblib` library provides an alternative to pickle that's more efficient for machine learning models, especially when working with numpy arrays. Let's see how to use it with our linear regression model:

```python
import joblib

# Save the model using joblib
joblib.dump(model, 'study_hours_model.joblib')
print("Model saved using joblib!")

# Load the joblib model
loaded_model_joblib = joblib.load('study_hours_model.joblib')

# Test that it works the same as our pickle-loaded model
test_hours = np.array([[17.5]])
pickle_prediction = loaded_model.predict(test_hours)[0][0]
joblib_prediction = loaded_model_joblib.predict(test_hours)[0][0]

print(f"Prediction using pickle-loaded model: {pickle_prediction:.1f}")
print(f"Prediction using joblib-loaded model: {joblib_prediction:.1f}")
print(f"Same prediction: {pickle_prediction == joblib_prediction}")
```

## Conclusion

In this tutorial, we've covered:

1. Basics of pickling Python objects
2. Creating and pickling custom objects
3. Saving and loading scikit-learn linear regression models with pickle
4. Using joblib as an alternative for machine learning models

Pickling is an essential tool for machine learning practitioners, allowing you to save your trained models for later use without having to retrain them each time. This is particularly useful when you want to deploy models in production or share them with others.

Remember that pickled files should only be loaded from trusted sources, as they can potentially execute arbitrary code during unpickling.

