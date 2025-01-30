import streamlit as st
import random
import os
import time
import matplotlib.pyplot as plt
import sys
import io
import numpy as np
import scipy.stats as stats

def generate_code(question):
    """Returns a random Python solution based on the question."""
    solutions = {
        "Simulating Projectile Motion": """
import numpy as np
import matplotlib.pyplot as plt

def projectile_motion(angle, speed):
    g = 9.81
    angle = np.radians(angle)
    t_flight = 2 * speed * np.sin(angle) / g
    t = np.linspace(0, t_flight, num=100)
    x = speed * np.cos(angle) * t
    y = speed * np.sin(angle) * t - 0.5 * g * t**2
    plt.plot(x, y)
    plt.xlabel('Distance')
    plt.ylabel('Height')
    plt.title('Projectile Motion')
    plt.show()

projectile_motion(45, 20)
        """,
        "Predicting Expected Rainfall": """
import numpy as np
import scipy.stats as stats

def expected_rainfall(mean, std_dev):
    x = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 100)
    y = stats.norm.pdf(x, mean, std_dev)
    return x[np.argmax(y)]

print("Expected Rainfall: ", expected_rainfall(100, 20), "mm")
        """
    }
    return solutions.get(question, "# Solution not available")

def execute_user_code(user_code):
    """Executes user code safely and captures output."""
    old_stdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()
    try:
        exec(user_code, {})
        output = mystdout.getvalue()
        return output, None
    except Exception as e:
        return None, str(e)
    finally:
        sys.stdout = old_stdout

st.title("🚀 Python Experiment Learning App 🚀")

st.sidebar.title("💡 Fun Zone")
if st.sidebar.button("Surprise me!"):
    with st.spinner("Loading something cool..."):
        time.sleep(2)
    st.sidebar.success("🎉 Keep learning, you're doing great!")

option = st.radio("What do you want to do?", ["I want to learn", "I'm too lazy"])

questions = [
    "Simulating Projectile Motion",
    "Predicting Expected Rainfall"
]

if option == "I want to learn":
    question = st.selectbox("Select a question to practice:", questions)
    user_code = st.text_area("Write your Python code here:")
    
    if st.button("Run Code"):
        if user_code.strip():
            output, error = execute_user_code(user_code)
            if error:
                st.error(f"❌ Error: {error}")
            else:
                st.success("✅ Code executed successfully!")
                st.text_area("Output:", output, height=150)
        else:
            st.warning("⚠️ Please enter some code to run!")
    
    if st.button("Show Solution"):
        st.code(generate_code(question))
    
    st.markdown("### 🤔 Quick Quiz!")
    quiz_question = "What does the projectile motion equation depend on?"
    options = ["Mass of object", "Gravity and initial velocity", "Temperature"]
    answer = st.radio(quiz_question, options)
    if st.button("Check Answer"):
        if answer == "Gravity and initial velocity":
            st.success("✅ Correct!")
        else:
            st.error("❌ Incorrect. Try again!")

elif option == "I'm too lazy":
    question = random.choice(questions)
    solution_code = generate_code(question)
    file_path = "solution.py"
    
    with open(file_path, "w") as f:
        f.write(solution_code)
    
    st.write(f"You got: {question}")
    st.code(solution_code)
    st.download_button("📥 Download Solution", file_path, file_name="solution.py")
    
    os.remove(file_path)
    
    fun_facts = [
        "Python was named after Monty Python, not the snake! 🐍",
        "The first version of Python was released in 1991.",
        "You can use Python to build AI, games, and even automate tasks! 🤖"
    ]
    st.sidebar.markdown(f"**💡 Fun Fact:** {random.choice(fun_facts)}")
