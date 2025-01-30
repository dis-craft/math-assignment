import streamlit as st
import random
import textwrap

st.title("Innovative Experiment Topics for Python")

topics = [
    "Simulating Projectile Motion and Optimizing Trajectories Using Python",
    "Predicting Expected Rainfall Using Probability Density Functions and Integration",
    "Estimating the Probability of Hitting a Target in 3D Space using Triple Integration (Uncertainty in Projectile Motion)"
]

st.markdown("### Select an experiment topic:")
selected_topic = st.selectbox("Choose a topic:", topics)

def generate_code(topic):
    """Generates unique Python code based on the selected topic."""
    if "Projectile Motion" in topic:
        code = textwrap.dedent(f"""
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Constants
        g = 9.81  # Gravity (m/s^2)
        angle = {random.randint(20, 70)}  # Random launch angle
        speed = {random.randint(10, 50)}  # Random speed (m/s)
        
        # Convert to radians
        theta = np.radians(angle)
        
        # Time of flight, max height, and range
        time_of_flight = (2 * speed * np.sin(theta)) / g
        max_height = (speed**2 * np.sin(theta)**2) / (2 * g)
        range_projectile = (speed**2 * np.sin(2 * theta)) / g
        
        # Plot trajectory
        t = np.linspace(0, time_of_flight, num=100)
        x = speed * np.cos(theta) * t
        y = speed * np.sin(theta) * t - 0.5 * g * t**2
        
        plt.plot(x, y)
        plt.xlabel('Distance (m)')
        plt.ylabel('Height (m)')
        plt.title('Projectile Motion Simulation')
        plt.show()
        """)
    
    elif "Rainfall" in topic:
        code = textwrap.dedent(f"""
        import numpy as np
        import scipy.stats as stats
        import matplotlib.pyplot as plt

        # Generate synthetic rainfall data
        mean_rainfall = {random.randint(50, 200)}
        std_dev = {random.randint(5, 30)}
        data = np.random.normal(mean_rainfall, std_dev, 1000)
        
        # Plot probability density function
        plt.hist(data, bins=30, density=True, alpha=0.6, color='b')
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mean_rainfall, std_dev)
        plt.plot(x, p, 'k', linewidth=2)
        
        plt.title('Predicted Rainfall Distribution')
        plt.xlabel('Rainfall (mm)')
        plt.ylabel('Density')
        plt.show()
        """)

    elif "Hitting a Target" in topic:
        code = textwrap.dedent(f"""
        import numpy as np
        from scipy.integrate import tplquad

        # Define probability density function
        def pdf(x, y, z):
            return np.exp(-{random.randint(1, 5)} * (x**2 + y**2 + z**2))

        # Triple integration over 3D space
        result, _ = tplquad(pdf, 0, 10, lambda x: 0, lambda x: 10, lambda x, y: 0, lambda x, y: 10)
        
        print(f"Probability of hitting target in 3D space: {{result:.5f}}")
        """)

    else:
        code = "# No code available for this topic."
    
    return code

# Generate a different code snippet for each user
if st.button("Generate Code"):
    generated_code = generate_code(selected_topic)
    st.code(generated_code, language="python")

    # Provide download link
    file_name = "experiment_code.py"
    with open(file_name, "w") as f:
        f.write(generated_code)

    with open(file_name, "rb") as f:
        st.download_button("Download Python Code", f, file_name, "text/x-python")

